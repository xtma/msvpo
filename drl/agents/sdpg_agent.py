import torch
from drl.models.mlp import MuMlpModel, ZofMuMlpModel
from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.distributions.gaussian import DistInfo, Gaussian
# from drl.models.nc_mlp import NonCrossZofMuMlpModel
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.nn.parallel import DistributedDataParallelCPU as DDPC  # Deprecated

AgentInfo = namedarraytuple("AgentInfo", ["mu"])


class SdpgAgent(BaseAgent):
    """Agent for deep deterministic policy gradient algorithm."""

    shared_mu_model = None

    def __init__(
        self,
        ModelCls=MuMlpModel,  # Mu model.
        ZModelCls=ZofMuMlpModel,
        model_kwargs=None,  # Mu model.
        z_model_kwargs=None,
        initial_model_state_dict=None,  # Mu model.
        initial_z_model_state_dict=None,
        action_std=0.1,
        action_noise_clip=None,
        num_quantiles=64,
    ):
        """Saves input arguments; default network sizes saved here."""
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[400, 300])
        if z_model_kwargs is None:
            z_model_kwargs = dict(hidden_sizes=[400, 300])
        save__init__args(locals())
        super().__init__()  # For async setup.

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        """Instantiates mu and q, and target_mu and target_q models."""
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)
        self.z_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        if self.initial_z_model_state_dict is not None:
            self.z_model.load_state_dict(self.initial_z_model_state_dict)
        self.target_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
        self.target_z_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        self.target_z_model.load_state_dict(self.z_model.state_dict())
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            std=self.action_std,
            noise_clip=self.action_noise_clip,
            clip=env_spaces.action.high[0],  # Assume symmetric low=-high.
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)  # Takes care of self.model.
        self.target_model.to(self.device)
        self.z_model.to(self.device)
        self.target_z_model.to(self.device)

    def data_parallel(self):
        device_id = super().data_parallel()  # Takes care of self.model.
        self.z_model = DDP(
            self.z_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        return device_id

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    @torch.no_grad()
    def tau(self, observation, prev_action, prev_reward, action):
        tau_shape = (*action.shape[:-1], self.num_quantiles)
        presum_tau = torch.rand(tau_shape, device=self.device)
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        tau = torch.cumsum(presum_tau, dim=-1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau)
            tau_hat[..., 0:1] = tau[..., 0:1] / 2.
            tau_hat[..., 1:] = (tau[..., 1:] + tau[..., :-1]) / 2.
        return tau_hat, presum_tau

    def z(self, observation, prev_action, prev_reward, action, tau):
        """Compute Q-value for input state/observation and action (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, action, tau), device=self.device)
        z = self.z_model(*model_inputs)
        # z = buffer_to(z, device='cpu')
        return z

    def z_at_mu(self, observation, prev_action, prev_reward, tau):
        """Compute Q-value for input state/observation, through the mu_model
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mu = self.model(*model_inputs)
        tau = buffer_to(tau, device=self.device)
        z = self.z_model(*model_inputs, mu, tau)
        # z = buffer_to(z, device='cpu')
        return z

    def target_z_at_mu(self, observation, prev_action, prev_reward, tau):
        """Compute target Q-value for input state/observation, through the
        target mu_model."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        target_mu = self.target_model(*model_inputs)
        tau = buffer_to(tau, device=self.device)
        target_z_at_mu = self.target_z_model(*model_inputs, target_mu, tau)
        # target_z_at_mu = buffer_to(target_z_at_mu, device='cpu')
        return target_z_at_mu

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes distribution parameters (mu) for state/observation,
        returns (gaussian) sampled action."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mu = self.model(*model_inputs)
        action = self.distribution.sample(DistInfo(mean=mu))
        agent_info = AgentInfo(mu=mu)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_model, self.model.state_dict(), tau)
        update_state_dict(self.target_z_model, self.z_model.state_dict(), tau)

    def z_parameters(self):
        return self.z_model.parameters()

    def mu_parameters(self):
        return self.model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.z_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.z_model.eval()
        self.distribution.set_std(self.action_std)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.z_model.eval()
        self.distribution.set_std(0.)  # Deterministic.

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            z_model=self.z_model.state_dict(),
            target_model=self.target_model.state_dict(),
            target_z_model=self.target_z_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.z_model.load_state_dict(state_dict["z_model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.target_z_model.load_state_dict(state_dict["target_z_model"])
