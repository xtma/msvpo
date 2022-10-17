from collections import namedtuple

import numpy as np
import torch
from drl.models.mlp import PiMlpModel, ZofMuMlpModel
# from drl.models.nc_mlp import NonCrossZofMuMlpModel
from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.distributions.gaussian import DistInfoStd, Gaussian
# from rlpyt.models.qpg.mlp import PiMlpModel
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.nn.parallel import DistributedDataParallelCPU as DDPC  # Deprecated

MIN_LOG_STD = -20
MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "z1", "z2"])


class DsacAgent(BaseAgent):
    """Agent for DSAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
        self,
        ModelCls=None,  # Pi model.
        ZModelCls=None,
        model_kwargs=None,  # Pi model.
        z_model_kwargs=None,
        initial_model_state_dict=None,  # All models.
        pretrain_std=0.75,  # With squash 0.75 is near uniform.
        tau_type='rand',  # Method to generate tau.
        num_quantiles=64,
    ):
        """Saves input arguments; network defaults stored within."""
        if ModelCls is None:
            ModelCls = PiMlpModel
        if ZModelCls is None:
            ZModelCls = ZofMuMlpModel  # Cos Emedding
            # ZModelCls = NonCrossZofMuMlpModel
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[256, 256])
        if z_model_kwargs is None:
            z_model_kwargs = dict(hidden_sizes=[256, 256])
        super().__init__(ModelCls=ModelCls,
                         model_kwargs=model_kwargs,
                         initial_model_state_dict=initial_model_state_dict)
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # Don't let base agent try to load.
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict
        self.z1_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        self.z2_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        self.target_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
        self.target_z1_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        self.target_z2_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_z1_model.load_state_dict(self.z1_model.state_dict())
        self.target_z2_model.load_state_dict(self.z2_model.state_dict())
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=env_spaces.action.high[0],
            min_std=np.exp(MIN_LOG_STD),
            max_std=np.exp(MAX_LOG_STD),
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.z1_model.to(self.device)
        self.z2_model.to(self.device)
        self.target_model.to(self.device)
        self.target_z1_model.to(self.device)
        self.target_z2_model.to(self.device)

    def data_parallel(self):
        device_id = super().data_parallel
        self.z1_model = DDP(
            self.z1_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        self.z2_model = DDP(
            self.z2_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        return device_id

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

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
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, action, tau), device=self.device)
        z1 = self.z1_model(*model_inputs)
        z2 = self.z2_model(*model_inputs)
        # z1, z2 = buffer_to((z1, z2), device="cpu")
        return z1, z2

    @torch.no_grad()
    def target_z(self, observation, prev_action, prev_reward, action, tau):
        """Compute twin target Q-values for state/observation and input
        action."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, action, tau), device=self.device)
        target_z1 = self.target_z1_model(*model_inputs)
        target_z2 = self.target_z2_model(*model_inputs)
        # target_z1, target_z2 = buffer_to((target_z1, target_z2), device="cpu")
        return target_z1, target_z2

    def pi(self, observation, prev_action, prev_reward):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def target_pi(self, observation, prev_action, prev_reward):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mean, log_std = self.target_model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        mean, log_std = self.model(*model_inputs)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_model, self.model.state_dict(), tau)
        update_state_dict(self.target_z1_model, self.z1_model.state_dict(), tau)
        update_state_dict(self.target_z2_model, self.z2_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, z1=self.z1_model, z2=self.z2_model)

    def pi_parameters(self):
        return self.model.parameters()

    def z1_parameters(self):
        return self.z1_model.parameters()

    def z2_parameters(self):
        return self.z2_model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.z1_model.train()
        self.z2_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.z1_model.eval()
        self.z2_model.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.z1_model.eval()
        self.z2_model.eval()
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            z1_model=self.z1_model.state_dict(),
            z2_model=self.z2_model.state_dict(),
            target_model=self.target_model.state_dict(),
            target_z1_model=self.target_z1_model.state_dict(),
            target_z2_model=self.target_z2_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.z1_model.load_state_dict(state_dict["z1_model"])
        self.z2_model.load_state_dict(state_dict["z2_model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.target_z1_model.load_state_dict(state_dict["target_z1_model"])
        self.target_z2_model.load_state_dict(state_dict["target_z2_model"])
