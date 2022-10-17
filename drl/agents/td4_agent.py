from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallelCPU as DDPC  # Deprecated.

from drl.agents.sdpg_agent import SdpgAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.gaussian import Gaussian, DistInfo
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger


class Td4Agent(SdpgAgent):
    """Agent for TD3 algorithm, using two Q-models and two target Q-models."""

    def __init__(
        self,
        pretrain_std=0.5,  # To make actions roughly uniform.
        target_noise_std=0.2,
        target_noise_clip=0.5,
        initial_z2_model_state_dict=None,
        **kwargs,
    ):
        """Saves input arguments."""
        super().__init__(**kwargs)
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        self.z2_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        if self.initial_z2_model_state_dict is not None:
            self.z2_model.load_state_dict(self.initial_z2_model_state_dict)
        self.target_z2_model = self.ZModelCls(**self.env_model_kwargs, **self.z_model_kwargs)
        self.target_z2_model.load_state_dict(self.z2_model.state_dict())
        self.target_distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            std=self.target_noise_std,
            noise_clip=self.target_noise_clip,
            clip=env_spaces.action.high[0],  # Assume symmetric low=-high.
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.z2_model.to(self.device)
        self.target_z2_model.to(self.device)

    def data_parallel(self):
        device_id = super().data_parallel()
        self.z2_model = DDP(
            self.z2_model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        return device_id

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def z(self, observation, prev_action, prev_reward, action, tau):
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, action, tau), device=self.device)
        z1 = self.z_model(*model_inputs)
        z2 = self.z2_model(*model_inputs)
#         z1, z2 = buffer_to((z1, z2), device="cpu")
        return z1, z2

    def target_z_at_mu(self, observation, prev_action, prev_reward, tau):
        """Compute twin target Q-values for state/observation, through
        target mu model."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        target_mu = self.target_model(*model_inputs)
        tau = buffer_to(tau, device=self.device)
        target_action = self.target_distribution.sample(DistInfo(mean=target_mu))
        target_z1_at_mu = self.target_z_model(*model_inputs, target_action, tau)
        target_z2_at_mu = self.target_z2_model(*model_inputs, target_action, tau)
#         target_z1_at_mu, target_z2_at_mu = buffer_to((target_z1_at_mu, target_z2_at_mu), device="cpu")
        return target_z1_at_mu, target_z2_at_mu

    def update_target(self, tau=1):
        super().update_target(tau)
        update_state_dict(self.target_z2_model, self.z2_model.state_dict(), tau)

    def z_parameters(self):
        yield from self.z_model.parameters()
        yield from self.z2_model.parameters()

    def set_target_noise(self, std, noise_clip=None):
        self.target_distribution.set_std(std)
        self.target_distribution.set_noise_clip(noise_clip)

    def train_mode(self, itr):
        super().train_mode(itr)
        self.z2_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.z2_model.eval()
        std = self.action_std if itr >= self.min_itr_learn else self.pretrain_std
        if itr == 0 or itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: {std}.")
        self.distribution.set_std(std)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.z2_model.eval()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["z2_model"] = self.z2_model.state_dict()
        state_dict["target_z2_model"] = self.target_z2_model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.z2_model.load_state_dict(state_dict["z2_model"])
        self.target_z2_model.load_state_dict(state_dict["target_z2_model"])
