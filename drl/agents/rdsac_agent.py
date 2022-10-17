import torch
from drl.agents.dsac_agent import DsacAgent
from drl.models.mlp import PiMlpModel
from drl.models.mlp_risk import RiskZofMuMlpModel
# from drl.models.nc_mlp_risk import RiskNonCrossZofMuMlpModel
from rlpyt.utils.buffer import buffer_to


class RdsacAgent(DsacAgent):
    """Agent for RDSAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
        self,
        *args,
        ModelCls=PiMlpModel,  # Pi model.
        ZModelCls=RiskZofMuMlpModel,
        **kwargs,
    ):
        super().__init__(*args, ModelCls=ModelCls, ZModelCls=ZModelCls, **kwargs)

    def z(self, observation, prev_action, prev_reward, action, tau):
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, action, tau), device=self.device)
        z1, h1 = self.z1_model(*model_inputs)
        z2, h2 = self.z2_model(*model_inputs)
        # z1, z2, h1, h2 = buffer_to((z1, z2, h1, h2), device="cpu")
        return z1, z2, h1, h2

    @torch.no_grad()
    def target_z(self, observation, prev_action, prev_reward, action, tau):
        """Compute twin target Q-values for state/observation and input
        action."""
        model_inputs = buffer_to((observation, prev_action, prev_reward, action, tau), device=self.device)
        target_z1, target_h1 = self.target_z1_model(*model_inputs)
        target_z2, target_h2 = self.target_z2_model(*model_inputs)
        # target_z1, target_z2, target_h1, target_h2 = \
        #     buffer_to((target_z1, target_z2, target_h1, target_h2), device="cpu")
        return target_z1, target_z2, target_h1, target_h2
