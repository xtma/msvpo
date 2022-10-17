import numpy as np

from rlpyt.samplers.collections import TrajInfo


class LongRunTrajInfo(TrajInfo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self._reward_list = []

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self._reward_list.append(reward)
        rewards = np.asfarray(self._reward_list)
        self.Reward = rewards.mean()
        self.SemivarUp = np.mean(np.clip(rewards - self.Reward, a_min=0., a_max=None)**2)
        self.SemivarDown = np.mean(np.clip(rewards - self.Reward, a_min=None, a_max=0.)**2)
        self.Variance = rewards.var()
        if done:
            del self._reward_list


class AdroitTrajInfo(LongRunTrajInfo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self._goal_achieved_count = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self._goal_achieved_count += env_info.goal_achieved
        self.Success = float(self._goal_achieved_count > 25)
