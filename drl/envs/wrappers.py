import gym
from gym.spaces import Box
from gym import ActionWrapper
import numpy as np

from rlpyt.envs.gym import GymEnvWrapper, EnvInfoWrapper


class NoiseAction(ActionWrapper):
    r"""Add noise to the continuous action and clip them within the valid bound. """

    def __init__(self, env, noise_scale=0.):
        assert isinstance(env.action_space, Box)
        super(NoiseAction, self).__init__(env)
        self._noise_scale = noise_scale

    def action(self, action):
        action += (self.action_space.high -
                   self.action_space.low) * self._noise_scale * self.np_random.randn(*action.shape)
        return np.clip(action, self.action_space.low, self.action_space.high)


class LongRun(gym.Wrapper):

    def __init__(self, env, max_episode_steps=None, extra_end_cost=10):
        super(LongRun, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self._extra_end_cost = extra_end_cost

    def step(self, action):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if done:  # reset the agent if it falls, and give it an extra cost
            observation = self.env.reset()
            reward -= self._extra_end_cost
            done = False
        if self._elapsed_steps >= self._max_episode_steps:
            info["timeout"] = True
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


def make_noise_env(*args, noise_scale=0., info_example=None, **kwargs):
    """Use as factory function for making instances of gym environment with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``.  If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    # noise_scale = kwargs.get('noise_scale', 0.)
    env = NoiseAction(gym.make(*args, **kwargs), noise_scale)
    if info_example is None:
        return GymEnvWrapper(env)
    else:
        return GymEnvWrapper(EnvInfoWrapper(env, info_example))


def make_longrun_noise_env(*args,
                           noise_scale=0.,
                           max_episode_steps=1000,
                           extra_end_cost=10,
                           info_example={'timeout': False},
                           **kwargs):
    """Use as factory function for making instances of gym environment with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``.  If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    # noise_scale = kwargs.get('noise_scale', 0.)
    env = gym.make(*args, **kwargs).env
    env = LongRun(env, max_episode_steps=max_episode_steps, extra_end_cost=extra_end_cost)
    env = NoiseAction(env, noise_scale)
    if info_example is None:
        return GymEnvWrapper(env)
    else:
        return GymEnvWrapper(EnvInfoWrapper(env, info_example))
