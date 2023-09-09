import gym
import numpy as np

from collections import deque
from gym.spaces import Box

class CastObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if self.env.observation_space.dtype != np.uint8:
            self._observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                shape=self.env.observation_space.shape,
                dtype=np.float32,
            )

    def observation(self, observation):
        if observation.dtype != np.uint8:
            return observation.astype(np.float32)
        else:
            return observation


class TimeLimit(gym.Wrapper):
    # https://github.com/openai/gym/blob/0.23.0/gym/wrappers/time_limit.py
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class SparseReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = float(info["success"])
        return obs, reward, done, info


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class NormalizeAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Only normalize bounded action dimensions
        space = env.action_space
        bounded = np.isfinite(space.low) & np.isfinite(space.high)
        self._action_space = Box(
            low=np.where(bounded, -1, space.low),
            high=np.where(bounded, 1, space.high),
            dtype=np.float32,
        )
        self._low = np.where(bounded, space.low, -1)
        self._high = np.where(bounded, space.high, 1)

    def step(self, action):
        orig_action = (action + 1) / 2 * (self._high - self._low) + self._low
        return self.env.step(orig_action)


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=1):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=self.num_stack)

        assert len(env.observation_space.shape) == 3
        width, height = env.observation_space.shape[1:]
        self._observation_space = Box(
            high=255,
            low=0,
            shape=(3 * self.num_stack, width, height),
            dtype=np.uint8,
        )

    @property
    def stacked_obs(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(self.frames, 0)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.num_stack)]
        return self.stacked_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.stacked_obs, reward, done, info


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, pixel_obs=False):
        # Clear settings to use Metaworld environments as standalone tasks
        env.unwrapped._set_task_called = True
        env.unwrapped.random_init = False
        env.unwrapped.max_path_length = np.inf

        super().__init__(env)
        self._pixel_obs = pixel_obs
        self._camera = "corner3"
        self._resolution = 64
        if pixel_obs:
            img_shape = (3, self._resolution, self._resolution)
            self._observation_space = Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

    def reset(self, **kwargs):
        state_obs = self.env.reset(**kwargs)
        if self._pixel_obs:
            obs = self.render("rgb_array")
        else:
            obs = state_obs
        return obs

    def step(self, action):
        state_obs, reward, done, info = self.env.step(action)
        if self._pixel_obs:
            obs = self.render("rgb_array")
            info["state_obs"] = state_obs
        else:
            obs = state_obs
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            image = self.env.sim.render(
                camera_name=self._camera,
                width=self._resolution,
                height=self._resolution,
                depth=False,
            )
            return image.transpose(2, 0, 1).copy()
        else:
            return self.env.render(mode)


class FrankaWrapper(gym.Wrapper):
    def __init__(self, env, pixel_obs=False):
        super().__init__(env)
        self._pixel_obs = pixel_obs
        if pixel_obs:
            self._observation_space = Box(
                low=0,
                high=255,
                shape=(6, 64, 64),
                dtype=np.uint8,
            )

    def _get_obs(self, obs):
        if self._pixel_obs:
            img = self.env.render(mode="rgb_array")
            obs = np.concatenate([img[...,0:3], img[...,4:7]], axis=-1)
            obs = obs.transpose(2, 0, 1).astype(np.uint8).copy()
        return obs
        

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info