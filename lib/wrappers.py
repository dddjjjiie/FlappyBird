import cv2
import gymnasium as gym
import numpy as np
import collections

from flappy_bird_env.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action): # 对于一个动作, 执行4帧, 取最近的两帧中较大的像素作为观测
        obs, total_reward, done, trunc, info = self.env.step(action)
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(0)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, trunc, info

    def reset(self, seed=None, options=None):
        """Clear past frame buffer and init. to first obs. from inner env."""
        super().reset(seed=seed)
        self._obs_buffer.clear()
        obs, info = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, info

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame): # 将图像大小转为1*84*84
        global cnt
        frame = frame.transpose(2, 1, 0) # WHC -> CHW
        if frame.size == 288 * 512 * 3:
            img = frame.astype(np.float32)
        else:
            assert False, "Unknown resolution."

        img = img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2, :, :] * 0.114
        img = img.astype(np.uint8)

        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[:84, :]
        x_t = np.reshape(x_t, [1, x_t.shape[0], x_t.shape[1]])
        return x_t.astype(np.uint8)

class ProcessFrame80(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame80, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 80, 80), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame80.process(obs)

    @staticmethod
    def process(frame): # 将图像大小转为1*80*80
        observation = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation, (1, 80, 80))



class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[0], old_shape[1], old_shape[2]),
                                                dtype=np.float32)

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0))
        self.buffer = np.zeros_like(self.observation_space.low)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.buffer = np.zeros_like(self.observation_space.low)
        obse, info = self.env.reset()
        return self.observation(obse), info

    def observation(self, obse):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obse
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    # env = MaxAndSkipEnv(env)
    # env = FireResetEnv(env)
    env = ProcessFrame84(env)
    # env = ProcessFrame80(env)
    # env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)

    return ScaledFloatFrame(env)