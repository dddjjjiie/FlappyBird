import cv2
import gymnasium as gym
import numpy as np
import collections

from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

from flappy_bird_env.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
writer = SummaryWriter()
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
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, _ = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame): # 将图像大小转为1*84*84
        frame = frame.transpose(2, 1, 0) # WHC -> CHW
        writer.add_image("img", frame)
        if frame.size == 288 * 512 * 3:
            img = frame.astype(np.float32)
        else:
            assert False, "Unknown resolution."

        img = img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2, :, :] * 0.114
        img = img.astype(np.uint8)
        writer.add_image("grey", img.reshape(1, img.shape[0], img.shape[1]))
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[:84, :]
        x_t = np.reshape(x_t, [1, x_t.shape[0], x_t.shape[1]])
        writer.add_image("resize", x_t)
        return x_t.astype(np.uint8)



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
        super().reset(seed=seed)
        self.buffer = np.zeros_like(self.observation_space.low)
        obse, _ = self.env.reset()
        return self.observation(obse), _

    def observation(self, obse):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obse
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    # env = MaxAndSkipEnv(env)
    # env = FireResetEnv(env)
    env = ProcessFrame84(env)
    # env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)