import time

import gymnasium
import numpy as np



from flappy_bird_env.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from lib import wrappers

env = wrappers.make_env("flappy_bird_env:FlappyBird-rgb-v0")

obs, _ = env.reset() # (288, 512, 3)
while True:
    env.render()
    print(obs.shape, type(obs))
    # Next action:
    # (feed the observation to your agent here)
    action = np.random.choice((0, 1), p=[0.9, 0.1])

    # Processing:
    obs, reward, terminated, _, info = env.step(action)

    time.sleep(0.3)
    # Checking if the player is still alive
    if terminated:
        break

env.close()