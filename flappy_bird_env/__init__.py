import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Exporting envs:
from flappy_bird_env.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from flappy_bird_env.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

# Exporting original game:
# from flappy_bird_env import original_game

# Exporting gym.make:
from gymnasium import make

# Registering environments:
from gymnasium.envs.registration import register

register(
    id="FlappyBird-v0",
    entry_point="flappy_bird_env.envs:FlappyBirdEnvSimple",
)

register(
    id="FlappyBird-rgb-v0",
    entry_point="flappy_bird_env.envs:FlappyBirdEnvRGB",
)

# Main names:
# __all__ = [
#     make.__name__,
#     FlappyBirdEnvRGB.__name__,
#     FlappyBirdEnvSimple.__name__,
# ]
