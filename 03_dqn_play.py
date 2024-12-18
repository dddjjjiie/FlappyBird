#!/usr/bin/env python3
import gymnasium
import time
import argparse
import numpy as np

import torch

from lib import wrappers
from lib import dqn

import collections

DEFAULT_ENV_NAME = "flappy_bird_env:FlappyBird-rgb-v0"
FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="flappy_bird_env:FlappyBird-rgb-v0-best.dat", help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()


    env = wrappers.make_env(args.env)
    # if args.record:
    #     env = gym.wrappers.RecordVideo(env, args.record)
    net = dqn.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state, _ = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.visualize:
            env.render()
        state_v = torch.tensor(np.array([state]))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

