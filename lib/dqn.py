import collections

from torch import nn
import torch
import numpy as np

GAME = 'Flappy Bird'

# ACTIONS = 2
# OBSERVE = 10000
# EXPLORE = 200000
# FINAL_EPSILON = 0.001
# INITIAL = 0.0001
# REPLAY_MOMORY = 50000
# BATCH = 32

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2), # 20 * 20
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 9
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        print("conv input shape:", input_shape)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)