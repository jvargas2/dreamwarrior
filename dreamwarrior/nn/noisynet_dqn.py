import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamwarrior.nn import Noisy

class NoisyNetDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(NoisyNetDQN, self).__init__()

        self.input_shape = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.noisy1 = Noisy(self.feature_size(), 512)
        self.noisy2 = Noisy(512, num_actions)

    def feature_size(self):
        zeros = torch.zeros(1, *self.input_shape)
        return self.features(zeros).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
