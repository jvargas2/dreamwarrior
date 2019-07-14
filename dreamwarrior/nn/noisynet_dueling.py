import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamwarrior.nn import Noisy

class NoisyNetDueling(nn.Module):
    input_shape = None
    num_actions = 0

    def __init__(self, input_shape, num_actions):
        super(NoisyNetDueling, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.noisy_value1 = Noisy(self.feature_size(), 512)
        self.noisy_value2 = Noisy(512, 1)
        
        self.noisy_advantage1 = Noisy(self.feature_size(), 512)
        self.noisy_advantage2 = Noisy(512, num_actions)

    def feature_size(self):
        zeros = torch.zeros(1, *self.input_shape)
        return self.features(zeros).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        value = value.view(batch_size, 1)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        advantage = advantage.view(batch_size, self.num_actions)

        x = value + advantage - advantage.mean()
        # x = value + advantage - advantage.mean(1, keepdim=True)
        # x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
