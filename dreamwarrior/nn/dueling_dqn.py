import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    input_shape = None
    num_actions = 0

    def __init__(self, input_shape, num_actions, num_atoms=1):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms

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
            nn.Linear(512, num_actions * num_atoms)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_atoms)
        )

    def feature_size(self):
        zeros = torch.zeros(1, *self.input_shape)
        return self.features(zeros).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        # return value + advantage - advantage.mean()

        if self.num_atoms > 1:
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
            x = value + advantage - advantage.mean(1, keepdim=True)
            x = F.softmax(x.view(-1, self.num_atoms), dim=1)
            x = x.view(-1, self.num_actions, self.num_atoms)
        else:
            value = value.view(batch_size, 1)
            advantage = advantage.view(batch_size, self.num_actions)
            x = value + advantage - advantage.mean()

        return x