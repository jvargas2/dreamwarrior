import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    input_shape = None
    num_actions = 0

    def __init__(self, input_shape, num_actions, num_atoms=0):
        super(DQN, self).__init__()

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

        output_size = num_actions * num_atoms if num_atoms > 0 else num_actions
        self.fully_connected = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def feature_size(self):
        zeros = torch.zeros(1, *self.input_shape)
        return self.features(zeros).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        if self.num_atoms > 0:
            x = F.softmax(x.view(-1, self.num_atoms), dim=1)
            x = x.view(-1, self.num_actions, self.num_atoms)

        return x