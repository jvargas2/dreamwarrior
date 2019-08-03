import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamwarrior.nn import Noisy

class NoisyNetDueling(nn.Module):
    input_shape = None
    num_actions = 0

    def __init__(self, input_shape, num_actions, num_atoms=1):
        super(NoisyNetDueling, self).__init__()

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

         # Calculate sizes
         # We can include num_atoms when not using categorical because the default is set to 1
        zeros = torch.zeros(1, *self.input_shape)
        feature_size = self.features(zeros).view(1, -1).size(1)

        self.noisy_value1 = Noisy(feature_size, 512)
        self.noisy_value2 = Noisy(512, num_atoms)
        
        self.noisy_advantage1 = Noisy(feature_size, 512)
        self.noisy_advantage2 = Noisy(512, num_actions * num_atoms)

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

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

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
