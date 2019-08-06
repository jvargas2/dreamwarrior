import random

import numpy as np
from collections import deque

import torch

class ReplayMemory:
    def __init__(self, config):
        self.device = config.device
        self.buffer = []
        self.capacity = config.capacity
        self.position = 0
        self.batch_size = config.batch_size
        self.multi_step = config.multi_step

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        transition = [state, action, reward, next_state, done]

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)

        state, action, reward, next_state, done = zip(*sample)

        state = torch.stack(state)
        action = torch.tensor(action, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).reshape(-1, 1)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, dtype=torch.float, device=self.device).reshape(-1, 1)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)