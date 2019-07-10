import random

import numpy as np
from collections import deque

import torch

class ReplayMemory(object):
    device = None
    buffer = None
    batch_size = 0

    def __init__(self, capacity, batch_size, device=None):
        self.device = torch.device('cpu' if device is None else device)
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if self.device.type == 'cpu':
            state = state.data.numpy()
            next_state = next_state.data.numpy()
        else:
            state = state.data.cpu().numpy()
            next_state = next_state.data.cpu().numpy()
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*sample)

        state = torch.tensor(state, device=self.device)
        action = torch.tensor(action, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).reshape(-1, 1)
        next_state = torch.tensor(next_state, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device).reshape(-1, 1)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)