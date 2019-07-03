import random
# from collections import namedtuple

import numpy as np
from collections import deque

import torch

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    buffer = None
    batch_size = 0
    # capacity = 0
    # memory = []
    # position = 0
    # batch_size = 32

    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        # self.capacity = capacity
        # self.memory = []
        # self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        # state = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*sample)

        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).reshape(-1, 1)
        next_state = torch.stack(next_state)
        done = torch.tensor(done, dtype=torch.float, device=self.device).reshape(-1, 1)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)