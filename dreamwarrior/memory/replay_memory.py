import random

import numpy as np
from collections import deque

import torch

class ReplayMemory:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.buffer = []
        self.capacity = config.capacity
        self.position = 0
        self.batch_size = config.batch_size
        self.multi_step = config.multi_step
        self.gamma = config.gamma

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        transition = [state, action, reward, next_state, done]

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def multi_step_sample(self, sample_indices):
        n = self.multi_step - 1
        
        sample = []
        final_index = len(self.buffer) - 1

        for sample_index, buffer_index in enumerate(sample_indices):
            state, action, reward, next_state, done = self.buffer[buffer_index]

            for i in range(1, n + 1):
                # Wrap around if we step past the end of the buffer
                i_index = buffer_index + i
                if i_index > final_index:
                    i_index %= (final_index + 1)

                discount = self.gamma ** i
                reward += discount * self.buffer[i_index][2]

            n_index = (buffer_index + n) % (self.position + 1)
            next_state = self.buffer[n_index][3]

            transition = [state, action, reward, next_state, done]
            sample.append(transition)
        
        return sample

    def get_possible_indices(self, reverse=False):
        # Make sure we don't step past the latest transition
        if reverse:
            possible_indices = []
        else:
            possible_indices = list(range(len(self.buffer)))

        latest_transition_index = self.position - 1
        if latest_transition_index < 0:
            latest_transition_index = len(self.buffer) + latest_transition_index

        remove_start = latest_transition_index - self.multi_step + 2
        if remove_start < 0:
            remove_start = len(self.buffer) + remove_start

        for i in range(remove_start, remove_start + self.multi_step - 1):
            if i > len(self.buffer) - 1:
                i %= len(self.buffer)

            if reverse:
                possible_indices.append(i)
            else:
                possible_indices.remove(i)

        return possible_indices

    def sample(self):
        sample = []

        if self.multi_step > 1:
            possible_indices = self.get_possible_indices()
            sample_indices = random.sample(possible_indices, self.batch_size)
            sample = self.multi_step_sample(sample_indices)
        else:
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