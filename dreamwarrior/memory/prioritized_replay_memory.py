"""
Prioritized Replay Memory as described in https://arxiv.org/abs/1511.05952

Based off simplified data storage from https://github.com/higgsfield/RL-Adventure due to
low capacity.
"""
import numpy as np
import torch

ALPHA = 0.6 # How much prioritization is used
BETA_START = 0.4
BETA_FRAMES = int(1e6)

class PrioritizedReplayMemory:
    def __init__(self, capacity, batch_size, device=None):
        self.device = device
        self.capacity = capacity
        self.buffer = []
        self.batch_size = batch_size

        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        transition = (state, action, reward, next_state, done)

        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, frame):
        """Select a sample using proportional prioritization.
        """
        beta = min(1.0, BETA_START + frame * (1.0 - BETA_START) / BETA_FRAMES)

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # P(i) = (pi^alpha) / (sum(pk^alpha))
        probabilities = priorities ** ALPHA
        probabilities /= probabilities.sum()

        # Select sample based on computed probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        sample = [self.buffer[index] for index in indices]

        # Importance sampling to reduce bias from the changed distribution
        # wi = [ (1/N) * (1/P(i)) ] ^ beta
        buffer_length = len(self.buffer) # N
        weights = (buffer_length * probabilities[indices]) ** (-beta)

        # Normalize weights so they only scale the update downwards
        # 1 / maxi wi
        # TODO Confirm whether this should be (1/maxi)*wi or 1/(maxi*wi)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Unzip sample
        state, action, reward, next_state, done = zip(*sample)

        # Apply proper torch operations
        states = torch.stack(state)
        actions = torch.tensor(action, device=self.device).unsqueeze(1)
        rewards = torch.tensor(reward, dtype=torch.float, device=self.device).reshape(-1, 1)
        next_states = torch.stack(next_state)
        dones = torch.tensor(done, dtype=torch.float, device=self.device).reshape(-1, 1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        # batch_priorities =
        for index, priority in zip(batch_indices, batch_priorities):
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)
