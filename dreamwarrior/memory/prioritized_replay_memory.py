"""
Prioritized Replay Memory as described in https://arxiv.org/abs/1511.05952

Based off simplified data storage from https://github.com/higgsfield/RL-Adventure due to
low capacity.
"""
import numpy as np
import torch

from dreamwarrior.memory import ReplayMemory

class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, config):
        super().__init__(config)

        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.alpha = config.alpha
        self.beta_start = config.beta_start
        self.beta_frames = config.beta_frames

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        old_position = self.position
        max_priority = self.priorities.max() if self.buffer else 1.0

        super().push(state, action, reward, next_state, done)

        self.priorities[old_position] = max_priority

    def sample(self, frame):
        """Select a sample using proportional prioritization.
        """
        start = self.beta_start
        beta = min(1.0, start + frame * (1.0 - start) / self.beta_frames)

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # (pi^alpha)
        probabilities = priorities ** self.alpha

        # Zero out probabilities for multi-step
        if self.multi_step > 1:
            zero_indices = self.get_possible_indices(reverse=True)

            for index in zero_indices:
                probabilities[index] = 0.0

        # P(i) = (pi^alpha) / (sum(pk^alpha))
        probabilities /= probabilities.sum()

        # Select sample based on computed probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
        # sample = [self.buffer[index] for index in indices]
        sample = self.multi_step_sample(indices)

        # Importance sampling to reduce bias from the changed distribution
        # wi = [ (1/N) * (1/P(i)) ] ^ beta
        buffer_length = len(self.buffer) # N
        weights = (buffer_length * probabilities[indices]) ** (-beta)

        # Normalize weights so they only scale the update downwards
        # 1 / maxi wi
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
        for index, priority in zip(batch_indices, batch_priorities):
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)
