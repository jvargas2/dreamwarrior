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
        self.gamma = config.gamma
        self.multi_step = config.multi_step

    def buffer_index(self, index):
        """Return the equivalent index that is appropriate for the buffer.

        Params:
            index: Index we want to keep inside the buffer.
        
        Returns:
            The new index.
        """
        # If index is below 0 wrap back around to the end
        if index < 0:
            index = len(self.buffer) + index
        
        # If index is past the end then wrap around to the beginning of the buffer
        if index >= len(self.buffer):
            index %= len(self.buffer)

        return index


    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        transition = [state, action, reward, next_state, done]

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def multi_step_sample(self, sample_indices):
        sample = []

        for first_transition_index in sample_indices:
            state, action, reward, next_state, done = self.buffer[first_transition_index]

            # If already done then don't take any steps at all
            if not done:
                # First step is already done as part of the first transition so range starts at 1
                # instead of 0.
                for step in range(1, self.multi_step):
                    # Get index of this step's transition
                    step_index = self.buffer_index(first_transition_index + step)

                    # Get the relevant values from this transition. Note that the overall next_state 
                    # is updated.
                    _, _, step_reward, next_state, step_done = self.buffer[step_index]

                    # Add the discounted reward to the overall reward
                    discount = self.gamma ** step
                    reward += discount * step_reward

                    if step_done:
                        # Truncate the steps. Need to pass the done to the optimization algorithm.
                        done = True
                        break

            # Save the new next_state and reward into a transition and add ot the sample
            transition = [state, action, reward, next_state, done]
            sample.append(transition)

        return sample

    def get_possible_indices(self, reverse=False):
        """Get a list of the possible indices for a multistep sample. The reverse flag gets a list
        of only the not usable indices instead.

        Args:
            reverse: Boolean on whether to reverse. Reversing returns a list of non usable indices.
        """
        if reverse:
            possible_indices = []
        else:
            possible_indices = list(range(len(self.buffer)))

        # Get the index of the last added transition
        latest_transition_index = self.position - 1
        latest_transition_index = self.buffer_index(latest_transition_index)

        # remove_start is the first index to be removed. It should only remove multi-step - 1
        # indices. This is because transitions already carry info for the next state.
        num_to_remove = self.multi_step - 1
        remove_index = latest_transition_index - (num_to_remove - 1)

        for _ in range(num_to_remove):
            remove_index = self.buffer_index(remove_index)

            if reverse:
                possible_indices.append(remove_index)
            else:
                possible_indices.remove(remove_index)

            remove_index += 1

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