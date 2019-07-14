import random
import logging

import torch
import torch.nn.functional as F

from dreamwarrior.models import DQN, DuelingDQN

class DQNAgent:
    device = None
    env = None
    num_actions = 0

    def __init__(self, env, config):
        self.device = torch.device(config.device)
        self.env = env
        self.config = config
        self.num_actions = env.action_space.n
       
        init_screen = env.get_full_state()

        if config.dueling:
            self.model = DuelingDQN(init_screen.shape, self.num_actions).to(self.device)
        else:
            self.model = DQN(init_screen.shape, self.num_actions).to(self.device)

        self.gamma = config.gamma
        self.prioritized_memory = config.prioritized
        self.frame_skip = config.frame_skip
        self.frame_update = config.frame_update

    def random_action(self):
        action = random.randrange(self.num_actions)

        return action

    def select_action(self, state):
        # Get state out of batch
        state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.model(state).max(1)[1]

        # Return the int instead of tensor
        return action.item()

    def optimize_model(self, optimizer, memory, frame=None):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return

        indices, weights = None, None

        if self.prioritized_memory:
            state, action, reward, next_state, done, indices, weights = memory.sample(frame)
        else:
            state, action, reward, next_state, done = memory.sample()

        # Get Q values for every action in first and second states
        q_values = self.model(state)
        next_q_values = self.model(next_state) 

        q_value = q_values.gather(1, action) # Actual action-value
        next_q_value = next_q_values.max(1)[0].unsqueeze(1) # Max Q value in second state

        # Calculate expected return
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        # Compute Huber loss
        loss, priorities = self.calculate_loss(q_value, expected_q_value, weights)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss, indices, priorities

    def calculate_loss(self, q_value, expected_q_value, weights=None):
        if weights is not None:
            loss = F.smooth_l1_loss(q_value, expected_q_value, reduction='none')
            loss *= torch.tensor(weights, device=self.device).unsqueeze(1)
            priorities = loss + 1e-5
            loss = loss.mean()
            return loss, priorities
        else:
            loss = F.smooth_l1_loss(q_value, expected_q_value)
            return loss, None

    def get_parameters(self):
        return self.model.parameters()

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    """
    For saving and loading:
    https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

    Will likely want to switch to method 3 when saving the final model versions
    """
    def save(self):
        torch.save(self.model.state_dict(), 'test-agent.pth')
        logging.info('Saved model.')

    def load(self, path='test-agent.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logging.info('Loaded model.')
