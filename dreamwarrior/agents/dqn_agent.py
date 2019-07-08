import random
import logging

import torch
import torch.nn.functional as F

from dreamwarrior.models import DQN, DuelingDQN

class DQNAgent:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    env = None
    num_actions = 0

    def __init__(self, env, model='dqn'):
        self.env = env
        self.num_actions = env.action_space.n

        init_screen = env.get_full_state()

        if model == 'dqn' or model == 'ddqn':
            self.model = DQN(init_screen.shape, self.num_actions).to(self.device)
        elif model == 'dueling-dqn':
            self.model = DuelingDQN(init_screen.shape, self.num_actions).to(self.device)
        else:
            raise ValueError('%s is not a supported model.' % model)

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

    def optimize_model(self, optimizer, memory, gamma):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return

        state, action, reward, next_state, done = memory.sample()

        # Get Q values for every action in first and second states
        q_values = self.model(state)
        next_q_values = self.model(next_state) 

        q_value = q_values.gather(1, action) # Actual action-value
        next_q_value = next_q_values.max(1)[0].unsqueeze(1) # Max Q value in second state

        # Calculate expected return
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_value, expected_q_value)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss

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
