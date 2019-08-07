import random
import logging
import math

import torch
import torch.nn.functional as F

from dreamwarrior.nn import DQN, DuelingDQN, NoisyNetDQN, NoisyNetDueling

class DQNAgent:
    device = None
    env = None
    num_actions = 0
    noisy = False

    def __init__(self, env, config):
        self.device = config.device
        self.env = env
        self.config = config
        self.num_actions = env.action_space.n
       
        init_screen = env.get_state()

        if config.dueling and config.noisy:
            self.model_class = NoisyNetDueling
        elif config.dueling:
            self.model_class = DuelingDQN
        elif config.noisy:
            self.model_class = NoisyNetDQN
        else:
            self.model_class = DQN

        if config.categorical:
            self.model = self.model_class(init_screen.shape, self.num_actions, config.atoms).to(self.device)
            self.target_model = self.model_class(init_screen.shape, self.num_actions, config.atoms).to(self.device)
        else:
            self.model = self.model_class(init_screen.shape, self.num_actions).to(self.device)
            self.target_model = self.model_class(init_screen.shape, self.num_actions).to(self.device)

        self.noisy = config.noisy
        self.gamma = config.gamma
        self.prioritized_memory = config.prioritized
        self.frame_skip = config.frame_skip
        self.frame_update = config.frame_update

        if not self.noisy:
            self.epsilon_start = config.epsilon_start
            self.epsilon_end = config.epsilon_end
            self.epsilon_decay = config.epsilon_decay

    def random_action(self):
        action = random.randrange(self.num_actions)

        return action

    def select_action(self, state, frame_count):
        # Get state out of batch
        state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)
            action = q_values.max(1)[1].item()

        return action

    def act(self, state, frame_count):
        action = None

        if self.noisy:
            action = self.select_action(state, frame_count)
            self.model.reset_noise()
        else:
            # Epsilon greedy strategy
            start = self.epsilon_start
            end = self.epsilon_end
            decay = self.epsilon_decay

            epsilon_threshold = end + (start - end) * math.exp(-1. * frame_count / decay)
            sample = random.random()
            
            if sample > epsilon_threshold:
                action = self.select_action(state, frame_count)
            else:
                action = self.random_action()

        return action

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

        # Get estimated q values
        q_values = self.model(state)
        q_value = q_values.gather(1, action) # Q-Values for selected actions

        # Calculate estimated q* value
        # Rt+1 + Ɣ max_a q*(s', a')
        next_q_values = self.target_model(next_state) 
        next_max_q_value = next_q_values.max(1)[0].unsqueeze(1) # Max Q value in next state
        q_star_value = reward + self.gamma * next_max_q_value * (1 - done)
        
        # Compute Huber loss
        loss, priorities = self.calculate_loss(q_value, q_star_value, weights)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.noisy:
            self.model.reset_noise()
        
        return loss.item(), indices, priorities

    def calculate_loss(self, q_value, q_star_value, weights=None):
        if weights is not None:
            loss = F.smooth_l1_loss(q_value, q_star_value, reduction='none')
            loss *= torch.tensor(weights, device=self.device).unsqueeze(1)
            priorities = loss + 1e-5
            loss = loss.mean()
            return loss, priorities
        else:
            loss = F.smooth_l1_loss(q_value, q_star_value)
            return loss, None

    def get_parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    """
    For saving and loading:
    https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

    Will likely want to switch to method 3 when saving the final model versions
    """
    def save(self):
        agent_name = '%s-agent.pt' % self.env.name

        torch.save({
            'game': self.env.gamename,
            'agent_class': self.__class__.__name__,
            'config': self.config,
            'state_dict': self.state_dict(),
        }, agent_name)

        logging.info('Saved model.')

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()

        logging.info('Loaded model.')
