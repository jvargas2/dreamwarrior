import random
import logging
import math

import numpy as np
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
    
        self.target_model.load_state_dict(self.model.state_dict())

        self.double = config.double
        self.noisy = config.noisy
        self.gamma = config.gamma
        self.prioritized_memory = config.prioritized
        self.frame_skip = config.frame_skip
        self.frame_update = config.frame_update
        self.update_counter = 1

        if not self.noisy:
            self.epsilon_start = config.epsilon_start
            self.epsilon_end = config.epsilon_end
            self.epsilon_decay = config.epsilon_decay

    def random_action(self):
        action = random.randrange(self.num_actions)
        return action

    def select_action(self, state):
        state = state.unsqueeze(0)
        q_values = self.model(state)
        action = q_values.max(1)[1].item()
        return action

    def act(self, state):
        action = None

        if self.noisy:
            with torch.no_grad():
                action = self.select_action(state)
            self.model.reset_noise()
        else:
            # Epsilon greedy strategy
            start = self.epsilon_start
            end = self.epsilon_end
            decay = self.epsilon_decay

            steps_left = decay - self.env.frame
            bonus = (start - end) * steps_left / decay
            bonus = np.clip(bonus, 0., 1. - end)
            epsilon_threshold = end + bonus

            sample = random.random()
            
            if sample > epsilon_threshold:
                action = self.select_action(state)
            else:
                action = self.random_action()

        # Update target if appropriate
        if self.update_counter >= self.frame_update:
            self.update_target()
            self.update_counter = 1
        else:
            self.update_counter += self.frame_skip

        return action

    def optimize_model(self, optimizer, memory):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return

        indices, weights = None, None

        if self.prioritized_memory:
            frame = self.env.frame
            transitions = memory.sample(frame)
            indices = transitions[5]
            weights = transitions[6]
            transitions = transitions[:5]
        else:
            transitions = memory.sample()

        loss, priorities = self.calculate_loss(transitions, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.noisy:
            self.model.reset_noise()
            self.target_model.reset_noise()
        
        return loss.item(), indices, priorities

    def calculate_loss(self, transitions, weights):
        state, action, reward, next_state, done = transitions

        # Get estimated q values
        q_values = self.model(state)
        q_value = q_values.gather(1, action) # Q-Values for selected actions

        # Get target q values for next state
        next_target_q_values = self.target_model(next_state)

        # Calculate estimated q* value
        if self.double: 
            # R' + Ɣ q(s', max_a(s', a'; ϴ); ϴ')
            next_q_values = self.model(next_state)
            next_max_actions = torch.max(next_q_values, 1)[1].unsqueeze(1)
            next_q_value = next_target_q_values.gather(1, next_max_actions)
        else:
            # R' + Ɣ max_a q(s', a'; ϴ')
            next_q_value = next_target_q_values.max(1)[0].unsqueeze(1) # Max Q value in next state
        
        target_q_value = reward + self.gamma * next_q_value * (1 - done) # Bellman Q*

        if weights is not None:
            loss = F.smooth_l1_loss(q_value, target_q_value, reduction='none')
            loss *= torch.tensor(weights, device=self.device).unsqueeze(1)
            priorities = loss + 1e-5
            loss = loss.mean()
            return loss, priorities
        else:
            loss = F.smooth_l1_loss(q_value, target_q_value)
            return loss, None

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

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

        self.update_target()
        self.target_model.eval()

        logging.info('Loaded model.')
