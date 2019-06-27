import random
import logging
from collections import namedtuple

import torch
import torch.nn.functional as F

from dreamwarrior.models import DQN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQNAgent:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = None
    n_actions = 0
    policy_net = None
    target_net = None

    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n

        init_screen = env.get_state()
        _, _, screen_height, screen_width = init_screen.shape

        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(self.device)

    def random_action(self):
        action = torch.tensor(
            [[random.randrange(self.n_actions)]],
            device=self.device,
            dtype=torch.long
        )

        return action

    def select_action(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = self.policy_net(state).max(1)[1].view(1, 1)

        return action

    def load_policy_net(self, state_dict):
        self.policy_net.load_state_dict(state_dict)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def start_target_net(self):
        self.update_target_net()
        self.target_net.eval()

    def get_state_dict(self):
        return self.policy_net.state_dict()

    def optimize_model(self, optimizer, memory, GAMMA):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return
        transitions = memory.sample()
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(memory.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss

    """
    For saving and loading:
    https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

    Will likely want to switch to method 3 when saving the final model versions
    """
    def save(self):
        torch.save(self.policy_net.state_dict(), 'test.pth')
        logging.info('Saved model.')

    def load(self, path='test.pth'):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        logging.info('Loaded model.')
