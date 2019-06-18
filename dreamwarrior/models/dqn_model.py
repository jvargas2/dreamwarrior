import logging
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import dreamwarrior
from dreamwarrior.networks import DQN
from dreamwarrior.utils import ReplayMemory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
# TARGET_UPDATE = 10
NUM_EPISODES = 100
FRAME_SKIP = 4

class DQN_Model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = None
    policy_net = None
    target_net = None

    def __init__(self, env):
        self.env = env

        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        n_actions = env.action_space.n

        self.policy_net = DQN(screen_height, screen_width, n_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, n_actions).to(self.device)

    def get_screen(self):
        """Get retro env render as a torch tensor.

        Returns: A torch tensor made from the RGB pixels
        """
        env = self.env

        # Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Resize, and add a batch dimension (BCHW)
        transforms.Compose([
            screen,
            transforms.ToPILImage(),
            transforms.Resize(40, interpolation=Image.CUBIC),
            transforms.ToTensor()
        ])

        return screen.unsqueeze(0).to(self.device)

    def optimize_model(self, optimizer, memory, BATCH_SIZE, GAMMA):
        """Optimize the model.
        """
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
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
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
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

    def select_action(self, state, steps_done):
        # Select and perform an action
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long
            )

        return action

    def train(self, optimizer_state=None, start_episode=0, watching=False):
        logging.info('Starting training...')
        env = self.env
        device = self.device

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to ???
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        optimizer = optim.RMSprop(self.policy_net.parameters())

        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        memory = ReplayMemory(10000)

        steps_done = 0

        for i_episode in range(start_episode, NUM_EPISODES):
            # Initialize the environment and state
            env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            previous_action = self.select_action(state, steps_done)

            for t in count():
                if t % FRAME_SKIP == 0:
                    action = self.select_action(state, steps_done)
                else:
                    action = previous_action

                retro_action = np.zeros((9,), dtype=int)
                retro_action[action.item()] = 1
                _, reward, done, _ = env.step(retro_action)
                reward = torch.tensor([reward], device=device)

                if reward > 0:
                    logging.info('t=%i got reward: %g' % (t, reward))
                elif reward < 0:
                    logging.info('t=%i got penalty: %g' % (t, reward))

                
                if watching:
                    env.render()

                # Observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = self.optimize_model(optimizer, memory, BATCH_SIZE, GAMMA)

                if t % 100 == 0 and len(memory) >= BATCH_SIZE:
                    logging.info('t=%d loss: %f' % (t, loss))

                # if t > 20:
                #     done = True

                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            # if i_episode % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

            logging.info('Finished episode ' + str(i_episode))
            self.training_save(i_episode, optimizer)

        env.close()
        logging.info('Finished training!')

    def run(self):
        env = self.env
        device = self.device

        # Initialize the environment and state
        env.reset()
        last_screen = self.get_screen()
        current_screen = self.get_screen()
        state = current_screen - last_screen
        final_reward = 0

        for t in count():
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)

            retro_action = np.zeros((9,), dtype=int)
            retro_action[action.item()] = 1
            _, reward, done, _ = env.step(retro_action)
            final_reward += reward
            reward = torch.tensor([reward], device=device)

            if reward > 0:
                print('t=%i got reward: %g' % (t, reward))

            env.render()

            # Observe new state
            last_screen = current_screen
            current_screen = self.get_screen()
            state = current_screen - last_screen

            if done:
                break

        env.close()
        logging.info('Finished run.')
        logging.info('Final reward: %d' % final_reward)

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

    def training_save(self, episode, optimizer):
        filepath = 'latest.pth'
        state = {
            'episode': episode,
            'state_dict': self.policy_net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)
        logging.info('Saved training at episode %d.' % episode)

    def continue_training(self, filepath):
        state = torch.load(filepath, map_location=self.device)
        episode = state['episode'] + 1

        self.policy_net.load_state_dict(state['state_dict'])
        logging.info('Continuing training at episode %d...' % episode)
        self.train(optimizer_state=state['optimizer'], start_episode=episode)
