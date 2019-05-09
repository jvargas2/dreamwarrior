"""DQN Test file
"""
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

import dreamwarrior

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQN_Model():
    env = None
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env):
        self.env = env

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

    def optimize_model(self):
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
                                            batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def train(self):
        batch_size = 128
        gamma = 0.999
        eps_start = 0.9
        eps_end = 0.05
        eps_decay = 200
        target_update = 10

        env = self.env
        device = self.device

        env.reset()

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to ???
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        n_actions = env.action_space.n

        policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net = DQN(screen_height, screen_width, n_actions).to(device)

        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        # memory = ReplayMemory(10000)

        steps_done = 0

        episode_durations = []

        num_episodes = 50
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                sample = random.random()
                eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
                steps_done += 1
                if sample > eps_threshold:
                    with torch.no_grad():
                        # t.max(1) will return largest column value of each row.
                        # second column on max result is index of where max element was
                        # found, so we pick action with the larger expected reward.
                        action = policy_net(state).max(1)[1].view(1, 1)
                else:
                    action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

                retro_action = np.zeros((9,), dtype=int)
                retro_action[action.item()] = 1
                _, reward, done, _ = env.step(retro_action)
                reward = torch.tensor([reward], device=device)

                if t % 10:
                    env.render()

                # Observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                # memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                # optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            print('Finished episode loop')

        env.close()
        print('Reached end of train function')

def main():
    env = dreamwarrior.make_custom_env('../data/', 'NightmareOnElmStreet-Nes')

    model = DQN_Model(env)
    model.train()

if __name__ == '__main__':
    main()

"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        ""Saves a transition."
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# env.reset()




print('Complete')
env.render()
env.close()"""