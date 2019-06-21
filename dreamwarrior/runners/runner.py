import logging
from itertools import count
from collections import namedtuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from dreamwarrior.agents import DQNAgent

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Runner:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = None
    player_one = None

    def __init__(self, env, player_one):
        self.env = env

        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape

        agent = DQNAgent(env, screen_height, screen_width)
        agent.load(player_one)
        self.player_one = agent

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
            action = self.player_one.select_action(state)

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
