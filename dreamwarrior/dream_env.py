"""
The custom Dream Warrior env class.
"""
import os
from collections import deque
import logging

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

import retro
from retro import RetroEnv

HISTORY_LENGTH = 4

class DreamEnv(RetroEnv):
    """DreamEnv is a child of the Gym Retro RetroEnv class. This class add a custom path for the
    games and a few functions to make training/playing easier.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    watching = False
    state_buffer = None

    def __init__(self, game, inttype=None, watching=False, **kwargs):
        if inttype is None:
            data_path = os.path.dirname(os.path.realpath(__file__))
            data_path += '/data'
            inttype = retro.data.Integrations.CUSTOM
            inttype.add_custom_path(os.path.abspath(data_path))

        super().__init__(game, inttype=inttype, **kwargs)

        self.watching = watching
        self.state_buffer = deque([], maxlen=4)

    def get_state(self):
        """Get retro env render as a torch tensor.

        Returns: A torch tensor made from the RGB pixels
        """
        # Transpose it into torch order (CHW).
        # screen = self.render(mode='rgb_array').transpose((2, 0, 1))
        screen = self.render(mode='rgb_array')

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # screen = torch.from_numpy(screen)

        # Resize, and add a batch dimension (BCHW)
        screen_transform = transforms.Compose([
            # screen,
            transforms.ToPILImage(),
            transforms.Resize(80, interpolation=Image.CUBIC),
            transforms.ToTensor()
        ])

        screen = screen_transform(screen)

        # return screen.unsqueeze(0).to(self.device)
        return screen.to(self.device)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        init_screen = self.get_state()
        _, screen_height, screen_width = init_screen.shape
        frame_buffer = torch.zeros(2, 3, screen_height, screen_width, device=self.device)

        total_reward, done, info = 0, False, None
        retro_action = np.zeros((9,), dtype=int)
        retro_action[action] = 1

        for t in range(4):
            _, reward, done, info = super().step(retro_action)
            total_reward += reward

            if self.watching:
                super().render()

            if t == 2:
                frame_buffer[0] = self.get_state()
            elif t == 3:
                frame_buffer[1] = self.get_state()

            if done:
                break

        state = frame_buffer.max(0)[0]
        self.state_buffer.append(state)

        return self.get_state(), total_reward, done, info

    def rainbow_step(self, action):
        # Return stack of state buffer instead of frames max pool
        _, reward, done, info = self.step(action)
        return torch.stack(list(self.state_buffer), 0), reward, done

    def _reset_buffer(self):
        for _ in range(HISTORY_LENGTH):
            self.state_buffer.append(torch.zeros(3, 224, 240, device=self.device))

    def rainbow_reset(self):
        self._reset_buffer()
        super().reset()
        state = self.get_state()[0]
        # for _ in range(4):
        #     self.state_buffer.append(state)
        return torch.stack(list(self.state_buffer), 0)
