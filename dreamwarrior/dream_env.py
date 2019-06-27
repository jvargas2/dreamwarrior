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
        self.state_buffer = deque([], maxlen=int(108e3))

    def get_state(self):
        """Get retro env render as a torch tensor.

        Returns: A torch tensor made from the RGB pixels
        """
        # Transpose it into torch order (CHW).
        screen = self.render(mode='rgb_array').transpose((2, 0, 1))

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

    def step(self, action):
        frame_buffer = torch.zeros(2, 224, 240, device=self.device)
        reward, done, info = 0, False, None

        for t in range(4):
            retro_action = np.zeros((9,), dtype=int)
            retro_action[action.item()] = 1
            _, reward, done, info = super().step(retro_action)

            if self.watching:
                super().render()

            if t == 2:
                # frame_buffer[0] = self.get_state()
                penultimate_state = self.get_state()
            elif t == 3:
                # frame_buffer[1] = self.get_state()
                ultimate_state = self.get_state()

            if done:
                break

        # observation = frame_buffer.max(0)[0]
        state = ultimate_state - penultimate_state
        self.state_buffer.append(state)

        return state, reward, done, info
