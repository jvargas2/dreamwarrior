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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    name = 'unnamed'
    watching = False
    state_buffer = None
    episode = 1

    def __init__(self, game, name=None, inttype=None, watching=False, **kwargs):
        if inttype is None:
            data_path = os.path.dirname(os.path.realpath(__file__))
            data_path += '/data'
            inttype = retro.data.Integrations.CUSTOM
            inttype.add_custom_path(os.path.abspath(data_path))

        super().__init__(game, inttype=inttype, **kwargs)

        if name is not None:
            self.name = name

        self.watching = watching
        self.state_buffer = deque([], maxlen=4)
        empty_state = self.get_state()

        for _ in range(4):
            self.state_buffer.append(empty_state)

    def get_state(self):
        """Get retro env render as a torch tensor.

        Returns: A torch tensor made from the RGB pixels
        """
        screen = self.render(mode='rgb_array')

        screen_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(112, interpolation=Image.CUBIC),
            transforms.ToTensor()
        ])

        screen = screen_transform(screen)

        return screen.to(self.device)

    def get_full_state(self):
        """Ensures the state will contain the full four states.
        """
        single_state = self.get_state()
        self.state_buffer.append(single_state)
        full_state = torch.cat(list(self.state_buffer))

        return full_state

    def reset(self):
        """Mostly original code from RetroEnv. Be careful when changing.
        """
        if self.initial_state:
            self.em.set_state(self.initial_state)
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            # rel_statename = os.path.splitext(os.path.basename(self.statename))[0]
            # self.record_movie(os.path.join(self.movie_path, '%s-%s-%06d.bk2' % (self.gamename, rel_statename, self.movie_id)))
            movie_path = os.path.join(self.movie_path, '%s-recordings' % self.name)

            if not os.path.exists(movie_path):
                os.makedirs(movie_path)

            self.record_movie(os.path.join(
                movie_path,
                '%s-%04d.bk2' % (self.name, self.episode)
            ))

            self.episode += 1
            # self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.data.reset()
        self.data.update_ram()
        return self._update_obs()

    def retro_step(self, action):
        super().step(action)

    def step(self, action):
        total_reward, done, info = 0, False, None
        retro_action = np.zeros((9,), dtype=int)
        retro_action[action] = 1

        for _ in range(4):
            _, reward, done, info = super().step(retro_action)
            total_reward += reward

            if self.watching:
                self.render()

            self.state_buffer.append(self.get_state())

            if done:
                break

        state = torch.cat(list(self.state_buffer))

        return state, total_reward, done, info

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
