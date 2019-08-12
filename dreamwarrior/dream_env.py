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
    def __init__(self, config, game, name=None, inttype=None, watching=False, **kwargs):
        self.frame = 1
        self.frame_skip = config.frame_skip
        self.device = config.device
        self.height = config.height
        self.width = config.width

        custom_data_directory = os.path.dirname(os.path.realpath(__file__))
        custom_data_directory += '/data'
        custom_games = os.listdir(custom_data_directory)

        if game in custom_games:
            inttype = retro.data.Integrations.CUSTOM
            inttype.add_custom_path(os.path.abspath(custom_data_directory))

        if inttype is not None:
            super().__init__(game, inttype=inttype, **kwargs)
        else:
            super().__init__(game, **kwargs)

        if name is not None:
            self.name = name
        else:
            self.name = 'unnamed'

        self.watching = watching
        self.episode = 1
        self.state_buffer = deque(maxlen=4)
        empty_state = self.get_frame()

        for _ in range(self.frame_skip):
            self.state_buffer.append(empty_state)


    def get_frame(self):
        """Get retro env render as a torch tensor.

        Returns: A torch tensor made from the RGB pixels
        """
        screen = self.render(mode='rgb_array')

        screen_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize([self.height, self.width], interpolation=Image.CUBIC),
            # transforms.CenterCrop([self.height + 10, self.width])
            transforms.ToTensor()
        ])

        screen = screen_transform(screen)

        return screen.to(self.device)

    def get_state(self):
        """Ensures the state will contain the full four states.
        """
        state = torch.cat(list(self.state_buffer))

        return state

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

        if type(action) == int:
            retro_action = np.zeros((9,), dtype=int)
            retro_action[action] = 1
            action = retro_action

        frame_buffer = deque(maxlen=2)

        for i in range(self.frame_skip):
            self.frame += 1
            _, reward, done, _ = super().step(action)
            total_reward += reward

            if self.watching:
                self.render()

            frame_buffer.append(self.get_frame())

            if done:
                break

        frame_buffer = torch.stack(list(frame_buffer))
        maxed_frame = frame_buffer.max(0)[0]
        self.state_buffer.append(maxed_frame)

        state = torch.cat(list(self.state_buffer))

        return state, total_reward, done, info
