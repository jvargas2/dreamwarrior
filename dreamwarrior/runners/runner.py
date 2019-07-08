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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    env = None
    player_one = None

    def __init__(self, env, player_one):
        self.env = env

        agent = DQNAgent(env)
        agent.load(player_one)
        self.player_one = agent

    def run(self):
        env = self.env
        device = self.device

        # Initialize the environment and state
        env.reset()
        state = env.get_state()
        final_reward = 0

        for t in count():
            action = self.player_one.select_action(state)

            state, reward, done, _ = env.step(action)
            final_reward += reward

            if reward > 0:
                logging.info('t=%i got reward: %g' % (t, reward))
            elif reward < 0:
                logging.info('t=%i got penalty: %g' % (t, reward))

            if done:
                break

        env.close()
        logging.info('Finished run.')
        logging.info('Final reward: %d' % final_reward)
