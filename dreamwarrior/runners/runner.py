import logging
from itertools import count
from collections import namedtuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from dreamwarrior.agents import DQNAgent, DoubleDQNAgent, CategoricalDQNAgent

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Runner:
    def __init__(self, env, agent_file):
        data = torch.load(agent_file, map_location='cpu')
        agent_class = data['agent_class']
        config = data['config']

        if agent_class == 'DQNAgent':
            agent_class = DQNAgent
        elif agent_class == 'DoubleDQNAgent':
            agent_class = DoubleDQNAgent
        elif agent_class == 'CategoricalDQNAgent':
            agent_class = CategoricalDQNAgent
        else:
            raise ValueError('%s is not a valid agent class.' % agent_class)

        self.device = config.device
        self.env = env
        self.frame_skip = config.frame_skip

        agent = agent_class(env, config)
        agent.load(agent_file)
        agent.model.eval()
        self.agent = agent

    def run(self):
        env = self.env
        device = self.device

        # Initialize the environment and state
        env.reset()
        state = env.get_state()
        frame = 0
        final_reward = 0

        for t in count():
            action = self.agent.select_action(state, frame)

            state, reward, done, _ = env.step(action)
            frame += self.frame_skip
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
