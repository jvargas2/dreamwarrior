import logging
from itertools import count
from collections import namedtuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from dreamwarrior import DreamEnv
from dreamwarrior.agents import DQNAgent, CategoricalDQNAgent

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Runner:
    def __init__(self, agent_file, watching=False, device_index=-1):
        data = torch.load(agent_file, map_location='cpu')
        game = data['game']
        agent_class = data['agent_class']
        config = data['config']
        config.set_device(device_index)
        self.device = config.device

        env = DreamEnv(config, game, watching=watching)

        if agent_class == 'DQNAgent':
            agent_class = DQNAgent
        elif agent_class == 'CategoricalDQNAgent':
            agent_class = CategoricalDQNAgent
        else:
            raise ValueError('%s is not a valid agent class.' % agent_class)

        self.env = env
        self.frame_skip = config.frame_skip

        agent = agent_class(env, config)
        agent.load(agent_file)
        agent.model.eval()
        self.agent = agent

    def run(self, final_run=True):
        env = self.env
        device = self.device

        # Initialize the environment and state
        env.reset()
        state = env.get_state()
        frame = 0
        final_reward = 0

        for t in count():
            action = self.agent.select_action(state)

            state, reward, done, _ = env.step(action)
            final_reward += reward

            if reward > 0:
                logging.debug('t=%i got reward: %g' % (t, reward))
            elif reward < 0:
                logging.debug('t=%i got penalty: %g' % (t, reward))

            if done:
                break

        if final_run:
            env.close()

        logging.info('Finished run.')
        logging.info('Final reward: %d' % final_reward)

        return final_reward

    def evaluate(self, runs=30):
        rewards = []
        final_run = False
    
        for i in range(runs):
            if i == runs - 1:
                final_run = True

            print('Running test %d...' % (i + 1))
            reward = self.run(final_run=final_run)
            rewards.append(reward)

        logging.info('Finished evaluation! Final rewards:')
        logging.info(rewards)
