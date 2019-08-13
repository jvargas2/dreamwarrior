import logging
from itertools import count
from collections import namedtuple
import time

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from dreamwarrior import DreamEnv, DreamConfig
from dreamwarrior.agents import DQNAgent, CategoricalDQNAgent

class Runner:
    def __init__(self, players, game=None, watching=False, device_index=-1):
        """
        Params:
            players: List of strings. Either an agent file name or 'human'
        """
        # Find best config to use
        config = None

        data = torch.load(players[0], map_location='cpu')
        config = data['config']
        
        if game is None:
            game = data['game']

        config.set_device(device_index)
        self.device = config.device

        env = DreamEnv(config, game, watching=watching)

        self.env = env
        self.frame_skip = config.frame_skip

        configured_players = []

        for player in players:
            agent = self.configure_agent(player, env)
            configured_players.append(agent)

        self.players = configured_players

    def configure_agent(self, agent_file, env):
        data = torch.load(agent_file, map_location='cpu')
        agent_class = data['agent_class']
        config = data['config']
        config.device = self.device

        if agent_class == 'DQNAgent':
            agent_class = DQNAgent
        elif agent_class == 'CategoricalDQNAgent':
            agent_class = CategoricalDQNAgent
        else:
            raise ValueError('%s is not a valid agent class.' % agent_class)

        agent = agent_class(env, config)
        agent.load(agent_file)
        agent.model.eval()

        return agent

    def run(self, final_run=True, random=False):
        env = self.env
        device = self.device

        # Initialize the environment and state
        env.reset()
        state = env.get_state()
        final_reward = 0

        for t in count():
            action = None

            for agent in self.players:
                if random:
                    action = agent.random_action()
                else:
                    action = agent.select_action(state)

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

    def evaluate(self, runs=30, random=False):
        rewards = []
        final_run = False
    
        for i in range(runs):
            if i == runs - 1:
                final_run = True

            print('Running test %d...' % (i + 1))
            reward = self.run(final_run=final_run, random=random)
            rewards.append(reward)

        logging.info('Finished evaluation! Final rewards:')
        logging.info(rewards)
