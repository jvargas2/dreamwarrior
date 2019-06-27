import argparse
import logging
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import dreamwarrior
from dreamwarrior.agents import DQNAgent
from dreamwarrior.memory import ReplayMemory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
NUM_EPISODES = 100
FRAME_SKIP = 4

class DQNTrainer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = None
    agent = None

    def __init__(self, env):
        self.env = env
        self.agent = DQNAgent(env)

    def training_select_action(self, state, steps_done):
        # Select and perform an action
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if sample > eps_threshold:
            action = self.agent.select_action(state)
        else:
            action = self.agent.random_action()

        return action

    def train(self, optimizer_state=None, start_episode=0):
        logging.info('Starting training...')
        env = self.env
        device = self.device
        
        self.agent.start_target_net()

        optimizer = optim.RMSprop(self.agent.policy_net.parameters())
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        memory = ReplayMemory(10000)

        steps_done = 0
        episode_rewards = []

        for i_episode in range(start_episode, NUM_EPISODES):
            # Initialize the environment and state
            episode_reward = 0

            env.reset()
            state = env.get_state()

            for t in count():
                action = self.training_select_action(state, steps_done)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                reward = torch.tensor([reward], device=device)

                if reward > 0:
                    logging.info('t=%i got reward: %g' % (t, reward))
                elif reward < 0:
                    logging.info('t=%i got penalty: %g' % (t, reward))

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = self.agent.optimize_model(optimizer, memory, GAMMA)

                if t % 1000 == 0 and loss is not None:
                    logging.info('t=%d loss: %f' % (t, loss))

                if done:
                    break

            # Update the target network, copying all weights and biases in DQN
            self.agent.update_target_net()

            logging.info('Finished episode ' + str(i_episode))
            logging.info('Final reward: %d' % episode_reward)
            episode_rewards.append(episode_reward)
            self.save_progress(i_episode, optimizer)

        self.agent.save()
        env.close()
        logging.info('Finished training! Final rewards per episode:')
        logging.info(episode_rewards)

    def save_progress(self, episode, optimizer):
        state = {
            'episode': episode,
            'optimizer': optimizer.state_dict(),
            'model': self.agent.get_state_dict()
        }
        torch.save(state, 'training_progress.pth')
        logging.info('Saved training after finishing episode %d.' % episode)

    def continue_training(self, filepath, watching=False):
        state = torch.load(filepath, map_location=self.device)
        episode = state['episode'] + 1

        self.agent.load_policy_net(state['model'])
        logging.info('Continuing training at episode %d...' % episode)
        self.train(optimizer_state=state['optimizer'], start_episode=episode, watching=watching)
