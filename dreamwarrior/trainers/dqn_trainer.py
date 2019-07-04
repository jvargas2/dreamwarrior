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

LEARNING_RATE = 0.0000625
BATCH_SIZE = 32
GAMMA = 0.999
FRAME_LIMIT = int(1e7) # 10 million
FRAME_SKIP = 4
LEARNING_RATE = 0.00001
MEMORY_SIZE = int(2e5) # 100k

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = int(5e4)

class DQNTrainer:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = None
    agent = None

    def __init__(self, env):
        self.env = env
        self.agent = DQNAgent(env)

    def training_select_action(self, state, frame_count):
        # Select and perform an action
        sample = random.random()
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            math.exp(-1. * frame_count / EPSILON_DECAY)
        
        if sample > eps_threshold:
            action = self.agent.select_action(state)
        else:
            action = self.agent.random_action()

        return action

    def train(self, frame=0, optimizer_state=None, episode=1):
        logging.info('Starting training...')
        env = self.env
        device = self.device
        
        optimizer = optim.RMSprop(self.agent.get_parameters(), lr=LEARNING_RATE)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        memory = ReplayMemory(MEMORY_SIZE, BATCH_SIZE)

        frame_count = frame
        episode_rewards = []

        # for i_episode in range(start_episode, NUM_FRAME):
        while frame_count < FRAME_LIMIT:
            episode_reward = 0
            losses = []

            env.reset()
            frame_count += 1
            state = env.get_state()

            for t in count():
                action = self.training_select_action(state, frame_count)

                next_state, reward, done, _ = env.step(action)
                frame_count += 4
                episode_reward += reward

                if reward > 0:
                    logging.info('t=%i got reward: %g' % (t, reward))
                elif reward < 0:
                    logging.info('t=%i got penalty: %g' % (t, reward))

                # Store the transition in memory
                memory.push(state, action, reward, next_state, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = self.agent.optimize_model(optimizer, memory, GAMMA)

                if loss is not None:
                    losses.append(loss)
                    if t % 1000 == 0:
                        average_loss = sum(losses) / len(losses)
                        logging.info('t=%d loss: %f' % (t, average_loss))
                        losses = []

                if done or frame_count >= FRAME_LIMIT or t > 100:
                    break

            logging.info('Finished episode ' + str(episode))
            logging.info('Final reward: %d' % episode_reward)
            logging.info('Training Progress: %dk/%dk (%.2f%%)' % (
                frame_count / 1000,
                FRAME_LIMIT / 1000,
                (frame_count / FRAME_LIMIT) * 100
            ))
            episode += 1
            episode_rewards.append(episode_reward)
            self.save_progress(frame_count, episode, optimizer)

        self.agent.save()
        env.close()
        logging.info('Finished training! Final rewards per episode:')
        logging.info(episode_rewards)

    def save_progress(self, frame, episode, optimizer):
        state = {
            'frame': frame,
            'episode': episode,
            'optimizer': optimizer.state_dict(),
            'model': self.agent.get_state_dict()
        }
        torch.save(state, 'training_progress.pth')
        logging.info('Saved training after finishing episode %s.' % str(episode - 1))

    def continue_training(self, filepath):
        state = torch.load(filepath, map_location=self.device)
        episode = state['episode']

        self.agent.load_state_dict(state['model'])
        self.env.episode = episode

        logging.info('Continuing training at episode %d...' % episode)
        self.train(state['frame'], state['optimizer'], episode)
