import logging
import math
from statistics import mean
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
from dreamwarrior.agents import DQNAgent, CategoricalDQNAgent
from dreamwarrior.memory import ReplayMemory, PrioritizedReplayMemory

class DQNTrainer:
    device = None
    env = None
    agent = None

    def __init__(self, env, config):
        self.device = config.device
        self.env = env
        self.config = config

        if config.categorical:
            self.agent = CategoricalDQNAgent(env, config)
        else:
            self.agent = DQNAgent(env, config)

        self.prioritized = config.prioritized
        self.min_frames = config.min_frames
        self.frame_limit = config.frame_limit
        self.frame_skip = config.frame_skip
        self.episode_frame_max = config.episode_frame_max
        self.learning_rate = config.learning_rate
        self.adam_epsilon = config.adam_epsilon
        self.multi_step = config.multi_step

    def train(self):
        logging.info('Starting training...')
        env = self.env
        device = self.device
        
        optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )

        if self.prioritized:
            memory = PrioritizedReplayMemory(self.config)
        else:
            memory = ReplayMemory(self.config)

        episode = 1
        episode_rewards = []
        episode_losses = []

        while self.env.frame < self.frame_limit:
            episode_reward = 0
            losses = []

            env.reset()
            state = env.get_state()

            for t in count():
                action = self.agent.act(state)

                next_state, reward, done, _ = env.step(action)

                if reward > 0:
                    logging.debug('t=%i got reward: %g' % (t, reward))
                    episode_reward += reward
                elif reward < 0:
                    logging.debug('t=%i got penalty: %g' % (t, reward))
                    episode_reward += reward

                # Store the transition in memory
                memory.push(state, action, reward, next_state, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss, indices, priorities = None, None, None
                if len(memory) >= memory.batch_size and self.env.frame > self.min_frames:
                    loss, indices, priorities = self.agent.optimize_model(optimizer, memory)

                if loss is not None:
                    losses.append(loss)
                    if self.prioritized:
                        memory.update_priorities(indices, priorities)

                if done or self.env.frame >= self.frame_limit or t * 4 >= self.episode_frame_max:
                    break

            mean_loss = mean(losses) if len(losses) > 0 else 0
            episode_losses.append(mean_loss)
            frame = self.env.frame
                
            logging.info('Finished episode ' + str(episode))
            logging.info('Final reward: %d' % episode_reward)
            logging.info('Episode average loss: %f' % mean_loss)
            logging.info('Training Progress: %dk/%dk (%.2f%%)' % (
                frame / 1000,
                self.frame_limit / 1000,
                (frame / self.frame_limit) * 100
            ))
            episode += 1
            episode_rewards.append(episode_reward)
            self.agent.save()

        env.close()
        logging.info('Finished training! Final rewards per episode:')
        logging.info(episode_rewards)
        logging.info('----------Losses:')
        logging.info([float('%.4f' % loss) for loss in episode_losses])
