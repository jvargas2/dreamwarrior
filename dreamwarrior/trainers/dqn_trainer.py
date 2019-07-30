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
from dreamwarrior.agents import DQNAgent, DoubleDQNAgent, CategoricalDQNAgent
from dreamwarrior.memory import ReplayMemory, PrioritizedReplayMemory

class DQNTrainer:
    device = None
    env = None
    agent = None

    def __init__(self, env, config):
        self.device = torch.device(config.device)
        self.env = env
        self.config = config

        if config.categorical:
            self.agent = CategoricalDQNAgent(env, config)
        elif config.double:
            self.agent = DoubleDQNAgent(env, config)
        else:
            self.agent = DQNAgent(env, config)

        self.prioritized = config.prioritized
        self.frame_limit = config.frame_limit
        self.learning_rate = config.learning_rate

    def train(self, frame=0, rewards=[], episode=1, optimizer_state=None):
        logging.info('Starting training...')
        env = self.env
        device = self.device
        
        # optimizer = optim.RMSprop(self.agent.get_parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(self.agent.get_parameters(), lr=self.learning_rate, eps=1.5e-4)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        if self.prioritized:
            memory = PrioritizedReplayMemory(self.config)
        else:
            memory = ReplayMemory(self.config)

        frame_count = frame
        episode_rewards = rewards

        # for i_episode in range(start_episode, NUM_FRAME):
        while frame_count < self.frame_limit:
            episode_reward = 0
            losses = []

            env.reset()
            state = env.get_full_state()

            for t in count():
                # action = self.training_select_action(state, frame_count)
                action = self.agent.act(state, frame_count)

                next_state, reward, done, _ = env.step(action)
                frame_count += 4

                if reward > 0:
                    # Only add reward to count if its positive
                    logging.info('t=%i got reward: %g' % (t, reward))
                    episode_reward += reward
                elif reward < 0:
                    logging.info('t=%i got penalty: %g' % (t, reward))
                else:
                    reward = -1

                # Store the transition in memory
                memory.push(state, action, reward, next_state, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = None
                if len(memory) >= memory.batch_size:
                    if self.prioritized:
                        loss, indices, priorities = self.agent.optimize_model(optimizer, memory, frame_count)
                    else:
                        loss, _, _ = self.agent.optimize_model(optimizer, memory, frame_count)

                if loss is not None:
                    if self.prioritized:
                        memory.update_priorities(indices, priorities)

                    losses.append(loss)
                    if t % 1000 == 0:
                        average_loss = sum(losses) / len(losses)
                        logging.info('t=%d loss: %f' % (t, average_loss))
                        losses = []

                if done or frame_count >= self.frame_limit:
                    break
                
            logging.info('Finished episode ' + str(episode))
            logging.info('Final reward: %d' % episode_reward)
            logging.info('Training Progress: %dk/%dk (%.2f%%)' % (
                frame_count / 1000,
                self.frame_limit / 1000,
                (frame_count / self.frame_limit) * 100
            ))
            episode += 1
            episode_rewards.append(episode_reward)
            self.save_progress(frame_count, episode_rewards, optimizer)

        self.agent.save()
        env.close()
        logging.info('Finished training! Final rewards per episode:')
        logging.info(episode_rewards)

    def save_progress(self, frame, rewards, optimizer):
        episode = len(rewards) + 1

        state = {
            'frame': frame,
            'rewards': rewards,
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
        self.train(state['frame'], state['rewards'], episode, state['optimizer'])
