"""
For the package's command line commands.
"""
import argparse
import logging
import time
from datetime import datetime

import retro

import dreamwarrior
from dreamwarrior import DreamEnv
from dreamwarrior.trainers import DQNTrainer
from dreamwarrior.agents import DQNAgent, DoubleDQNAgent
from dreamwarrior.runners import Runner

def train(args):
    env = DreamEnv('NightmareOnElmStreet-Nes', name=args.name, watching=args.watching, record=True)
    model = args.model
    agent = None

    if model == 'dqn':
        agent = DQNAgent(env, model)
    elif model == 'ddqn':
        agent = DoubleDQNAgent(env, model)
    elif model == 'dueling-dqn':
        agent = DoubleDQNAgent(env, model)

    trainer = DQNTrainer(env, agent)

    if args.continue_file:
        trainer.continue_training(args.continue_file)
    else:
        trainer.train()

def run(args):
    env = DreamEnv('NightmareOnElmStreet-Nes', watching=True)

    runner = Runner(env, args.agent)
    runner.run()

def play_movie(args):
    movie = retro.Movie(args.filename)
    movie.step()

    env = DreamEnv(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players
    )

    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))

        env.retro_step(keys)
        env.render()

def main():
    """Main function for parsing command line arguments.
    """
    # Argparse setup
    parser = argparse.ArgumentParser(prog='dreamwarrior', description='Train and test Gym Retro agents.')
    parser.add_argument('-p', '--print_logs', action='store_true', help='Use to print logs to console.')
    subparsers = parser.add_subparsers()

    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train a new agent.')
    parser_train.add_argument('-m', '--model', choices=['dqn', 'ddqn', 'dueling-dqn'], default='dqn', help='Type of model to use for agent.')
    parser_train.add_argument('-w', '--watching', action='store_true', help='Use to have Gym Retro render the environment.')
    parser_train.add_argument('-n', '--name', help='Name of model for properly naming files.')
    parser_train.add_argument('-c', '--continue_file', help='.pth path when continuing training.')
    parser_train.set_defaults(func=train)

    # run arguments
    parser_run = subparsers.add_parser('run', help='run the game with trained agents or human players')
    parser_run.add_argument('-a', '--agent', help='Saved agent to watch.')
    parser_run.set_defaults(func=run)

    # Movie arguments
    parser_movie = subparsers.add_parser('movie', help='Play a recording.')
    parser_movie.add_argument('filename', help='Path to .bk2 file') # TODO Make this required
    parser_movie.set_defaults(func=play_movie)

    # Parse initial args
    args = parser.parse_args()

    # Logging
    logging.basicConfig(filename='dreamwarrior.log', format='%(asctime)-15s: %(message)s', level=logging.INFO)
    if args.print_logs:
        logging.getLogger().addHandler(logging.StreamHandler())
    start_time = time.time()
    current_time = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('\nSTART TIME: ' + current_time)

    # Run proper function
    args.func(args)

    # Calculate/print end time
    end_time = time.time()
    current_time = datetime.fromtimestamp(end_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('END TIME: ' + current_time)

    # Calculate/print time elapsed
    seconds_elapsed = end_time - start_time
    minutes_elapsed, seconds_elapsed = divmod(seconds_elapsed, 60)
    hours_elapsed, minutes_elapsed = divmod(minutes_elapsed, 60)

    logging.info('H:M:S ELAPSED: %d:%d:%d' % (hours_elapsed, minutes_elapsed, seconds_elapsed))


if __name__ == '__main__':
    main()
