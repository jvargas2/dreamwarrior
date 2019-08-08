"""
For the package's command line commands.
"""
import argparse
import logging
import time
from datetime import datetime

import retro

from dreamwarrior import DreamEnv, DreamConfig
from dreamwarrior.trainers import DQNTrainer
from dreamwarrior.runners import Runner

def train(args, config):
    config.set_device(args.cuda)
    env = DreamEnv(config, args.game, name=args.name, watching=args.watching, record=True)

    trainer = DQNTrainer(env, config)
    trainer.train()

def run(args, config):
    runner = Runner(args.agent, args.cuda)
    runner.run()

def evaluate(args, config):
    runner = Runner(args.agent, args.cuda)
    runner.evaluate(args.episodes)

def play_movie(args, config):
    movie = retro.Movie(args.filename)
    movie.step()

    env = DreamEnv(
        config=config,
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
    parser.add_argument('-g', '--game', default='NightmareOnElmStreet-Nes', help='Name of game to use for environment.')
    parser.add_argument('-p', '--print_logs', action='store_true', help='Use to print logs to console.')
    parser.add_argument('-c', '--cuda', type=int, default=-1, help='Which CUDA device to use. Only supply integer.')
    parser.add_argument('-i', '--ini', help='.ini config file to use')
    subparsers = parser.add_subparsers(dest='command')

    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train a new agent.')
    parser_train.add_argument('name', default='dream', help='Name of model for properly naming files.')
    parser_train.add_argument('-w', '--watching', action='store_true', help='Use to have Gym Retro render the environment.')
    parser_train.set_defaults(func=train)

    # Run arguments
    parser_run = subparsers.add_parser('run', help='run the game with trained agents or human players')
    parser_run.add_argument('-a', '--agent', help='Saved agent to watch.')
    parser_run.set_defaults(func=run)

    # Evaluation arguments
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate agent performance over n episodes.')
    parser_evaluate.add_argument('-a', '--agent', help='Saved agent to watch.')
    parser_evaluate.add_argument('-e', '--episodes', default=30, type=int, help='Number of episodes')
    parser_evaluate.set_defaults(func=evaluate)

    # Movie arguments
    parser_movie = subparsers.add_parser('movie', help='Play a recording.')
    parser_movie.add_argument('filename', help='Path to .bk2 file') # TODO Make this required
    parser_movie.set_defaults(func=play_movie)

    # Parse initial args
    args = parser.parse_args()

    # Logging
    filename = '%s.log' % args.name if args.command == 'train' else 'dreamwarrior.log'
    logging.basicConfig(filename=filename, format='%(asctime)-15s: %(message)s', level=logging.INFO)
    if args.print_logs:
        logging.getLogger().addHandler(logging.StreamHandler())
    start_time = time.time()
    current_time = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H.%M.%S')
    logging.info('\nSTART TIME: ' + current_time)

    # Run proper function
    if args.ini is None:
        config = DreamConfig()
    else:
        config = DreamConfig(args.ini)
    args.func(args, config)

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
