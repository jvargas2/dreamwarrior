"""
For the package's command line commands.
"""
import argparse
import logging

import dreamwarrior
from dreamwarrior.trainers import DQNTrainer
from dreamwarrior.runners import Runner

import sys # TODO Remove

def train(args):
    env = dreamwarrior.make_custom_env('NightmareOnElmStreet-Nes', record=True)

    if args.model == 'dqn':
        trainer = DQNTrainer(env)
        trainer.train(watching=args.watching)

def run(args):
    env = dreamwarrior.make_custom_env('NightmareOnElmStreet-Nes')
    
    runner = Runner(env, args.agent)
    runner.run()

def movie(args):
    dreamwarrior.play_movie(args.filename)

def main():
    """Main function for parsing command line arguments.
    """
    # Argparse setup
    parser = argparse.ArgumentParser(prog='dreamwarrior', description='Train and test Gym Retro agents.')
    parser.add_argument('-p', '--print_logs', action='store_true', help='Use to print logs to console.')
    subparsers = parser.add_subparsers()

    # Train arguments
    parser_train = subparsers.add_parser('train', help='Train a new agent.')
    parser_train.add_argument('-m', '--model', choices=['dqn'], default='dqn', help='Type of model to use for agent.')
    parser_train.add_argument('-w', '--watching', action='store_true', help='Use to have Gym Retro render the environment.')
    parser_train.set_defaults(func=train)

    # run arguments
    parser_run = subparsers.add_parser('run', help='run the game with trained agents or human players')
    parser_run.add_argument('-a', '--agent', help='Saved agent to watch.')
    parser_run.set_defaults(func=run)

    # Movie arguments
    parser_movie = subparsers.add_parser('movie', help='Play a recording.')
    parser_movie.add_argument('filename', help='Path to .bk2 file') # TODO Make this required
    parser_movie.set_defaults(func=movie)

    # Parse initial args
    args = parser.parse_args()

    # Logging
    logging.basicConfig(filename='dreamwarrior.log', level=logging.INFO)

    if args.print_logs:
        logging.getLogger().addHandler(logging.StreamHandler())

    # Run proper function
    args.func(args)

if __name__ == '__main__':
    main()
