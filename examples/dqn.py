"""DQN Test file
"""
from sys import argv
import logging
import dreamwarrior
from dreamwarrior.models import DQN_Model

def main():
    logging.basicConfig(filename='pytorch.log', level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)
    env = dreamwarrior.make_custom_env('NightmareOnElmStreet-Nes', record=True)
    model = DQN_Model(env)

    command = argv[1]

    if command == 'train':
        model.train()
        model.save()
    elif command == 'run':
        model.load()
        model.run()
    else:
        print('Wrong command')

if __name__ == '__main__':
    main()
