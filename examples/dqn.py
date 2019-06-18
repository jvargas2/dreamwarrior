"""DQN Test file
"""
from sys import argv
import logging
import dreamwarrior
from dreamwarrior.models import DQN_Model

def main():
    logging.basicConfig(filename='pytorch.log', level=logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    command = argv[1]

    record = False
    if command == 'train':
        record = True

    env = dreamwarrior.make_custom_env('NightmareOnElmStreet-Nes', record=record)
    model = DQN_Model(env)

    if command == 'train':
        if len(argv) < 3:
            model.train()
        else:
            model.continue_training(argv[2])
        model.save()
            
    elif command == 'run':
        if len(argv) < 3:
            model.load()
        else:
            model.load(argv[2])
        model.run()
    else:
        print('Wrong command')

if __name__ == '__main__':
    main()
