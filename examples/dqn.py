"""DQN Test file
"""
import dreamwarrior
from dreamwarrior.models import DQN_Model

def main():
    env = dreamwarrior.make_custom_env('NightmareOnElmStreet-Nes', record=True)
    model = DQN_Model(env)
    # model.train()
    # model.save()
    model.load()
    model.run()

if __name__ == '__main__':
    main()
