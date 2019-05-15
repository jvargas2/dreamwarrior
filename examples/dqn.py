"""DQN Test file
"""
import dreamwarrior
from dreamwarrior.models import DQN_Model

def main():
    env = dreamwarrior.make_custom_env('../dreamwarrior/data/', 'NightmareOnElmStreet-Nes', record=True)
    model = DQN_Model(env)
    model.train()

if __name__ == '__main__':
    main()
