import os
import retro

from sys import exit

def main():
    integrations = retro.data.Integrations.CUSTOM
    integrations.add_custom_path(os.path.abspath('.'))
    env = retro.make(game='NightmareOnElmStreet-Nes', inttype=integrations, record='.')

    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
            env.close()


if __name__ == "__main__":
    main()