"""Non-cooperative agent for A Nightmare on Elm Street that just makes random moves.
"""
import retro
import dreamwarrior

def main():
    game = 'NightmareOnElmStreet-Nes'
    record = False
    observation_type = retro.Observations.IMAGE # can be RAM
    players = 1

    env = dreamwarrior.make_custom_env(
        game,
        record=record,
        players=players,
        obs_type=observation_type
    )

    env.reset()
    time = 0
    total_reward = 0

    while True:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        time += 1
        if time % 10 == 0:
            env.render()
        total_reward += reward

        if reward > 0:
            print('t=%i got reward: %g, current reward: %g' % (time, reward, total_reward))
        elif reward < 0:
            print('t=%i got penalty: %g, current reward: %g' % (time, reward, total_reward))

        if done:
            env.render()
            print("done! total reward: time=%i, reward=%r" % (time, total_reward))
            input('hit enter to end')
            break

if __name__ == '__main__':
    main()
