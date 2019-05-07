"""
Simple tool for playing .bk2 recordings. Based on example found at:
https://retro.readthedocs.io/en/latest/python.html#observations
"""
import argparse
import retro
import dreamwarrior

def play(bk2_path):
    print('Attempting to play %s' % bk2_path)
    movie = retro.Movie(bk2_path)
    movie.step()

    env = dreamwarrior.make_custom_env(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )

    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        env.step(keys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', help='the name or path of the .bk2 file to use')
    args = parser.parse_args()
    play(args.path)