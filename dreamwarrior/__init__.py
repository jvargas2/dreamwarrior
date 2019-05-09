"""Init file for dreamwarrior package. Currently just helper functions for gym retro.
"""
import os
import retro
from retro.enums import State

# Environments

def make_custom_env(path, game, state=State.DEFAULT, **kwargs):
    """Uses retro.make() but with local custom games path
    """
    integrations = retro.data.Integrations.CUSTOM
    # TODO: Make path relative to package or a param
    integrations.add_custom_path(os.path.abspath(path))
    env = retro.make(
        game=game,
        state=state,
        inttype=integrations,
        **kwargs
    )
    return env

# Movies

def play_movie(bk2_path):
    """Plays recording from a .bk2 file.
    Based on example found at: https://retro.readthedocs.io/en/latest/python.html#observations

    Args:
        bk2_path: path to the .bk2 file
    """
    print('Attempting to play %s' % bk2_path)
    movie = retro.Movie(bk2_path)
    # movie.step()

    env = make_custom_env(
        path='../data/',
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
    )
    print('Made env')

    env.initial_state = movie.get_state()
    env.reset()
    time = 0

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        env.step(keys)

        if time % 10 == 0:
            env.render()

        time += 1
