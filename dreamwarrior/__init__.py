import os
import retro
from retro.enums import State

def make_custom_env(game, state=State.DEFAULT, **kwargs):
    integrations = retro.data.Integrations.CUSTOM
    integrations.add_custom_path(os.path.abspath('../data'))
    env = retro.make(game='NightmareOnElmStreet-Nes', state=state, inttype=integrations, **kwargs)
    return env