from dreamwarrior.agents import BaseAgent
from dreamwarrior.models import DQN

class DQNAgent(BaseAgent):
    model = None

    def __init__(self, env):
        super().__init__(env)
        init_screen = env.get_full_state()
        self.model = DQN(init_screen.shape, self.num_actions).to(self.device)
