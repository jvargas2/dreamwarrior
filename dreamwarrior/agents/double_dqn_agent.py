import torch
import torch.nn.functional as F

from dreamwarrior.agents import DQNAgent
from dreamwarrior.models import DQN, DuelingDQN

class DoubleDQNAgent(DQNAgent):
    frame = 0

    def __init__(self, env, model='dqn', device=None):
        super().__init__(env, model, device)
        init_screen = env.get_full_state()
        model_class = None

        if model == 'dueling-dqn':
            model_class = DuelingDQN
        else:
            model_class = DQN

        self.target_model = model_class(init_screen.shape, self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def optimize_model(self, optimizer, memory, gamma, frame):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return

        state, action, reward, next_state, done, indices, weights = memory.sample(frame)

        # Get Q values for every action in first and second states
        q_values = self.model(state)
        next_q_values = self.model(next_state)
        next_q_state_values = self.target_model(next_state)

        # Actual action-value selected
        q_value = q_values.gather(1, action)

        # Select evaluation values from target by using the indicies of the online model max
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))

        # Calculate expected return
        expected_q_value = reward + gamma * next_q_value * (1 - done)
        
        # Compute Huber loss
        loss = (q_value - expected_q_value.detach()).pow(2) * torch.tensor(weights).unsqueeze(1)
        # loss = F.smooth_l1_loss(q_value, expected_q_value)
        priorities = loss + 1e-5 # pi = |δi| + ε
        loss = loss.mean()
            
        optimizer.zero_grad()
        loss.backward()
        # update_priorities(indices, priorities.data.cpu().numpy())
        optimizer.step()

        # Update target if appropriate
        if self.frame > 10000:
            self.update_target()
            self.frame = 0
        else:
            self.frame += 4

        return loss, indices, priorities

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, path=None):
        if path is None:
            super().load()
        else:
            super().load(path=path)

        self.update_target()
