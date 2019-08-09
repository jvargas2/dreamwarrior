import torch
from dreamwarrior.agents import DQNAgent

class CategoricalDQNAgent(DQNAgent):

    def __init__(self, env, config):
        super().__init__(env, config)

        self.batch_size = config.batch_size
        self.atoms = config.atoms # number of atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms, device=self.device)

        self.offset = torch.linspace(
            0,
            (self.batch_size - 1) * self.atoms,
            self.batch_size,
            device=self.device
        ).long().unsqueeze(1).expand(self.batch_size, self.atoms)

    def select_action(self, state):
        state = state.unsqueeze(0)
        distribution = self.model(state)
        distribution = distribution * self.support
        action_values = distribution.sum(2)
        action = action_values.max(1)[1].item()
        return action

    def projection_distribution(self, next_state, rewards, dones):
        atoms = self.atoms
        v_min = self.v_min
        v_max = self.v_max
        delta_z = self.delta_z
        
        target_next_distribution = self.target_model(next_state) * self.support
        next_action = None

        if self.double:
            online_next_distribution = self.model(next_state) * self.support
            next_action = online_next_distribution.sum(2).max(1)[1]
        else:
            next_action = target_next_distribution.sum(2).max(1)[1]

        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(
            target_next_distribution.size(0),
            1,
            target_next_distribution.size(2)
        )
        next_distribution = target_next_distribution.gather(1, next_action).squeeze(1)
            
        rewards = rewards.expand_as(next_distribution)
        dones = dones.expand_as(next_distribution)
        support = self.support.unsqueeze(0).expand_as(next_distribution)
        
        Tz = rewards + (1 - dones) * self.gamma * support
        Tz = Tz.clamp(min=v_min, max=v_max)
        b  = (Tz - v_min) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()

        # Calculate projected distribution
        projection = torch.zeros(next_distribution.size(), device=self.device)
        projection.view(-1).index_add_(
            0,
            (l + self.offset).view(-1),
            (next_distribution * (u.float() - b)
        ).view(-1))
        projection.view(-1).index_add_(
            0,
            (u + self.offset).view(-1),
            (next_distribution * (b - l.float())
        ).view(-1))
            
        return projection

    def calculate_loss(self, transitions, weights):
        state, action, reward, next_state, done = transitions

        projected_distribution = self.projection_distribution(next_state, reward, done)

        distribution = self.model(state)
        action = action.unsqueeze(1).expand(self.batch_size, 1, self.atoms)
        distribution = distribution.gather(1, action).squeeze(1)
        distribution.data.clamp_(0.01, 0.99)
        loss = -(projected_distribution * distribution.log()).sum(1)
        priorities = torch.abs(loss) + 1e-5
        loss = loss.mean()

        return loss, priorities
