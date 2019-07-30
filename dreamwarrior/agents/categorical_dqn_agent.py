import torch

from dreamwarrior.agents import DQNAgent

#
import math

class CategoricalDQNAgent(DQNAgent):
    frame = 0

    def __init__(self, env, config):
        super().__init__(env, config)
        init_screen = env.get_full_state()

        self.target_model = self.model_class(init_screen.shape, self.num_actions, config.atoms).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.batch_size = config.batch_size
        self.atoms = config.atoms # number of atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.delta_z = float(self.v_max - self.v_min) / (self.atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms)

    def projection_distribution(self, next_state, rewards, dones):
        atoms = self.atoms
        v_min = self.v_min
        v_max = self.v_max
        delta_z = self.delta_z
        support = self.support

        batch_size = next_state.size(0)
        
        next_distribution = self.target_model.c51_forward(next_state).data.cpu() * support
        next_action = next_distribution.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(
            next_distribution.size(0),
            1,
            next_distribution.size(2)
        )
        next_distribution = next_distribution.gather(1, next_action).squeeze(1)
            
        rewards = rewards.expand_as(next_distribution)
        dones = dones.expand_as(next_distribution)
        support = support.unsqueeze(0).expand_as(next_distribution)
        
        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=v_min, max=v_max)
        b  = (Tz - v_min) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()
            
        offset = torch.linspace(
            0,
            (batch_size - 1) * atoms,
            batch_size
        ).long().unsqueeze(1).expand(batch_size, atoms)

        # Calculate projected distribution
        projection = torch.zeros(next_distribution.size())
        projection.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_distribution * (u.float() - b)
        ).view(-1))
        projection.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_distribution * (b - l.float())
        ).view(-1))
            
        return projection

    def optimize_model(self, optimizer, memory, frame=None):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return

        indices, weights, priorities = None, None, None

        if self.prioritized_memory:
            state, action, reward, next_state, done, indices, weights = memory.sample(frame)
        else:
            state, action, reward, next_state, done = memory.sample()

        projected_distribution = self.projection_distribution(next_state, reward, done)

        distribution = self.model.c51_forward(state)
        action = action.unsqueeze(1).expand(self.batch_size, 1, self.atoms)
        distribution = distribution.gather(1, action).squeeze(1)
        distribution.data.clamp_(0.01, 0.99)
        loss = -(projected_distribution * distribution.log()).sum(1)
        priorities = loss + 1e-5
        loss = loss.mean()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if self.noisy:
            self.model.reset_noise()
            self.target_model.reset_noise()

        # Update target if appropriate
        if self.frame > self.frame_update:
            self.update_target()
            self.frame = 0
        else:
            self.frame += self.frame_skip

        return loss, indices, priorities

    def act(self, state, frame_count):
        state = state.unsqueeze(0)
        distribution = self.model.c51_forward(state).data.cpu()
        distribution = distribution * torch.linspace(self.v_min, self.v_max, self.atoms)
        action = distribution.sum(2).max(1)[1].item()
        return action

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, path=None):
        if path is None:
            super().load()
        else:
            super().load(path=path)

        self.update_target()
