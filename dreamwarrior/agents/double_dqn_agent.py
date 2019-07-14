import torch

from dreamwarrior.agents import DQNAgent

class DoubleDQNAgent(DQNAgent):
    frame = 0

    def __init__(self, env, config):
        super().__init__(env, config)
        init_screen = env.get_full_state()

        self.target_model = self.model_class(init_screen.shape, self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

    def optimize_model(self, optimizer, memory, frame=None):
        """Optimize the model.
        """
        if len(memory) < memory.batch_size:
            return

        indices, weights = None, None

        if self.prioritized_memory:
            state, action, reward, next_state, done, indices, weights = memory.sample(frame)
        else:
            state, action, reward, next_state, done = memory.sample()

        # Get Q values for every action in first and second states
        q_values = self.model(state)
        next_q_values = self.model(next_state)
        next_q_state_values = self.target_model(next_state)

        # Actual action-value selected
        q_value = q_values.gather(1, action)

        # Select evaluation values from target by using the indicies of the online model max
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))

        # Calculate expected return
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        # Compute Huber loss
        loss, priorities = self.calculate_loss(q_value, expected_q_value, weights)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target if appropriate
        if self.frame > self.frame_update:
            self.update_target()
            self.frame = 0
        else:
            self.frame += self.frame_skip

        return loss, indices, priorities

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, path=None):
        if path is None:
            super().load()
        else:
            super().load(path=path)

        self.update_target()
