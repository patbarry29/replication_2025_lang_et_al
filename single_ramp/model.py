import torch
import torch.nn as nn
from torch.distributions import Normal

class SharedActorCritic(nn.Module):
    def __init__(self, state_dim):
        super(SharedActorCritic, self).__init__()

        # Shared Feature Extractor (3 FC layers)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # Actor Network (Policy - 2 FC layers)
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Bounds the expectation to [0, 1]
        )

        # Trainable standard deviation parameter for the normal distribution
        self.actor_log_std = nn.Parameter(torch.full((1,), -2.0))

        # Critic Network (Value - 2 FC layers)
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        shared_features = self.shared(state)

        # Critic Output
        state_value = self.critic(shared_features)

        # Actor Output
        action_mean = self.actor_mean(shared_features)
        action_std = self.actor_log_std.exp()

        dist = Normal(action_mean, action_std)

        return dist, state_value