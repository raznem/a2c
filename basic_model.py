""""
CartPole-v0 version.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical, Normal, Independent


class Actor(nn.Module):
    def __init__(self, obs_dim: int, ac_dim: int, discrete: bool = True):
        super().__init__()

        self.ac_dim = ac_dim
        self.obs_dim = obs_dim
        self.discrete = discrete
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, ac_dim)
        if not self.discrete:
            self.log_scale = nn.Parameter(torch.ones(self.ac_dim), requires_grad=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.discrete:
            x = torch.softmax(self.fc2(x), dim=1)
        else:
            x = torch.tanh(self.fc2(x))
        return x

    def act(self, obs):
        action_prob = self.forward(obs)
        if self.discrete:
            dist = Categorical(action_prob)
        else:
            normal = Normal(action_prob, torch.exp(self.log_scale))
            dist = Independent(normal, 1)
        action = dist.sample()
        action_logprobs = dist.log_prob(torch.squeeze(action))
        return action, action_logprobs

class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x = 2 * torch.tanh(self.fc3(x))
        x = self.fc3(x)
        return x