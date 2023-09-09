import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from .cnns import CNN
from .mlps import GaussianMLP
from .utils import TanhDistribution

class MLPPolicyNetwork(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dim):
        super().__init__()
        obs_dim = np.prod(obs_shape)
        act_dim = np.prod(act_shape)
        hidden_dims = [hidden_dim for _ in range(2)]
        self.net = GaussianMLP(
            obs_dim, hidden_dims, act_dim, max_logvar=4, min_logvar=-40
        )

    def forward(self, obs, deterministic=False):
        if deterministic:
            mean, _ = self.net.forward(obs)
            act = torch.tanh(mean)
            log_prob, entropy = None, None
        else:
            normal = self.net.forward_dist(obs)
            act = normal.rsample()
            log_prob = normal.log_prob(act).sum(-1, keepdim=True)
            entropy = normal.entropy().sum(-1, keepdim=True)
        return act, log_prob, entropy

    def evaluate(self, obs, act):
        normal = self.net.forward_dist(obs)
        log_prob = normal.log_prob(act).sum(-1, keepdim=True)
        entropy = normal.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, normal.mean, normal.stddev
    
# class MLPPolicyNetwork(nn.Module):
#     def __init__(self, obs_shape, act_shape, hidden_dim):
#         super().__init__()
#         obs_dim = np.prod(obs_shape)
#         act_dim = np.prod(act_shape)
#         hidden_dims = [hidden_dim for _ in range(2)]
#         self.net = GaussianMLP(
#             obs_dim, hidden_dims, act_dim, max_logvar=4, min_logvar=-40
#         )

#     def forward(self, obs, deterministic=False):
#         if deterministic:
#             mean, _ = self.net.forward(obs)
#             act = torch.tanh(mean)
#             log_prob, entropy = None, None
#         else:
#             normal = self.net.forward_dist(obs)
#             act = normal.rsample()
#             log_prob = normal.log_prob(act).sum(-1, keepdim=True)
#             entropy = normal.entropy().sum(-1, keepdim=True)
#         return act, log_prob, entropy

#     def evaluate(self, obs, act):
#         normal = self.net.forward_dist(obs)
#         log_prob = normal.log_prob(act).sum(-1, keepdim=True)
#         entropy = normal.entropy().sum(-1, keepdim=True)
#         return log_prob, entropy

class GaussianPolicy(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_dim):
        super().__init__()
        self.pixel_obs = len(obs_shape) == 3
        if self.pixel_obs:
            self.encoder = CNN(
                input_chn=obs_shape[0],
                output_dim=hidden_dim,
                output_act="ReLU",
            )
            input_dim = hidden_dim
        else:
            input_dim = np.prod(obs_shape)
        hidden_dims = [hidden_dim for _ in range(2)]
        self.head = GaussianMLP(input_dim, hidden_dims, np.prod(act_shape))

    def forward_dist(self, obs):
        if self.pixel_obs:
            obs = self.encoder(obs)
        return self.head.forward_dist(obs)

    def forward(self, obs, deterministic=False):
        # Return action and log prob
        dist = self.forward_dist(obs)
        if deterministic:
            act = dist.mean
            log_prob = None
        else:
            act = dist.rsample()
            log_prob = dist.log_prob(act).sum(-1, True)
        return act, log_prob

    def evaluate(self, obs, act):
        # Return log prob
        dist = self.forward_dist(obs)
        log_prob = dist.log_prob(act).sum(-1, True)
        return log_prob


class TanhGaussianPolicy(GaussianPolicy):
    def __init__(self, obs_shape, act_shape, hidden_dim, act_space=None):
        super().__init__(obs_shape, act_shape, hidden_dim)
        if act_space is None:
            self.loc = torch.tensor(0.0)
            self.scale = torch.tensor(1.0)
        else:
            self.loc = torch.tensor((act_space.high + act_space.low) / 2.0)
            self.scale = torch.tensor((act_space.high - act_space.low) / 2.0)

    def forward_dist(self, obs):
        dist = super().forward_dist(obs)
        return TanhDistribution(dist, self.loc, self.scale)


class EntropyGaussianPolicy(GaussianPolicy):
    def forward(self, obs, deterministic=False):
        # Return action, log prob, and entropy
        dist = self.forward_dist(obs)
        if deterministic:
            act = dist.mean
            log_prob, entropy = None, None
        else:
            act = dist.rsample()
            log_prob = dist.log_prob(act).sum(-1, True)
            entropy = dist.entropy().sum(-1, True)
        return act, log_prob, entropy

    def evaluate(self, obs, act):
        # Return log prob and entropy
        dist = self.forward_dist(obs)
        log_prob = dist.log_prob(act).sum(-1, True)
        entropy = dist.entropy().sum(-1, True)
        return log_prob, entropy


class RNNPolicy(nn.Module):
    def __init__(
            self,
            obs_size,
            act_size, 
            belief_size=256,
            hidden_size=256,
            min_std=1e-4,
            sphere_vel_in_obs=True,
        ):
        
        super().__init__()
        self.obs_size = obs_size
        self.act_size = act_size
        self.belief_size = belief_size
        self.hidden_size = hidden_size
        self.min_std = min_std

        self.sphere_vel_in_obs = sphere_vel_in_obs
        if self.sphere_vel_in_obs:
            self.obs_size -= 3
            
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.belief_size)
        )
        
        # RNN
        self.rnn = nn.GRUCell(input_size=self.belief_size, hidden_size=self.belief_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * act_size),
        )

    def init_belief(self, batch_size, device):
        return torch.zeros((batch_size, self.belief_size)).to(device)

    def forward(self, obs):

        # remove sphere velocity from observation
        if self.sphere_vel_in_obs:
            obs = torch.cat([obs[...,:6], obs[...,9:]], dim=-1)
        
        # Input shape (T, B, dim)
        T, B = obs.shape[:2]

        # Initialize belief
        belief = self.init_belief(B, device=obs.device)

        # Encode observations
        embeds = self.encoder(obs)

        beliefs = [torch.empty(0)] * T
        for t in range(T):
            belief = self.rnn(embeds[t], belief)
            beliefs[t] = belief
        beliefs = torch.stack(beliefs, dim=0)

        action_means, action_stds = self.decoder(beliefs).chunk(2, -1)
        action_stds = self.min_std + F.softplus(action_stds)
        return action_means, action_stds
    
    def step(self, prev_belief, obs):
        # remove sphere velocity from observation
        if self.sphere_vel_in_obs:
            obs = torch.cat([obs[...,:6], obs[...,9:]], dim=-1)
        hidden = self.encoder(obs)
        belief = self.rnn(hidden, prev_belief)
        action_mean, action_std = self.decoder(belief).chunk(2, -1)
        action_std = self.min_std + F.softplus(action_std)
        action = action_mean + torch.randn_like(action_mean) * action_std
        return belief, action
 
    def compute_loss(self, obs, actions, rewards):
        means, stds = self.forward(obs)
        dists = Independent(Normal(means, stds), 1)
        log_probs = dists.log_prob(actions)
        loss = -(log_probs * rewards.exp()).mean()
        return loss
    

class PixelRNNPolicy(RNNPolicy):
    def __init__(
            self,
            img_size,
            act_size,
            belief_size=256,
            hidden_size=256,
            min_std=1e-4,
        ):    
        super().__init__(
            obs_size=hidden_size,
            act_size=act_size,
            belief_size=belief_size,
            hidden_size=hidden_size,
            min_std=min_std,
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_size[0], 32, 3, stride=2),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, belief_size),
        )
    
    def forward(self, obs):
        # Input shape (T, B, dim)
        T, B = obs.shape[:2]

        # Initialize belief
        belief = self.init_belief(B, device=obs.device)

        # Encode observations
        embeds = self.encoder(obs.reshape(T * B, *obs.shape[2:]))
        embeds = embeds.reshape(T, B, -1)

        beliefs = [torch.empty(0)] * T
        for t in range(T):
            belief = self.rnn(embeds[t], belief)
            beliefs[t] = belief
        beliefs = torch.stack(beliefs, dim=0)

        action_means, action_stds = self.decoder(beliefs).chunk(2, -1)
        action_stds = self.min_std + F.softplus(action_stds)
        return action_means, action_stds
        

class AutoregressiveRNNPolicy(nn.Module):
    def __init__(
            self,
            obs_size,
            act_size, 
            belief_size=256,
            hidden_size=256,
            min_std=1e-4,
        ):
        
        super().__init__()
        self.obs_size = obs_size
        self.act_size = act_size
        self.belief_size = belief_size
        self.hidden_size = hidden_size
        self.min_std = min_std

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, belief_size)
        )
        
        # RNN
        self.rnn = nn.GRUCell(input_size=belief_size, hidden_size=belief_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * act_size),
        )

    def init_belief_and_action(self, batch_size, device):
        return (
            torch.zeros((batch_size, self.belief_size)).to(device),
            torch.zeros((batch_size, self.act_size)).to(device),
        )

    def forward(self, obs):
        # Input shape (T, B, dim)
        T, batch_size = obs.shape[:2]

        # Initialize belief
        init_belief, init_action = self.init_belief_and_action(batch_size, device=obs.device)

        beliefs = [torch.empty(0)] * (T + 1)
        actions = [torch.empty(0)] * (T + 1)
        action_means = [torch.empty(0)] * (T + 1)
        action_stds = [torch.empty(0)] * (T + 1)

        beliefs[0] = init_belief
        actions[0] = init_action

        # Forward through rnn
        for t in range(T):
            next_belief, next_action, action_mean, action_std = self.step(
                beliefs[t], actions[t], obs[t]
            )

            beliefs[t+1] = next_belief
            action_means[t+1] = action_mean
            action_stds[t+1] = action_std
            actions[t+1] = next_action

        return (
            torch.stack(actions[1:], dim=0),
            torch.stack(action_means[1:], dim=0),
            torch.stack(action_stds[1:], dim=0),   
        )
    
    def step(self, prev_belief, prev_action, obs):
        hidden = self.encoder(torch.cat((obs, prev_action), -1))
        belief = self.rnn(hidden, prev_belief)
        action_mean, action_std = self.decoder(belief).chunk(2, -1)
        action_std = self.min_std + F.softplus(action_std)
        action = action_mean + torch.randn_like(action_mean) * action_std
        return belief, action, action_mean, action_std
 
    def compute_loss(self, obs, actions, rewards):
        _, means, stds = self.forward(obs)
        dists = Independent(Normal(means, stds), 1)
        log_probs = dists.log_prob(actions)
        loss = -(log_probs * rewards.exp()).mean()
        return loss