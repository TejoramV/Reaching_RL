import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn import functional as F

class RNNEstimator(nn.Module):
    def __init__(
            self,
            obs_size,
            act_size, 
            zeta_size,
            belief_size=256,
            hidden_size=256,
            min_std=1e-4,
            use_act=False,
        ):
        
        super().__init__()
        self.obs_size = obs_size
        self.act_size = act_size
        self.zeta_size = zeta_size
        self.belief_size = belief_size
        self.hidden_size = hidden_size
        self.min_std = min_std

        self.use_act = use_act

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_size + act_size if self.use_act else obs_size, hidden_size),
            # not used in hidden 32 exps
            nn.ReLU(),
            nn.Linear(hidden_size, belief_size)
        )
        
        # RNN
        self.rnn = nn.GRUCell(input_size=belief_size, hidden_size=belief_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * zeta_size),
        )

    def init_belief(self, batch_size, device):
        return torch.zeros((batch_size, self.belief_size)).to(device)

    def forward(self, obs, act):

        # # # # only use sphere state
        # obs = obs[...,3:6]
        # # # only use EE and sphere state
        # obs = obs[...,:6]
        # # # only use sphere velo
        # obs = obs[...,6:9]
        # # only use sphere state and velo
        # obs = obs[...,3:9]
        
        # delta sphere pos + vel / obs (t_{0} to t_{T-1} ) - (t_{1} to t_{T})
        delta_pos = obs[:-1,:,3:5] - obs[1:,:,3:5]
        delta_vel = obs[:-1,:,6:8] - obs[1:,:,6:8]
        obs = torch.cat((delta_pos, delta_vel), -1)
        act = act[:-1]

        # Input shape (T, B, dim)
        T, batch_size = obs.shape[:2]

        # Initialize belief
        belief = self.init_belief(batch_size, device=obs.device)

        # Encode observations
        if self.use_act:
            embeds = self.encoder(torch.cat((obs, act), -1))
        else:
            embeds = self.encoder(obs)
        # Forward through RNN
        for t in range(T):
            belief = self.rnn(embeds[t], belief)

        # Decode predictions
        means, stds = self.decoder(belief).chunk(2, -1)
        stds = self.min_std + F.softplus(stds)
        return means, stds
 
    def compute_loss(self, obs, actions, zetas):
        means, stds = self.forward(obs)
        dists = Independent(Normal(means, stds), 1)
        loss = -dists.log_prob(zetas).mean()
        return loss

    def compute_rewards(self, obs, actions, zetas):
        means, stds = self.forward(obs)
        dists = Independent(Normal(means, stds), 1)
        rewards = dists.log_prob(zetas)
        return rewards

class MLPEstimator(nn.Module):
    def __init__(
        self, 
        obs_size, 
        act_size, 
        zeta_size,
        seq_len, 
        hidden_size, 
        log_std_min=-20.,
        log_std_max=2.,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.zeta_size = zeta_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        input_size = (obs_size + act_size) * seq_len
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * zeta_size),
        )
    
    def forward(self, obs, act):
        T, B, _ = obs.shape
        out = self.net(torch.cat((obs, act), -1).transpose(0, 1).flatten(1))
        mean, log_std = out.chunk(2, -1)
        log_std = self.log_std_max - F.softplus(self.log_std_max - log_std)
        log_std = self.log_std_min + F.softplus(log_std - self.log_std_min)
        return mean, log_std.exp()