from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .utils import init_weights


class CNNSmall(nn.Module):
    def __init__(self, input_chn, output_dim, act="ReLU", output_act="Identity"):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        self.layers = nn.Sequential(
            nn.Conv2d(input_chn, 32, 8, stride=4),
            act_fn(),
            nn.Conv2d(32, 64, 4, stride=2),
            act_fn(),
            nn.Conv2d(64, 64, 3, stride=1),
            act_fn(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim),
            output_act_fn(),
        )
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)
    

class CNN(nn.Module):
    def __init__(self, input_chn, output_dim, act="ReLU", output_act="Identity"):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        self.layers = nn.Sequential(
            nn.Conv2d(input_chn, 32, 3, stride=2),
            act_fn(), 
            nn.Conv2d(32, 32, 3, stride=1),
            act_fn(), 
            nn.Conv2d(32, 32, 3, stride=1),
            act_fn(),
            nn.Conv2d(32, 32, 3, stride=1),
            act_fn(),
            nn.Flatten(),
            nn.Linear(32 * 35 * 35, output_dim),
            output_act_fn(),
        )
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)


class MultiheadedCNN(CNN):
    def __init__(
        self, input_chn, num_heads, output_dim, act="ReLU", output_act="Identity"
    ):
        super().__init__(input_chn, output_dim * num_heads, act, output_act)
        self.output_dim = output_dim
        self.num_heads = num_heads

    def forward(self, x, select_inds=None):
        out = super().forward(x)
        out = out.view(-1, self.num_heads, self.output_dim)
        if select_inds is not None:
            out = out[torch.arange(out.shape[0]), select_inds]
        return out


class TransposeCNN(nn.Module):
    def __init__(self, input_dim, output_chn, act="ReLU", output_act="Identity"):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64 * 4 * 4),
            act_fn(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 64, 3, stride=1),
            act_fn(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(32, output_chn, 8, stride=4),
            output_act_fn(),
        )
        self.apply(partial(init_weights, act=act))

    def forward(self, x):
        return self.layers(x)


class BC2NN(nn.Module):
    def __init__(
        self, 
        img_shape, 
        proprio_shape, 
        action_shape, 
        hidden_size_rnn=256, 
        act="ReLU", 
        output_act="Identity",
        min_std=0.1,
        pretrain=False
    ):
        super().__init__()
        act_fn = getattr(nn, act)
        output_act_fn = getattr(nn, output_act)
        self.min_std = min_std
        
        self.pretrain = pretrain
        if self.pretrain:
            assert img_shape[1] == 224 and img_shape[2] == 224, "r3m only supports 224x224 images"
            from r3m import load_r3m
            r3m = load_r3m("resnet18")
            r3m.eval()
            self.cnn = r3m.to("cuda")

        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(img_shape[0], 32, kernel_size=8, stride=4),
                act_fn(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                act_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                act_fn(),
                nn.Flatten(),
            )

        with torch.no_grad():
            if self.pretrain:
                cnn_out_dim = self.cnn(torch.zeros(1, 3, img_shape[1], img_shape[2])).shape[-1]
                cnn_out_dim *= img_shape[0] // 3
            else:
                cnn_out_dim = self.cnn(torch.zeros(1, *img_shape)).shape[-1]

        img_embed_dim = 192
        self.linear_img = nn.Sequential(
            nn.Linear(cnn_out_dim, img_embed_dim),
            act_fn(),
        )

        proprio_embed_dim = 64
        self.linear_proprio = nn.Sequential(
            nn.Linear(proprio_shape[0], proprio_embed_dim),
            act_fn(),
        )

        self.rnn_input_dim = img_embed_dim + proprio_embed_dim
        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim, 
            hidden_size=hidden_size_rnn,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size_rnn, 2 * action_shape[0]),
            output_act_fn(),
        )
        
        self.apply(partial(init_weights, act=act))

    def init_belief(self, batch_size, device):
        return torch.zeros((1, batch_size, self.rnn.hidden_size), device=device)

    def step(self, prev_belief, obs, proprio):
        # prev_belief: (b, hidden_size_rnn)
        # obs: (b, img_dims)

        # preprocess obs
        # feat = self.cnn(obs)
        if self.pretrain:
            b, c, h, w = obs.shape
            ratio = c // 3
            feat = obs.reshape(b*ratio, 3, h, w)
        else:
            feat = obs

        # for layer in self.cnn:
        #     feat = layer(feat)
        feat = self.cnn(feat)

        if self.pretrain:
            # feat = feat.reshape(b, ratio, -1)
            feat = feat.reshape(b, -1)

        feat = self.linear_img(feat)

        # preprocess proprio
        proprio = self.linear_proprio(proprio)

        # predict
        cat = torch.cat((feat, proprio), dim=-1)
        out, belief = self.rnn(cat.unsqueeze(0), prev_belief)
        out = self.mlp(out.squeeze(0))

        mean, std = out.chunk(2, dim=-1)
        std = self.min_std + F.softplus(std)
        action = mean + torch.randn_like(mean) * std
        return belief, action
    
    def forward(self, obs, proprio):
        # X: (t, b, c, h, w)
        # proprio: (t, b, proprio_dims)
        # recurrent_cell: (b, hidden_size_rnn)

        T, B = obs.shape[:2]

        # initialize recurrent cell
        belief = self.init_belief(B, device=obs.device)

        if self.pretrain:
            T, B, c, h, w = obs.shape
            ratio = c // 3
            feat = obs.reshape(T*B*ratio, 3, h, w)
        else:
            feat = obs.reshape(T*B, *obs.shape[2:])

        # preprocess imgs
        feat = self.cnn(feat)

        # if self.pretrain:
        #     # feat = feat.reshape(b, ratio, -1)
        #     feat = feat.reshape(T, B, -1)
        # else:
        feat = feat.reshape(T, B, -1)
        feat = self.linear_img(feat)
        
        # preprocess proprioception
        proprio = self.linear_proprio(proprio)
        proprio = proprio.reshape(T, B, -1)

        # predict actions
        cat = torch.cat((feat, proprio), dim=-1)
        out, _ = self.rnn(cat, belief)
        out = self.mlp(out)

        mean, std = out.chunk(2, dim=-1)
        std = self.min_std + F.softplus(std)
        return mean, std