import glob
import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from common.buffers import RolloutBuffer
from common.utils import get_device, to_torch, to_np, preprocess
from models import MLPPolicyNetwork, MLPValueNetwork, CNN

from common.logger import Video

from tqdm import tqdm, trange

class PPO:
    def __init__(self, config, envs, eval_envs, logger, device):
        self.c = config
        self.envs = envs
        self.eval_envs = eval_envs
        self.logger = logger
        self.device = device

        self.step = 0
        self.epoch = 0
        self.obs = envs.reset()
        self.setup_models()

    def setup_models(self):
        obs_shape = self.envs.observation_space.shape
        act_shape = self.envs.action_space.shape

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            self.c.rollout_length,
            self.envs.num_envs,
            obs_shape,
            act_shape,
            obs_type=np.uint8 if self.c.pixel_obs else np.float32,
        )

        # Encoder
        if self.c.pixel_obs:
            self.encoder = CNN(self.c.input_dim, obs_shape[-1], self.c.repr_size).to(self.device)
            self.encoder_optim = Adam(self.encoder.parameters(), lr=self.c.encoder_lr)
            obs_shape = (self.c.repr_size,)

        # Policy
        self.policy = MLPPolicyNetwork(obs_shape, act_shape, self.c.mlp_hidden_size).to(
            self.device
        )
        self.policy_optim = Adam(self.policy.parameters(), lr=self.c.policy_lr)

        # Value function
        self.value_function = MLPValueNetwork(obs_shape, self.c.mlp_hidden_size).to(
            self.device
        )
        self.value_function_optim = Adam(
            self.value_function.parameters(), lr=self.c.value_lr
        )

    def collect_rollouts(self, num_steps):
        self.policy.eval()
        self.value_function.eval()

        # Clear rollout buffer
        self.rollout_buffer.reset()

        steps_per_env = num_steps // self.envs.num_envs
        for _ in trange(steps_per_env, desc='collecting rollouts'):
            # Select action
            with torch.no_grad():
                obs_tensor = to_torch(preprocess(self.obs))
                if self.c.pixel_obs:
                    obs_tensor = self.encoder(obs_tensor)
                actions, log_probs, entropies = map(to_np, self.policy(obs_tensor))
                values = to_np(self.value_function(obs_tensor))

            # Take environment step
            next_obs, rewards, dones, infos = self.envs.step(actions)
            rewards = rewards[:, None]
            dones = dones[:, None]

            # Handle termination and truncation
            for i, done in enumerate(dones):
                if done:
                    # Record episode statistics
                    self.logger.record("train/return", infos[i]["episode_return"])
                    self.logger.record("train/success", infos[i]["episode_success"])
                    # Handle truncation by bootstraping from value function
                    if infos[i].get("TimeLimit.truncated", False):
                        term_obs = to_torch(preprocess(infos[i]["terminal_obs"][None]))
                        with torch.no_grad():
                            if self.c.pixel_obs:
                                term_obs = self.encoder(term_obs)
                            term_value = to_np(self.value_function(term_obs))
                        rewards[i] += self.c.gamma * term_value[0]

            # Add transition to buffer
            self.rollout_buffer.push(
                self.obs, actions, rewards, dones, values, log_probs, entropies
            )
            self.obs = next_obs
            self.step += self.envs.num_envs

        # Compute returns and advantages
        with torch.no_grad():
            next_obs_tensor = to_torch(preprocess(next_obs))
            if self.c.pixel_obs:
                next_obs_tensor = self.encoder(next_obs_tensor)
            last_values = to_np(self.value_function(next_obs_tensor))
        self.rollout_buffer.compute_returns_and_advantages(
            last_values, self.c.gamma, self.c.gae_lambda, self.c.max_ent_coef
        )

    def compute_policy_loss(self, obs, actions, old_ll, advantages):
        new_ll, ent, mean, stddev = self.policy.evaluate(obs, actions)
        # Compute surrogate objective
        lr = (new_ll - old_ll).exp()
        surrogate = lr * advantages
        # Compute clipped surrotate objective
        lr_clip = torch.clamp(lr, min=1 - self.c.clip_range, max=1 + self.c.clip_range)
        surrogate_clip = lr_clip * advantages
        # Take minimum of the two objectives
        objective = torch.min(surrogate, surrogate_clip)
        # Add entropy regularization
        objective += self.c.ent_reg_coef * ent
        return -objective.mean(), surrogate, surrogate_clip, ent, mean, stddev

    def compute_value_loss(self, obs, returns):
        value_preds = self.value_function(obs)
        return F.mse_loss(value_preds, returns)

    def update_parameters(self):
        self.policy.train()
        self.value_function.train()

        if not self.rollout_buffer.ready:
            self.rollout_buffer.prepare_rollouts()

        for _ in trange(self.c.train_epochs, desc="update"):
            for batch in self.rollout_buffer.iterate(self.c.batch_size):
                obs = to_torch(preprocess(batch[0]))
                actions, log_probs, advantages, returns = map(to_torch, batch[1:])

                self.policy_optim.zero_grad()
                self.value_function_optim.zero_grad()
                if self.c.pixel_obs:
                    self.encoder_optim.zero_grad()
                    obs = self.encoder(obs)

                policy_loss, surrogate, surrogate_clip, ent, mean, stddev = self.compute_policy_loss(
                    obs, actions, log_probs, advantages
                )
                value_loss = self.compute_value_loss(obs, returns)
                total_loss = policy_loss + value_loss
                total_loss.backward()

                if self.c.clip_grad_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.c.clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), self.c.clip_grad_norm)

                grad_norm_policy = 0.
                for p in self.policy.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm_policy += param_norm ** 2
                grad_norm_policy = grad_norm_policy ** 0.5

                grad_norm_value = 0.
                for p in self.value_function.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm_value += param_norm ** 2
                grad_norm_value = grad_norm_value ** 0.5

                self.policy_optim.step()
                self.value_function_optim.step()
                if self.c.pixel_obs:
                    self.encoder_optim.step()

                self.logger.record("train/policy_loss", policy_loss.item())
                self.logger.record("train/policy_grad_norm", grad_norm_policy.item())

                self.logger.record("train/surrogate", surrogate.mean().item())
                self.logger.record("train/surrogate_clip", surrogate_clip.mean().item())
                self.logger.record("train/entropy", ent.mean().item())

                self.logger.record("train/mean", mean.mean().item())
                self.logger.record("train/stddev", stddev.mean().item())

                self.logger.record("train/value_loss", value_loss.item())
                self.logger.record("train/value_grad_norm", grad_norm_value.item())


    def train(self):
        self.load_checkpoint()

        while self.step < self.c.num_steps:
            self.collect_rollouts(self.c.rollout_length)
            self.update_parameters()
            self.epoch += 1

            if self.epoch % self.c.eval_freq == 0:
                self.evaluate()

            if self.epoch % self.c.checkpoint_freq == 0:
                self.save_checkpoint()

            self.logger.record("train/step", self.step)
            self.logger.dump(step=self.step)

    def evaluate(self):
        self.policy.eval()
        self.value_function.eval()

        obs = self.eval_envs.reset()
        dones, infos, frames = False, [], []
        # Assume all eval_envs terminate at the same time
        while not np.all(dones):
            # Select action
            with torch.no_grad():
                obs_tensor = to_torch(preprocess(obs))
                if self.c.pixel_obs:
                    obs_tensor = self.encoder(obs_tensor)
                actions = to_np(self.policy(obs_tensor)[0])

            # Take environment step
            next_obs, _, dones, infos = self.eval_envs.step(actions)
            # if self.c.pixel_obs:
            #     frames.append(obs)
            frames.append(self.eval_envs.render()[...,:3])
            obs = next_obs
        
        # Record episode statistics
        avg_return = np.mean([info["episode_return"] for info in infos])
        avg_success = np.mean([info["episode_success"] for info in infos])
        self.logger.record(f"eval/return", avg_return)
        self.logger.record(f"eval/success", avg_success)

        # Record videos
        # if self.c.pixel_obs:
        frames = np.stack(frames).swapaxes(0, 1)
        for i in range(self.eval_envs.num_envs):
            stack = frames[i]
            for j in range(stack.shape[-1] // 3):
                video = stack[:, :, :, j * 3 : (j + 1) * 3]
                video = video.transpose(0, 3, 1, 2)[None]
                self.logger.record(
                    f"eval/trajectory/env-{i}/camera-{j}",
                    Video(video, fps=30),
                    exclude=["stdout"],
                )

    def save_checkpoint(self):
        ckpt_path = os.path.join(self.logger.dir, "models_%d.pt" % self.epoch)
        ckpt = {
            "step": self.step,
            "epoch": self.epoch,
            "policy": self.policy.state_dict(),
            "policy_optim": self.policy_optim.state_dict(),
            "value_function": self.value_function.state_dict(),
            "value_function_optim": self.value_function_optim.state_dict(),
        }
        if self.c.pixel_obs:
            ckpt["encoder"] = self.encoder.state_dict()
            ckpt["encoder_optim"] = self.encoder_optim.state_dict()
        torch.save(ckpt, ckpt_path)

    def load_checkpoint(self, ckpt_path=None):
        # Load models from the latest checkpoint
        if ckpt_path:
            ckpt_paths = list(glob.glob(os.path.join(ckpt_path, "models_*.pt")))
        else:
            ckpt_paths = list(glob.glob(os.path.join(self.logger.dir, "models_*.pt")))
        if len(ckpt_paths) > 0:
            max_epoch = 0
            for path in ckpt_paths:
                epoch = path[path.rfind("/") + 8 : -3]
                if epoch.isdigit() and int(epoch) > max_epoch:
                    max_epoch = int(epoch)
            ckpt_path = os.path.join(self.logger.dir, f"models_{max_epoch}.pt")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            print(f"Loaded checkpoint from {ckpt_path}")

            self.step = ckpt["step"]
            self.epoch = ckpt["epoch"]
            self.policy.load_state_dict(ckpt["policy"])
            self.policy_optim.load_state_dict(ckpt["policy_optim"])
            self.value_function.load_state_dict(ckpt["value_function"])
            self.value_function_optim.load_state_dict(ckpt["value_function_optim"])
            if self.c.pixel_obs:
                self.encoder.load_state_dict(ckpt["encoder"])
                self.encoder_optim.load_state_dict(ckpt["encoder_optim"])
