import numpy as np


class ReplayBuffer:
    def __init__(
        self, capacity, obs_shape, act_shape, obs_type=np.float32, act_type=np.float32
    ):
        self.capacity = capacity
        self.observations = np.zeros((self.capacity,) + obs_shape, dtype=obs_type)
        self.actions = np.zeros((self.capacity,) + act_shape, dtype=act_type)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity,) + obs_shape, dtype=obs_type)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.capacity
        return self.pos

    def push(self, obs, act, rew, next_obs, done):
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(act).copy()
        self.rewards[self.pos] = np.array(rew).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.capacity:
            self.pos = 0
            self.full = True

    def push_batch(self, obs, act, rew, next_obs, done):
        batch_size = len(obs)
        if batch_size >= self.capacity:
            # Replace the entire buffer
            self.observations[:] = obs[-self.capacity :]
            self.actions[:] = act[-self.capacity :]
            self.rewards[:] = rew[-self.capacity :]
            self.next_observations[:] = next_obs[-self.capacity :]
            self.dones[:] = done[-self.capacity :]
            self.pos = 0
            self.full = True
        else:
            chunk = min(batch_size, self.capacity - self.pos)
            self.observations[self.pos : self.pos + chunk] = obs[:chunk]
            self.actions[self.pos : self.pos + chunk] = act[:chunk]
            self.rewards[self.pos : self.pos + chunk] = rew[:chunk]
            self.next_observations[self.pos : self.pos + chunk] = next_obs[:chunk]
            self.dones[self.pos : self.pos + chunk] = done[:chunk]
            if chunk < self.capacity - self.pos:
                # Not reaching the end of buffer, self.full does not change
                self.pos += chunk
            else:
                # Reaching the end of buffer, start from beginning
                rem = batch_size - chunk
                self.observations[:rem] = obs[chunk:]
                self.actions[:rem] = act[chunk:]
                self.rewards[:rem] = rew[chunk:]
                self.next_observations[:rem] = next_obs[chunk:]
                self.dones[:rem] = done[chunk:]
                self.pos = rem
                self.full = True

    def sample(self, batch_size, replace=True):
        batch_inds = np.random.choice(len(self), size=batch_size, replace=replace)
        return self._get_samples(batch_inds)

    def iterate(self, batch_size):
        random_inds = np.random.permutation(len(self))
        for i in range(0, len(self) - batch_size, batch_size):
            batch_inds = random_inds[i : i + batch_size]
            yield self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        obs = self.observations[batch_inds]
        act = self.actions[batch_inds]
        rew = self.rewards[batch_inds]
        next_obs = self.next_observations[batch_inds]
        done = self.dones[batch_inds]
        return obs, act, rew, next_obs, done

    def save(self, path):
        np.savez(path, **self.__dict__)

    def load(self, path):
        data = np.load(path)
        for key in self.__dict__.keys():
            setattr(self, key, data[key])


class SequenceReplayBuffer:
    def __init__(
        self, capacity, obs_shape, act_shape, obs_type=np.float32, act_type=np.float32
    ):
        self.capacity = capacity
        self.observations = np.zeros((self.capacity,) + obs_shape, dtype=obs_type)
        self.actions = np.zeros((self.capacity,) + act_shape, dtype=act_type)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.capacity
        return self.pos

    def push(self, obs, act, rew, done):
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(act).copy()
        self.rewards[self.pos] = np.array(rew).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.capacity:
            self.pos = 0
            self.full = True

    def sample(self, batch_size, seq_len):
        # Returns tuple of (seq_len, batch_size, feature_size)
        start_inds = np.random.choice(len(self) - seq_len, size=batch_size)
        batch_inds = np.stack([np.arange(s, s + seq_len) for s in start_inds], 0)
        batch_inds = batch_inds.transpose().reshape(-1)
        if self.full:
            # Prevent sampling across end of buffer
            batch_inds = (batch_inds + self.pos) % len(self)
        batch = self._get_samples(batch_inds)
        batch = [data.reshape(seq_len, batch_size, *data.shape[1:]) for data in batch]
        return tuple(batch)

    def iterate(self, batch_size, seq_len):
        all_start_inds = np.arange(0, len(self) - seq_len, seq_len)
        if self.full:
            all_start_inds = (all_start_inds + self.pos) % len(self)
        np.random.shuffle(all_start_inds)
        for i in range(0, len(all_start_inds) - batch_size, batch_size):
            start_inds = all_start_inds[i : i + batch_size]
            batch_inds = np.stack([np.arange(s, s + seq_len) for s in start_inds], 0)
            batch_inds = batch_inds.transpose().reshape(-1)
            if self.full:
                # Prevent sampling across end of buffer
                batch_inds = (batch_inds + self.pos) % len(self)
            batch = self._get_samples(batch_inds)
            batch = [
                data.reshape(seq_len, batch_size, *data.shape[1:]) for data in batch
            ]
            yield batch

    def _get_samples(self, batch_inds):
        obs = self.observations[batch_inds]
        act = self.actions[batch_inds]
        rew = self.rewards[batch_inds]
        done = self.dones[batch_inds]
        return obs, act, rew, done

    def save(self, path):
        np.savez(path, **self.__dict__)

    def load(self, path):
        with np.load(path) as buffer:
            for key in self.__dict__.keys():
                setattr(self, key, buffer[key])
        # Set last done to true
        if self.pos > 0 or self.full:
            self.dones[self.pos - 1] = 1


class RolloutBuffer:
    def __init__(
        self,
        capacity,
        num_envs,
        obs_shape,
        act_shape,
        obs_type=np.float32,
        act_type=np.float32,
    ):
        self.buffer_len = max(capacity // num_envs, 1)
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.obs_type = obs_type
        self.act_type = act_type
        self.reset()

    def reset(self):
        # Transitions
        self.observations = np.zeros(
            (self.buffer_len, self.num_envs) + self.obs_shape, dtype=self.obs_type
        )
        self.actions = np.zeros(
            (self.buffer_len, self.num_envs) + self.act_shape, dtype=self.act_type
        )
        self.rewards = np.zeros((self.buffer_len, self.num_envs, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_len, self.num_envs, 1), dtype=np.float32)

        # Additional information
        self.values = np.zeros((self.buffer_len, self.num_envs, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_len, self.num_envs, 1), dtype=np.float32)
        self.entropies = np.zeros((self.buffer_len, self.num_envs, 1), dtype=np.float32)

        # Computed at the end of rollout
        self.advantages = np.zeros(
            (self.buffer_len, self.num_envs, 1), dtype=np.float32
        )
        self.returns = np.zeros((self.buffer_len, self.num_envs, 1), dtype=np.float32)

        self.pos = 0
        self.full = False
        self.ready = False

    def push(self, obs, action, reward, done, value, log_prob, entropy):
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = np.array(value).copy()
        self.log_probs[self.pos] = np.array(log_prob).copy()
        self.entropies[self.pos] = np.array(entropy).copy()
        self.pos += 1
        if self.pos == self.buffer_len:
            self.full = True

    def compute_returns_and_advantages(
        self, last_values, gamma, gae_lambda, max_ent_coef=0
    ):
        # Add dense entropy to reward
        maxent_rewards = self.rewards + max_ent_coef * self.entropies
        for t in reversed(range(self.buffer_len)):
            if t == self.buffer_len - 1:
                next_values = last_values
                next_returns = last_values
                next_advs = 0
            else:
                next_values = self.values[t + 1]
                next_returns = self.returns[t + 1]
                next_advs = self.advantages[t + 1]
            next_nonterms = 1 - self.dones[t]
            # Compute generalized advantage function using maxent rewards
            delta = (
                maxent_rewards[t] + gamma * next_nonterms * next_values - self.values[t]
            )
            self.advantages[t] = delta + gamma * gae_lambda * next_nonterms * next_advs
            # Compute return using original rewards
            self.returns[t] = self.rewards[t] + gamma * next_nonterms * next_returns

    def swap_and_flatten(self, arr):
        return arr.swapaxes(0, 1).reshape(-1, *arr.shape[2:])

    def prepare_rollouts(self):
        assert self.full, "buffer is not full"
        assert not self.ready, "calling prepare_rollouts() multiple times"
        keys = [
            "observations",
            "actions",
            "log_probs",
            "advantages",
            "returns",
        ]
        for key in keys:
            self.__dict__[key] = self.swap_and_flatten(self.__dict__[key])
        self.ready = True

    def iterate(self, batch_size):
        if not self.ready:
            self.prepare_rollouts()

        num_samples = self.buffer_len * self.num_envs
        random_inds = np.random.permutation(num_samples)
        for i in range(0, num_samples - batch_size, batch_size):
            batch_inds = random_inds[i : i + batch_size]
            yield self._get_samples(batch_inds)

    def sample(self, batch_size, replace=True):
        if not self.ready:
            self.prepare_rollouts()

        num_samples = self.buffer_len * self.num_envs
        batch_inds = np.random.choice(num_samples, size=batch_size, replace=replace)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        obs = self.observations[batch_inds]
        act = self.actions[batch_inds]
        logp = self.log_probs[batch_inds]
        adv = self.advantages[batch_inds]
        ret = self.returns[batch_inds]
        return obs, act, logp, adv, ret


class EpisodicReplayBuffer:
    def __init__(
        self,
        capacity,
        episode_len,
        obs_shape,
        act_shape,
        obs_type=np.float32,
        act_type=np.float32,
    ):
        self.episode_len = episode_len
        self.num_episodes = capacity // episode_len

        self.observations = np.zeros(
            (self.num_episodes, self.episode_len) + obs_shape, dtype=obs_type
        )
        self.actions = np.zeros(
            (self.num_episodes, self.episode_len) + act_shape, dtype=act_type
        )
        self.rewards = np.zeros(
            (self.num_episodes, self.episode_len, 1), dtype=np.float32
        )
        self.dones = np.zeros(
            (self.num_episodes, self.episode_len, 1), dtype=np.float32
        )

        self.episode = 0
        self.pos = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.num_episodes * self.episode_len
        return self.episode * self.episode_len + self.pos

    def push(self, obs, act, rew, done):
        assert self.observations.dtype == obs.dtype, "data types must match"
        self.observations[self.episode, self.pos] = np.array(obs).copy()
        self.actions[self.episode, self.pos] = np.array(act).copy()
        self.rewards[self.episode, self.pos] = np.array(rew).copy()
        self.dones[self.episode, self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.episode_len:
            self.pos = 0
            self.episode += 1
            if self.episode == self.num_episodes:
                self.episode = 0
                self.full = True

    def sample(self, batch_size, seq_len):
        # Returns tuple of (seq_len, batch_size, feature_size)
        # Sample random episodes
        episode_inds = np.random.choice(
            (self.episode if not self.full else self.num_episodes) - 1,
            size=batch_size,
        )
        # Avoid sampling the current episode
        if self.full:
            episode_inds = (episode_inds + self.episode + 1) % self.num_episodes
        # Sample random timesteps
        start_inds = np.random.choice(self.episode_len - seq_len + 1, size=batch_size)
        timesteps = np.stack([np.arange(s, s + seq_len) for s in start_inds], 0)
        batch = self._get_samples(episode_inds, timesteps)
        batch = [np.swapaxes(data, 0, 1) for data in batch]
        return tuple(batch)

    def _get_samples(self, episode_inds, timesteps):
        episode_inds = episode_inds[:, None]
        obs = self.observations[episode_inds, timesteps]
        act = self.actions[episode_inds, timesteps]
        rew = self.rewards[episode_inds, timesteps]
        done = self.dones[episode_inds, timesteps]
        return obs, act, rew, done
    

class NORABuffer(EpisodicReplayBuffer):
    def __init__(
        self,
        capacity,
        episode_len,
        obs_shape,
        act_shape,
        rew_shape=1,
        obs_type=np.float32,
        act_type=np.float32,
    ):
        super().__init__(capacity, episode_len, obs_shape, act_shape, obs_type, act_type)
        self.rewards = np.zeros(
            (self.num_episodes, self.episode_len) + rew_shape, dtype=np.float32
        )


class MultimodalEpisodicReplayBuffer:
    def __init__(
        self,
        capacity,
        episode_len,
        image_shape,
        state_shape,
        act_shape,
        rew_shape,
    ):
        self.episode_len = episode_len
        self.num_episodes = capacity // episode_len

        self.images = np.zeros(
            (self.num_episodes, self.episode_len) + image_shape, dtype=np.uint8
        )
        self.states = np.zeros(
            (self.num_episodes, self.episode_len) + state_shape, dtype=np.float32
        )
        self.actions = np.zeros(
            (self.num_episodes, self.episode_len) + act_shape, dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.num_episodes, self.episode_len) + rew_shape, dtype=np.float32
        )
        self.dones = np.zeros(
            (self.num_episodes, self.episode_len, 1), dtype=np.float32
        )

        self.episode = 0
        self.pos = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.num_episodes * self.episode_len
        return self.episode * self.episode_len + self.pos

    def push(self, image, state, act, rew, done):
        self.images[self.episode, self.pos] = np.array(image).copy()
        self.states[self.episode, self.pos] = np.array(state).copy()
        self.actions[self.episode, self.pos] = np.array(act).copy()
        self.rewards[self.episode, self.pos] = np.array(rew).copy()
        self.dones[self.episode, self.pos] = np.array(done).copy()
        self.pos += 1
        if self.pos == self.episode_len:
            self.pos = 0
            self.episode += 1
            if self.episode == self.num_episodes:
                self.episode = 0
                self.full = True

    def sample(self, batch_size, seq_len, image=True):
        # Returns tuple of (seq_len, batch_size, feature_size)
        # Sample random episodes
        episode_inds = np.random.choice(
            (self.episode if not self.full else self.num_episodes) - 1,
            size=batch_size,
        )
        # Avoid sampling the current episode
        if self.full:
            episode_inds = (episode_inds + self.episode + 1) % self.num_episodes
        # Sample random timesteps
        start_inds = np.random.choice(self.episode_len - seq_len + 1, size=batch_size)
        timesteps = np.stack([np.arange(s, s + seq_len) for s in start_inds], 0)
        batch = self._get_samples(episode_inds, timesteps, image)
        batch = [np.swapaxes(data, 0, 1) for data in batch]
        return tuple(batch)

    def _get_samples(self, episode_inds, timesteps, image):
        episode_inds = episode_inds[:, None]
        state = self.states[episode_inds, timesteps]
        act = self.actions[episode_inds, timesteps]
        rew = self.rewards[episode_inds, timesteps]
        done = self.dones[episode_inds, timesteps]
        if image: 
            image = self.images[episode_inds, timesteps]
            return image, state, act, rew, done
        else:
            return state, act, rew, done