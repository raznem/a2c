# Memory for A2C model
import numpy as np


class Memory:
    def __init__(self):
        self._obs = None
        self._next_obs_idx = None
        self._actions = None
        self._action_logprobs = None
        self._rewards = None
        self._done = None

    @property
    def obs(self):
        return self._obs[:-1]

    @property
    def next_obs(self):
        return self._obs[1:]

    @property
    def actions(self):
        return self._actions

    @property
    def action_logprobs(self):
        return self._action_logprobs

    @property
    def rewards(self):
        return self._rewards

    @property
    def done(self):
        return self._done

    @property
    def reward_per_rollout(self):
        return sum(self._rewards) / sum(self._done)

    def add_obs(self, obs):
        if self._obs is None:
            self._obs = []
            self._next_obs_idx = []
            self._actions = []
            self._action_logprobs = []
            self._rewards = []
            self._done = []

        self._obs.append(obs)
        obs_idx = len(self._obs) - 1
        return obs_idx

    def add_timestep(self, obs_idx, next_obs_idx, action, action_logprobs, reward, done):
        self._next_obs_idx.append(next_obs_idx)
        self._actions.append(action)
        self._action_logprobs.append(action_logprobs)
        self._rewards.append(reward)
        self._done.append(done)

    def __len__(self):
        if self._next_obs_idx is None:
            return 0
        else:
            return len(self._next_obs_idx)

    def __getitem__(self, obs_idx):
        next_obs_idx = self._next_obs_idx[obs_idx]
        return (
            np.array(self._obs[obs_idx]),
            np.array(self._obs[next_obs_idx]),
            np.array(self._actions[obs_idx]),
            np.array(self._action_logprobs[obs_idx]),
            np.array(self._rewards[obs_idx]),
            np.array(self._done[obs_idx])
        )

    def new_rollout(self):
        if self._obs is not None:
            self._obs.pop()


class ReplayBuffer(Memory):
    """ Usage: first add_obs to get obs_idx, then after env.step add the rest """
    def __init__(self, size):
        super().__init__()

        self.size = size
        self.current_idx = 0
        self.current_len = 0

    @property
    def obs(self):
        return self._obs[:self.current_len]

    @property
    def next_obs(self):
        return [self._obs[i] for i in self._next_obs_idx[:self.current_len]]

    @property
    def actions(self):
        return self._actions[:self.current_len]

    @property
    def action_logprobs(self):
        return self._action_logprobs[:self.current_len]

    @property
    def rewards(self):
        return self._rewards[:self.current_len]

    @property
    def done(self):
        return self._done[:self.current_len]

    def add_obs(self, obs):
        if self._obs is None:
            self._obs = np.empty([self.size] + list(obs.shape))
            self._obs_idx = np.empty(self.size, dtype=np.int)
            self._next_obs_idx = np.empty(self.size, dtype=np.int)
            self._actions = np.empty(self.size)
            self._rewards = np.empty(self.size)
            self._done = np.empty(self.size)

            self._action_logprobs = [None] * self.size # Will keep tensors with grad

        self._obs[self.current_idx] = obs
        self._obs_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.size

        self.current_len = min(self.size, self.current_len + 1)
        return self._obs_idx

    def add_timestep(self, obs_idx, next_obs_idx, action, action_logprobs, reward, done):
        self._next_obs_idx[obs_idx] = next_obs_idx
        self._actions[obs_idx] = action
        self._action_logprobs[obs_idx] = action_logprobs
        self._rewards[obs_idx] = reward
        self._done[obs_idx] = done

    def __len__(self):
        return self.current_len

    def new_rollout(self):
        self.current_idx -= 1
        self.current_len -= 1
