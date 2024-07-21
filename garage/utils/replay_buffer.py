from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import tqdm
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


class QReplayBuffer(object):
    """
    Adapted from https://github.com/gkswamy98/fast_irl/blob/master/learners/buffer.py
    """

    def __init__(
        self, state_dim: int, action_dim: int, max_size: int = int(1e6)
    ) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _convert_if_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if torch.is_tensor(x):
            return x.cpu().detach().numpy()
        return x

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        next_state: Union[np.ndarray, torch.Tensor],
        reward: Union[np.ndarray, torch.Tensor],
        done: Union[np.ndarray, torch.Tensor],
    ) -> None:
        self.state[self.ptr] = self._convert_if_tensor(state)
        self.action[self.ptr] = self._convert_if_tensor(action)
        self.next_state[self.ptr] = self._convert_if_tensor(next_state)
        self.reward[self.ptr] = self._convert_if_tensor(reward)
        self.not_done[self.ptr] = 1.0 - self._convert_if_tensor(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

    def add_d4rl_dataset(self, dataset: dict) -> None:
        self.state = dataset["observations"][: self.max_size]
        self.action = dataset["actions"][: self.max_size]
        self.next_state = dataset["next_observations"][: self.max_size]
        self.reward = dataset["rewards"][: self.max_size].reshape(-1, 1)
        self.not_done = 1.0 - dataset["terminals"][: self.max_size].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL(self, dataset, q_dataset, samps=int(1e6)):
        j = 0
        m = []
        for i in tqdm.tqdm(range(len(q_dataset["observations"]))):
            while (
                np.linalg.norm(
                    q_dataset["observations"][i] - dataset["observations"][j]
                )
                > 1e-10
            ):
                j += 1
            m.append(j)
        m = np.array(m)
        goals = dataset["infos/goal"][m]

        j = 0
        m = []
        for i in tqdm.tqdm(range(len(q_dataset["next_observations"]))):
            while (
                np.linalg.norm(
                    q_dataset["next_observations"][i] - dataset["observations"][j]
                )
                > 1e-10
            ):
                j += 1
            m.append(j)
        m = np.array(m)
        next_goals = dataset["infos/goal"][m]

        self.state = np.concatenate([q_dataset["observations"], goals], axis=1)[:samps]
        self.action = q_dataset["actions"][:samps]
        self.next_state = np.concatenate(
            [q_dataset["next_observations"], next_goals], axis=1
        )[:samps]
        self.reward = q_dataset["rewards"].reshape(-1, 1)[:samps]
        self.not_done = 1.0 - q_dataset["terminals"].reshape(-1, 1)[:samps]
        self.size = self.state.shape[0]


class HybridReplayBuffer(ReplayBuffer):
    """
    Subclass of SB3 ReplayBuffer that allows sampling from
    both the learner and expert data.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cuda",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        expert_data: dict = dict(),
        hybrid_sampling: bool = False,
        fixed_hybrid_schedule: bool = False,
        sampling_schedule: List[List[float]] = None,
    ) -> None:
        super(HybridReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.expert_states = expert_data["observations"]
        self.expert_actions = expert_data["actions"]
        self.expert_next_states = expert_data["next_observations"]
        self.expert_dones = expert_data["terminals"]
        self.expert_rewards = expert_data["rewards"]
        self.expert_timeouts = expert_data["timeouts"]

        self.hybrid_sampling = hybrid_sampling
        self.fixed_hybrid_schedule = fixed_hybrid_schedule
        if sampling_schedule:
            self.sampling_schedule = np.array(sampling_schedule)
        self.steps = 0
        self.ratio_lag = 0

    def _get_ratio(self, t: int) -> float:
        if self.fixed_hybrid_schedule:
            return self.sampling_schedule[0, 0]
        if t > self.sampling_schedule[0, 2] and self.sampling_schedule.shape[0] > 1:
            self.sampling_schedule = np.delete(self.sampling_schedule, 0, 0)
            self.ratio_lag = t
        max_lr, min_lr, lr_steps = self.sampling_schedule[0]
        ratio = max_lr - min(1, (t - self.ratio_lag) / (lr_steps - self.ratio_lag)) * (
            max_lr - min_lr
        )
        self.steps += 1
        return ratio

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
        sample_expert_only=False,
    ) -> ReplayBufferSamples:
        num_samples = len(batch_inds)
        if self.hybrid_sampling:
            # Sample both learner and expert samples
            # For details, see third bullet point here:
            # https://github.com/jren03/garage/tree/main/garage/algorithms#model-free-inverse-reinforcement-learning
            offline_ratio = self._get_ratio(self.steps)
            num_expert_samples = int(num_samples * offline_ratio)
            num_learner_samples = int(num_samples * (1 - offline_ratio))
            learner_inds = batch_inds[:num_learner_samples]
            expert_inds = np.random.randint(
                0, len(self.expert_states), size=num_expert_samples, dtype=int
            )

            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(
                    self.observations[(learner_inds + 1) % self.buffer_size, 0, :],
                    env,
                )
            else:
                next_obs = self._normalize_obs(
                    self.next_observations[learner_inds, 0, :], env
                )

            obs = self._normalize_obs(self.observations[learner_inds, 0, :], env)
            actions = self.actions[learner_inds, 0, :]
            dones = self.dones[learner_inds]
            rewards = self.rewards[learner_inds]

            if expert_inds.shape[0] > 0:
                next_obs = np.concatenate(
                    (
                        next_obs,
                        self._normalize_obs(self.expert_next_states[expert_inds], env),
                    ),
                    axis=0,
                )
                obs = np.concatenate(
                    (obs, self._normalize_obs(self.expert_states[expert_inds], env)),
                    axis=0,
                )
                actions = np.concatenate(
                    (
                        actions,
                        self.expert_actions[expert_inds].reshape(
                            num_expert_samples, -1
                        ),
                    ),
                    axis=0,
                )
                dones = np.concatenate(
                    (
                        dones,
                        self.expert_dones[expert_inds].reshape(num_expert_samples, -1),
                    ),
                    axis=0,
                )
                rewards = np.concatenate(
                    (
                        rewards,
                        self.expert_rewards[expert_inds].reshape(
                            num_expert_samples, -1
                        ),
                    ),
                    axis=0,
                )
            data = (
                obs,
                actions,
                next_obs,
                (dones).reshape(-1, 1),
                self._normalize_reward(rewards.reshape(-1, 1), env),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

        elif sample_expert_only:
            # Sample only from expert buffer
            env_indices = np.random.randint(
                0, high=self.n_envs, size=(len(batch_inds),)
            )
            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(
                    self.expert_states[
                        (batch_inds + 1) % self.buffer_size, env_indices, :
                    ],
                    env,
                )
            else:
                next_obs = self._normalize_obs(
                    self.expert_next_states[batch_inds, env_indices, :], env
                )
            data = (
                self._normalize_obs(
                    self.expert_states[batch_inds, env_indices, :], env
                ),
                self.expert_actions[batch_inds, env_indices, :],
                next_obs,
                (self.expert_dones[batch_inds, env_indices]).reshape(-1, 1),
                self._normalize_reward(
                    self.expert_rewards[batch_inds, env_indices].reshape(-1, 1), env
                ),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

        else:
            # Sample only from learner buffer
            env_indices = np.random.randint(
                0, high=self.n_envs, size=(len(batch_inds),)
            )
            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(
                    self.observations[
                        (batch_inds + 1) % self.buffer_size, env_indices, :
                    ],
                    env,
                )
            else:
                next_obs = self._normalize_obs(
                    self.next_observations[batch_inds, env_indices, :], env
                )

            data = (
                self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
                self.actions[batch_inds, env_indices, :],
                next_obs,
                (self.dones[batch_inds, env_indices]).reshape(-1, 1),
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                ),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sample_expert_only(self, batch_size, env):
        """
        Sample only a batch from the expert dataset. Can be used
        downstream in BC regularization.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env, sample_expert_only=True)

    def normalize_expert_obs(self, obs: np.ndarray) -> np.ndarray:
        def compute_mean_std(states: np.ndarray, eps: float):
            mean = states.mean(0)
            std = states.std(0) + eps
            return mean, std

        def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
            return (states - mean) / std

        state_mean, state_std = compute_mean_std(obs, eps=1e-3)
        return normalize_states(obs, state_mean, state_std)
