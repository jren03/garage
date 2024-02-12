"""
TD3-BC implementation adapted from https://github.com/sfujim/TD3_BC
and https://github.com/gkswamy98/fast_irl/blob/master/learners/TD3_BC.py.

Changes from https://github.com/gkswamy98/fast_irl/blob/master/learners/TD3_BC.py
are flagged with BEGIN CHANGES and END CHANGES comments.
"""


import copy
from typing import cast, Optional, Tuple

import gym
import numpy as np
import omegaconf
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import optim as optim

# import mbrl.types
import garage.mbrl.types as type_constants
from garage.utils.oadam import OAdam
from garage.utils.replay_buffer import QReplayBuffer
from garage.mbrl.util.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        expert_buffer: QReplayBuffer,
        learner_buffer: QReplayBuffer,
        discriminator: nn.Module,
        cfg: omegaconf.DictConfig,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise_scalar: float = 0.2,
        noise_clip_scalar: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        decay_lr: bool = False,
        hybrid_sampling: bool = False,
        device: str = "cuda",
    ):
        super(TD3_BC, self).__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = OAdam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = OAdam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise_scalar * max_action
        self.noise_clip = noise_clip_scalar * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.expert_buffer = expert_buffer
        self.learner_buffer = learner_buffer
        self.env = env
        self.discriminator = discriminator
        self.hybrid_sampling = hybrid_sampling
        self.device = device

        self.updates_made = 0

        self.schedule = False
        if decay_lr:
            self.schedule = True
            decay_lr_scheduler_cfg = cfg.overrides.decay_lr_scheduler
            if "cosine" in decay_lr_scheduler_cfg._target_.lower():
                self.critic_scheduler = hydra.utils.instantiate(
                    decay_lr_scheduler_cfg,
                    optimizer=self.critic_optimizer,
                    T_max=cfg.overrides.num_policy_updates_per_step
                    * decay_lr_scheduler_cfg.T_max,
                )
                self.actor_scheduler = hydra.utils.instantiate(
                    decay_lr_scheduler_cfg,
                    optimizer=self.critic_optimizer,
                    T_max=cfg.overrides.num_policy_updates_per_step
                    * decay_lr_scheduler_cfg.T_max,
                )
            else:
                raise NotImplementedError

    def reset(self) -> None:
        # for mbrl: do nothing
        return

    def act(self, obs: np.ndarray, batched: bool = False, **kwargs) -> np.ndarray:
        # wrapper to handle mbrl calls
        return self.predict(obs, batched=batched)[0]

    def predict(
        self,
        obs: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = True,
        batched: bool = False,
    ) -> Tuple[np.ndarray, None]:
        obs = torch.FloatTensor(obs).to(self.device)
        if batched:
            return self.actor(obs).cpu().data.numpy(), None
        else:
            return self.actor(obs.unsqueeze(0)).cpu().data.numpy().flatten(), None

    def learn(
        self, total_timesteps: int, log_interval: int = 1000, bc: bool = False
    ) -> None:
        if bc:
            for _ in tqdm.tqdm(
                range(total_timesteps), desc="BC initialization", ncols=0, leave=True
            ):
                self.step(bc=bc)
        else:
            obs = self.env.reset()
            done = False
            for _ in tqdm.tqdm(
                range(total_timesteps), ncols=0, leave=False, desc="TD3 Learn"
            ):
                act = self.predict(obs)[0]
                next_obs, rew, done, _ = self.env.step(act)
                self.learner_buffer.add(obs, act, next_obs, rew.cpu().detach(), done)
                self.step(bc=bc)
                obs = next_obs
                if done:
                    obs = self.env.reset()
                    done = False

    def split_mbrl_batch(
        self, batch: type_constants.TransitionBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, next_state, reward, not_done = list(
            map(
                lambda x: torch.FloatTensor(x).to(self.device),
                cast(type_constants.TransitionBatch, batch).astuple(),
            )
        )
        reward = reward.reshape(-1, 1)
        not_done = not_done.reshape(-1, 1)
        done = 1 - not_done
        return state, action, next_state, reward, done

    def step(self, batch_size: int = 256, bc: bool = False) -> None:
        self.updates_made += 1

        # -------------------------- BEGIN CHANGES --------------------------
        if not bc and self.hybrid_sampling:
            # 50/50 sample, loss function on all data
            learner_batch = self.learner_buffer.sample(batch_size // 2)
            if isinstance(self.learner_buffer, ReplayBuffer):
                state, action, next_state, reward, not_done = self.split_mbrl_batch(
                    learner_batch
                )
            else:
                state, action, next_state, reward, not_done = learner_batch
            if self.discriminator is not None:
                reward = -self.discriminator(torch.cat([state, action], dim=1)).reshape(
                    reward.shape
                )
            (
                exp_state,
                exp_action,
                exp_next_state,
                exp_reward,
                exp_not_done,
            ) = self.expert_buffer.sample(batch_size // 2)
            if self.discriminator is not None:
                exp_reward = -self.discriminator(
                    torch.cat([exp_state, exp_action], dim=1)
                ).reshape(exp_reward.shape)
            state = torch.cat([state, exp_state], dim=0)
            action = torch.cat([action, exp_action], dim=0)
            next_state = torch.cat([next_state, exp_next_state], dim=0)
            reward = torch.cat([reward, exp_reward], dim=0)
            not_done = torch.cat([not_done, exp_not_done], dim=0)
            pi_data = False
        # -------------------------- END CHANGES --------------------------
        else:
            state, action, next_state, reward, not_done = self.expert_buffer.sample(
                batch_size
            )
            if self.discriminator is not None:
                sa = torch.cat([state, action], dim=1)
                reward = -self.discriminator(sa).reshape(reward.shape)
            pi_data = False

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.schedule and not bc:
            self.critic_scheduler.step()

        # Delayed policy updates
        if self.updates_made % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() * (1 - bc) + F.mse_loss(pi, action) * (
                1 - pi_data
            )

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if self.schedule and not bc:
                self.actor_scheduler.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filename: str) -> None:
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename: str) -> None:
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
