from math import exp
import os

import gym
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Any, Optional, Tuple, Union

from mbrl.third_party.pytorch_sac_pranz24.model import (
    DeterministicPolicy,
    GaussianPolicy,
    QNetwork,
)
from mbrl.third_party.pytorch_sac_pranz24.utils import hard_update, soft_update

from garage.utils.oadam import OAdam
from garage.utils.nn_utils import linear_schedule
from garage.utils.replay_buffer import ReplayBuffer


class SAC(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        relabel_samples: bool,
        action_space: gym.Space,
        args: omegaconf.DictConfig,
    ):
        super(SAC, self).__init__()
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None or args.target_entropy == -1:
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.discriminator = None
        self.updates_made = 0
        self.relabel_samples = relabel_samples
        self.decay_horizon = args.decay_horizon

    def add_discriminator(self, discriminator: nn.Module) -> None:
        self.discriminator = discriminator

    def reset_optimizers(self, optim_oadam: bool = True) -> None:
        if optim_oadam:
            self.critic_optim = OAdam(self.critic.parameters(), lr=self.args.lr)
            self.policy_optim = OAdam(self.policy.parameters(), lr=self.args.lr)
        else:
            self.critic_optim = Adam(self.critic.parameters(), lr=self.args.lr)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.args.lr)
        self.get_schedule_fn = linear_schedule(self.args.lr)
        self.updates_made = 0

    def step_lr(self) -> None:
        optimizers = [self.critic_optim, self.policy_optim]
        for optim in optimizers:
            for param_group in optim.param_groups:
                progress_remaining = max(
                    1 - self.updates_made / self.decay_horizon, 1e-8
                )
                param_group["lr"] = self.get_schedule_fn(progress_remaining)

    def select_action(
        self, state: np.ndarray, batched: bool = False, evaluate: bool = False
    ) -> np.ndarray:
        state = torch.FloatTensor(state)
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def estimate_value(self, state: np.ndarray) -> torch.Tensor:
        _, _, action = self.policy.sample(state)
        q1, q2 = self.critic(state, action)

        return torch.min(q1, q2)

    def update_parameters(
        self,
        memory: ReplayBuffer,
        batch_size: int,
        reverse_mask: bool = False,
    ) -> None:
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        reward_batch = self._relabel_with_discriminator(
            state_batch,
            action_batch,
            torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1),
        )
        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        self.updates_made += 1
        if self.updates_made % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    def adv_update_parameters(
        self,
        memory: ReplayBuffer,
        batch_size: int,
        expert_memory: Optional[ReplayBuffer] = None,
        reverse_mask: bool = False,
    ) -> None:
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = memory.sample(batch_size).astuple()

        if expert_memory:
            (
                expert_state_batch,
                expert_action_batch,
                *_,
            ) = expert_memory.sample(batch_size).astuple()
        else:
            expert_state_batch, expert_action_batch, *_ = memory.sample(
                batch_size
            ).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        reward_batch = self._relabel_with_discriminator(
            state_batch,
            action_batch,
            torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1),
        )

        expert_state_batch = torch.FloatTensor(expert_state_batch).to(self.device)
        expert_action_batch = torch.FloatTensor(expert_action_batch).to(self.device)

        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            try:
                next_state_action, next_state_log_pi, _ = self.policy.sample(
                    next_state_batch
                )
            except:
                breakpoint()
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(expert_state_batch)

        qf1_pi, qf2_pi = self.critic(expert_state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        with torch.no_grad():
            qf1_expert, qf2_expert = self.critic_target(
                expert_state_batch, expert_action_batch
            )
            min_qf_expert = torch.min(qf1_expert, qf2_expert)

        policy_loss = (
            (self.alpha * log_pi) + (min_qf_expert - min_qf_pi)
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        self.updates_made += 1
        if self.updates_made % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    # relabel rewards with discriminator
    # For details, see third bullet point here:
    # https://github.com/jren03/garage/tree/main/garage/algorithms#model-based-inverse-reinforcement-learning
    @torch.no_grad()
    def _relabel_with_discriminator(
        self,
        state_batch: Tuple[Any],
        action_batch: Tuple[Any],
        reward_batch: Tuple[Any],
    ) -> Tuple[Any]:
        # relabel rewards with discriminator
        if self.discriminator is not None and self.relabel_samples:
            sa_pair = torch.cat((state_batch, action_batch), dim=1)
            reward_batch = -self.discriminator(sa_pair).reshape(reward_batch.shape)
        return reward_batch

    # Save model parameters
    def save_checkpoint(
        self,
        env_name: Optional[str] = None,
        suffix: str = "",
        ckpt_path: Optional[str] = None,
    ) -> None:
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, ckpt_path: str, evaluate: bool = False):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cuda:0")
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
