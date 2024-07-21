"""
Subclass of SB3 SAC agent (https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/sac.py)

This reimplementation allows for relabelling and dual-buffer sampling. Changes are marked with "BEGIN CHANGES" and "END CHANGES".
"""

import torch as th
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class SAC(SB3_SAC):
    def __init__(
        self,
        discriminator: th.nn.Module,
        bc_reg: bool = False,
        bc_weight: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super(SAC, self).__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.relabel_rewards = self.discriminator is not None
        self.bc_reg = bc_reg
        self.bc_weight = bc_weight

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # ! Train mode is only supported in later versions of stable-baselines3
        # Switch to train mode (this affects batch norm / dropout)
        # self.policy.set_training_mode(True)
        # self.policy.train()

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term

                # ----------- BEGIN CHANGES ------------
                # Relabel with current reward function.
                # See description here under bullet point 1 here:
                # https://github.com/jren03/garage/tree/main/garage/algorithms#model-free-inverse-reinforcement-learning
                if self.relabel_rewards:
                    # Get the negative of reward from the discriminator (since we use cost)
                    rewards = -self.discriminator(
                        th.cat(
                            (
                                replay_data.observations.to(th.float),
                                replay_data.actions.to(th.float),
                            ),
                            axis=1,
                        )
                    ).reshape(replay_data.rewards.shape)
                    target_q_values = (
                        rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    )
                else:
                    target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * next_q_values
                    )
                # ----------- END CHANGES ------------

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            # ----------- BEGIN CHANGES ------------
            if self.bc_reg:
                expert_data = self.replay_buffer.sample_expert_only(
                    batch_size, env=self._vec_normalize_env
                )
                actions_bc, _ = self.actor.action_log_prob(expert_data.observations)
                actor_loss = (
                    ent_coef * log_prob - min_qf_pi
                ).mean() + self.bc_weight * F.mse_loss(actions_bc, expert_data.actions)
            else:
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            # ----------- END CHANGES ------------
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                # Copy running stats, see GH issue #996
                # polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
