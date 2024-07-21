"""
This code is extended from https://github.com/vvanirudh/LAMPS-MBRL/blob/master/MujocoSysID/mbrl/algorithms/lamps.py
and follows largely the same structure. The main difference is the addition a
different reset function within the model and switching the SAC policy optimizer
to TD3-BC for antmaze experiments.
"""

import os
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import gym
import hydra
import numpy as np
import omegaconf
from pathlib import Path
import torch
from tqdm import tqdm

import garage.mbrl.planning as mbrl_planning
import garage.mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import garage.mbrl.types as type_constants
import garage.mbrl.util.common as mbrl_common
from garage.mbrl.models import ModelEnv, ModelTrainer
from garage.mbrl.util.env import EnvHandler
from garage.mbrl.planning.sac_wrapper import SACAgent
from garage.mbrl.util import ReplayBuffer
from garage.mbrl.util.math import truncated_linear
from garage.models.discriminator import Discriminator, DiscriminatorEnsemble
from garage.utils.common import PROJECT_ROOT, MB_LOG_FORMAT, rollout_agent_in_real_env
from garage.utils.ema_wrapper import EMA
from garage.utils.gym_wrappers import (
    GoalWrapper,
    RewardWrapper,
    TremblingHandWrapper,
)
from garage.utils.logger import Logger
from garage.utils.nn_utils import gradient_penalty
from garage.utils.oadam import OAdam
from garage.utils.replay_buffer import QReplayBuffer


def train(cfg: omegaconf.DictConfig, demos_dict: Dict[str, Any]) -> None:
    """
    Main training loop for model-based inverse reinforcement learning.

    Args:
        cfg (omegaconf.DictConfig): Configuration for the experiment.
        demos_dict (Dict[str, Any]): Dictionary containing the expert demonstrations.

    Returns:
        None
    """

    device = cfg.device
    env_name = cfg.overrides.env
    is_maze = "maze" in env_name

    # --------------- Wrap environment and init discriminator ---------------
    env, term_fn, reward_fn = EnvHandler.make_env(cfg)
    eval_env, *_ = EnvHandler.make_env(cfg)
    if is_maze:
        env = GoalWrapper(env, demos_dict["goals"][0][0])
        eval_env = GoalWrapper(eval_env, demos_dict["goals"][0][0])

    discriminator_cfg = cfg.overrides.discriminator
    if discriminator_cfg.ensemble_size > 1:
        f_net = DiscriminatorEnsemble(
            env,
            ensemble_size=discriminator_cfg.ensemble_size,
            clip_output=discriminator_cfg.clip_output,
        )
    else:
        f_net = Discriminator(env, clip_output=discriminator_cfg.clip_output)
    f_net.to(device)
    f_opt = OAdam(
        f_net.parameters(),
        lr=discriminator_cfg.lr,
        weight_decay=cfg.overrides.model_wd if discriminator_cfg.weight_decay else 0,
    )
    env = RewardWrapper(env, f_net)
    env = TremblingHandWrapper(env, cfg.overrides.p_tremble)
    eval_env = TremblingHandWrapper(eval_env, cfg.overrides.p_tremble)

    # --------------- Initialize Agent ---------------
    if is_maze:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        expert_buffer = QReplayBuffer(state_dim, action_dim)
        expert_buffer.add_d4rl_dataset(demos_dict["dataset"])
        learner_buffer = QReplayBuffer(state_dim, action_dim)
        agent = hydra.utils.instantiate(
            cfg.algorithm.td3_agent,
            env=env,
            expert_buffer=expert_buffer,
            learner_buffer=learner_buffer,
            discriminator=f_net,
            cfg=cfg,
        )
        agent.learn(total_timesteps=cfg.algorithm.bc_init_steps, bc=True)
    else:
        mbrl_planning.complete_agent_cfg(env, cfg.algorithm.agent)
        agent = SACAgent(
            cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
        )
        agent.sac_agent.add_discriminator(f_net)
        agent.sac_agent.reset_optimizers()
    if cfg.overrides.ema_agent:
        ema_agent = EMA(agent)

    # --------------- Setup Buffers ---------------
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    torch_generator.manual_seed(cfg.seed)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    hybrid_buffer = mbrl_common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )  # to store mixture of offline and online data
    policy_buffer = mbrl_common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )  # to store only online data
    expert_buffer = ReplayBuffer(
        capacity=cfg.overrides.expert_dataset_size,
        obs_shape=obs_shape,
        action_shape=act_shape,
        rng=rng,
    )  # to store only offline data
    expert_dataset = demos_dict["dataset"]
    hybrid_buffer.add_batch(
        expert_dataset["observations"][: cfg.overrides.initial_expert_steps],
        expert_dataset["actions"][: cfg.overrides.initial_expert_steps],
        expert_dataset["next_observations"][: cfg.overrides.initial_expert_steps],
        expert_dataset["rewards"][: cfg.overrides.initial_expert_steps],
        expert_dataset["terminals"][: cfg.overrides.initial_expert_steps],
    )
    expert_buffer.add_batch(
        expert_dataset["observations"][: cfg.overrides.expert_dataset_size],
        expert_dataset["actions"][: cfg.overrides.expert_dataset_size],
        expert_dataset["next_observations"][: cfg.overrides.expert_dataset_size],
        expert_dataset["rewards"][: cfg.overrides.expert_dataset_size],
        expert_dataset["terminals"][: cfg.overrides.expert_dataset_size],
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl_common.rollout_agent_trajectories(
        env,
        cfg.overrides.initial_exploration_steps,
        mbrl_planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=policy_buffer,
        additional_buffer=hybrid_buffer,
    )

    # --------------- Setup Model ---------------
    model_dir = Path(PROJECT_ROOT, "garage", "pretrained_models", env_name)
    dynamics_model = mbrl_common.create_one_dim_tr_model(
        cfg,
        obs_shape,
        act_shape,
        model_dir=model_dir
        if cfg.overrides.get("pretrained_dynamics_model", False)
        else None,
    )
    model_env = ModelEnv(
        env,
        dynamics_model,
        termination_fn=term_fn,
        reward_fn=None,
        generator=torch_generator,
    )
    model_trainer = ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        schedule=cfg.overrides.schedule_model_lr,
    )

    # --------------- Logging ---------------
    work_dir = os.getcwd()
    logger = Logger(work_dir, cfg)
    log_name = f"{cfg.algorithm.name}_{env_name}"
    logger.register_group(log_name, MB_LOG_FORMAT, color="cyan")
    save_path = Path(
        PROJECT_ROOT,
        "garage",
        "experiment_results",
        env_name,
        f"{cfg.algorithm.name}_{cfg.seed}.npz",
    )
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # ----------------- Train -----------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    total_env_steps = cfg.algorithm.total_env_steps
    expert_sa_pairs = demos_dict["expert_sa_pairs"].to(device)
    expert_reset_states = demos_dict["expert_reset_states"]

    learner_buffer = None  # to store rollouts in learned model
    epoch, env_steps, disc_steps = 0, 0, 0
    mean_rewards, std_rewards = [], []

    # The model is updated after `cfg.overrides.freq_train_model` steps, and
    # the discriminator is updated after `discriminator_cfg.train_every` steps
    # by the policy in the learned model.
    tbar = tqdm(range(total_env_steps), ncols=0)
    while env_steps < total_env_steps:
        rollout_length = int(
            truncated_linear(*(cfg.overrides.rollout_schedule + [epoch + 1]))
        )
        learner_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        learner_buffer_capacity *= cfg.overrides.num_epochs_to_retain_learner_buffer
        learner_buffer = maybe_replace_learner_buffer(
            learner_buffer, obs_shape, act_shape, learner_buffer_capacity, cfg.seed
        )
        if is_maze:
            agent.learner_buffer = learner_buffer
        obs, done = None, False

        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False

            # --- Doing env step and adding to buffers ---
            next_obs, _, done, _ = mbrl_common.step_env_and_add_to_buffer(
                env, obs, agent, {}, hybrid_buffer, policy_buffer
            )
            (
                exp_obs,
                exp_next_obs,
                exp_act,
                exp_reward,
                exp_done,
            ) = expert_buffer.sample_one()
            hybrid_buffer.add(exp_obs, exp_act, exp_next_obs, exp_reward, exp_done)

            # -------------- Model Training --------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                # Reset to states from hybrid_buffer to train the model
                # For details, see first bullet point here:
                # https://github.com/jren03/garage/tree/main/garage/algorithms#model-based-inverse-reinforcement-learning
                mbrl_common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    hybrid_buffer,
                    work_dir=work_dir,
                )

                # --- Rollout new model and store imagined trajectories ---
                # For details, see the second bullet point here:
                # https://github.com/jren03/garage/tree/main/garage/algorithms#model-based-inverse-reinforcement-learning
                if is_maze:
                    # Reset to sliding window backwards in time for antmaze
                    # Reset to [-i/n * T, (i/n + 0.05) * T]
                    row_indices = np.random.choice(
                        len(expert_reset_states),
                        size=rollout_batch_size,
                        replace=True,
                    )
                    percent_remaining = 1 - env_steps / cfg.overrides.total_env_steps
                    low = int(percent_remaining * cfg.overrides.epoch_length)
                    high = min(
                        cfg.overrides.epoch_length,
                        int((percent_remaining + 0.1) * cfg.overrides.epoch_length),
                    )
                    column_indices = np.random.randint(low, high, size=len(row_indices))
                    indices = np.stack((row_indices, column_indices), axis=-1)
                    reset_states = expert_reset_states[indices[:, 0], indices[:, 1]]
                else:
                    reset_to_exp_states = rng.random() < cfg.overrides.sac_reset_ratio
                    reset_buffer = (
                        expert_buffer if reset_to_exp_states else policy_buffer
                    )
                    batch = reset_buffer.sample(rollout_batch_size)
                    reset_states, *_ = cast(
                        type_constants.TransitionBatch, batch
                    ).astuple()
                rollout_model_and_populate_learner_buffer(
                    model_env,
                    reset_states,
                    agent=agent,
                    learner_buffer=learner_buffer,
                    sac_samples_action=cfg.algorithm.sac_samples_action,
                    rollout_horizon=rollout_length,
                )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_policy_updates_per_step):
                if is_maze:
                    agent.step(bc=False)
                elif (
                    env_steps + 1
                ) % cfg.overrides.policy_updates_every_steps != 0 or len(
                    learner_buffer
                ) < cfg.overrides.sac_batch_size:
                    break
                else:
                    agent.sac_agent.adv_update_parameters(
                        learner_buffer,
                        cfg.overrides.sac_batch_size,
                        reverse_mask=True,
                    )
                if cfg.overrides.ema_agent:
                    ema_agent.update()
                if not is_maze and cfg.overrides.schedule_actor_lr:
                    agent.sac_agent.step_lr()

            # --------------- Discriminator Training ---------------
            if is_maze:
                updates_made = agent.updates_made
            else:
                updates_made = agent.sac_agent.updates_made
            if updates_made > 0 and updates_made % discriminator_cfg.train_every == 0:
                if disc_steps == 0:
                    disc_lr = discriminator_cfg.lr
                else:
                    disc_lr = discriminator_cfg.lr / disc_steps
                f_opt = OAdam(
                    f_net.parameters(),
                    lr=disc_lr,
                    weight_decay=cfg.overrides.model_wd
                    if discriminator_cfg.weight_decay
                    else 0,
                )

                curr_states, curr_actions, _ = rollout_agent_in_real_env(
                    env, agent, discriminator_cfg.num_sample_trajectories
                )
                learner_sa_pairs = torch.cat((curr_states, curr_actions), dim=1).to(
                    device
                )
                for _ in range(discriminator_cfg.num_update_steps):
                    learner_sa = learner_sa_pairs[
                        np.random.choice(
                            len(learner_sa_pairs), discriminator_cfg.batch_size
                        )
                    ]
                    expert_sa = expert_sa_pairs[
                        np.random.choice(
                            len(expert_sa_pairs), discriminator_cfg.batch_size
                        )
                    ]
                    f_opt.zero_grad()
                    f_learner = f_net(learner_sa.float())
                    f_expert = f_net(expert_sa.float())
                    gp = gradient_penalty(learner_sa, expert_sa, f_net)
                    loss = f_expert.mean() - f_learner.mean() + 10 * gp
                    loss.backward()
                    f_opt.step()
                disc_steps += 1
                if not is_maze and cfg.overrides.schedule_actor_lr:
                    agent.sac_agent.reset_optimizers()

            # --------------- Logging ---------------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                epoch += 1
            if (env_steps + 1) % cfg.overrides.eval_frequency == 0:
                if is_maze:
                    mean_reward, std_reward = evaluate_in_real_env(
                        eval_env, agent, n_eval_episodes=25
                    )
                    mean_reward = mean_reward * 100
                    std_reward = std_reward * 100
                else:
                    mean_reward, std_reward = evaluate_in_real_env(
                        eval_env, agent, n_eval_episodes=10
                    )
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                logger.log_data(
                    log_name,
                    {
                        "env_steps": env_steps,
                        "mean_reward": mean_reward,
                        "std_reward": std_reward,
                    },
                )
                np.savez(
                    str(save_path),
                    means=mean_rewards,
                    stds=std_rewards,
                )

            # --------------- Update ---------------
            obs = next_obs
            env_steps += 1
            tbar.update(1)

    # ------------- Save results -------------
    print(f"Results saved to {save_path}")


def evaluate_in_real_env(
    env: gym.Env,
    agent: SACAgent,
    n_eval_episodes: int,
) -> Tuple[float, float]:
    """
    Evaluate the agent in the real environment.

    Args:
        env (gym.Env): The environment to evaluate the agent in.
        agent (SACAgent): The agent to evaluate.
        n_eval_episodes (int): The number of episodes to evaluate the agent over.

    Returns:
        Tuple[float, float]: The mean and standard deviation of the rewards obtained
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if torch.is_tensor(reward):
                reward = reward.cpu().detach().item()
            episode_reward += reward
        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def maybe_replace_learner_buffer(
    learner_buffer: Optional[ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> ReplayBuffer:
    """
    Replace the learner buffer if it is None or if the max capacity has changed.

    Args:
        learner_buffer (Optional[ReplayBuffer]): The current learner buffer.
        obs_shape (Sequence[int]): The shape of the observations.
        act_shape (Sequence[int]): The shape of the actions.
        new_capacity (int): The new capacity of the buffer.
        seed (int): The seed for the random number generator.

    Returns:
        ReplayBuffer: The new learner buffer.
    """
    if learner_buffer is None or new_capacity != learner_buffer.capacity:
        if learner_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = learner_buffer.rng
        new_buffer = ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if learner_buffer is None:
            return new_buffer
        obs, action, next_obs, reward, done = learner_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, done)
        return new_buffer
    return learner_buffer


def rollout_model_and_populate_learner_buffer(
    model_env: ModelEnv,
    initial_obs: np.ndarray,
    agent: SACAgent,
    learner_buffer: ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
) -> None:
    """
    Rollout the model and populate the learner buffer with the trajectories from learned model.

    Args:
        model_env (ModelEnv): The model environment to rollout the model in.
        initial_obs (np.ndarray): The initial observations to rollout from.
        agent (SACAgent): The agent to use to rollout the model.
        learner_buffer (ReplayBuffer): The buffer to populate with the trajectories from learned model.
        sac_samples_action (bool): Whether the SAC agent samples actions or not.
        rollout_horizon (int): The length of the rollout.

    Returns:
        None
    """
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        learner_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()
