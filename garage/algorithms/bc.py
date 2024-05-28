from pathlib import Path
from typing import Any, Dict

import gym
import hydra
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from garage.models.sac import SAC
from garage.utils.common import PROJECT_ROOT
from garage.utils.gym_wrappers import (
    GoalWrapper,
    TremblingHandWrapper,
)
from garage.utils.nn_utils import linear_schedule
from garage.utils.replay_buffer import QReplayBuffer


class BCDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = obs
        self.acts = acts

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


def train(cfg: omegaconf.DictConfig, demos_dict: Dict[str, Any]) -> None:
    """
    Main training loop for behavioral cloning

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
    env = gym.make(cfg.overrides.env)
    eval_env = gym.make(cfg.overrides.env)
    if is_maze:
        env = GoalWrapper(eval_env, demos_dict["goals"][0][0])
        eval_env = GoalWrapper(eval_env, demos_dict["goals"][0][0])
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
            discriminator=None,
            cfg=cfg,
        )
    else:
        sac_agent_cfg = cfg.algorithm.sac_agent
        agent = SAC(
            env=env,
            discriminator=None,
            learning_rate=linear_schedule(7.3e-4),
            bc_reg=sac_agent_cfg.bc_reg,
            bc_weight=sac_agent_cfg.bc_weight,
            policy=sac_agent_cfg.policy,
            verbose=sac_agent_cfg.verbose,
            ent_coef=sac_agent_cfg.ent_coef,
            train_freq=sac_agent_cfg.train_freq,
            gradient_steps=sac_agent_cfg.gradient_steps,
            gamma=sac_agent_cfg.gamma,
            tau=sac_agent_cfg.tau,
            device=device,
        )
        pi = agent.policy.actor
        expert_obs = torch.from_numpy(
            demos_dict["dataset"]["observations"][: cfg.overrides.expert_dataset_size]
        ).to(device)
        expert_acts = torch.from_numpy(
            demos_dict["dataset"]["actions"][: cfg.overrides.expert_dataset_size]
        ).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(pi.parameters(), lr=cfg.algorithm.lr)
        bc_dataset = BCDataset(expert_obs, expert_acts)
        bc_dataloader = DataLoader(
            bc_dataset, batch_size=cfg.algorithm.batch_size, shuffle=True
        )

    # --------------- Logging ---------------
    save_path = Path(
        PROJECT_ROOT,
        "garage",
        "experiment_results",
        env_name,
        f"{cfg.algorithm.name}_{cfg.seed}.npz",
    )
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # ----------------- Train -----------------
    total_train_steps = cfg.algorithm.total_train_steps
    if is_maze:
        agent.learn(total_timesteps=total_train_steps, bc=True)
    else:
        total_it = 0
        tbar = tqdm(total=total_train_steps, ncols=0)
        while total_it < total_train_steps:
            for batch in bc_dataloader:
                states, actions = batch
                optimizer.zero_grad()
                outputs = pi(states)
                loss = loss_fn(outputs, actions)
                loss.backward()
                optimizer.step()
                total_it += 1
                tbar.update(1)
        tbar.close()

    if is_maze:
        mean_reward, std_reward = evaluate_policy(
            agent, eval_env, n_eval_episodes=cfg.algorithm.total_eval_trajs
        )
        mean_reward = mean_reward * 100
        std_reward = std_reward * 100
    else:
        mean_reward, std_reward = evaluate_policy(
            agent, eval_env, n_eval_episodes=cfg.algorithm.total_eval_trajs
        )
    print(f"{mean_reward=}, {std_reward=}")
    np.savez(
        str(save_path),
        means=mean_reward,
        stds=std_reward,
        p_tremble=cfg.overrides.p_tremble,
    )

    # ------------- Save results -------------
    print(f"Results saved to {save_path}")
