import os
import random
from pathlib import Path
from typing import Tuple

import gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import wget
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

# ---------- Constants ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MF_LOG_FORMAT = [
    ("env_steps", "ES", "int"),
    ("mean_reward", "MR", "float"),
    ("std_reward", "SR", "float"),
]
MB_LOG_FORMAT = MF_LOG_FORMAT + [
    ("rollout_length", "RL", "int"),
]
AGENT_LOG_FORMAT = [
    ("agent/critic_loss", "c_loss", "float"),
    ("agent/actor_loss", "a_loss", "float"),
    ("agent/qf1_loss", "q1_loss", "float"),
    ("agent/qf2_loss", "q2_loss", "float"),
    ("agent/policy_loss", "p_loss", "float"),
    ("agent/alpha_loss", "al_loss", "float"),
    ("agent/critic_grad_norm", "c_grad", "float"),
    ("agent/actor_grad_norm", "a_grad", "float"),
    ("agent/policy_grad_norm", "p_grad", "float"),
    ("agent/q_values_mean", "q_mean", "float"),
    ("agent/q_values_std", "q_std", "float"),
    ("agent/target_q_mean", "tq_mean", "float"),
    ("agent/target_q_std", "tq_std", "float"),
    ("agent/reward_mean", "r_mean", "float"),
    ("agent/reward_std", "r_std", "float"),
    ("agent/log_pi_mean", "lp_mean", "float"),
    ("agent/log_pi_std", "lp_std", "float"),
    ("agent/alpha", "alpha", "float"),
    ("agent/lambda_value", "lambda", "float"),
    ("agent/action_mean", "act_mean", "float"),
    ("agent/action_std", "act_std", "float"),
    ("agent/buffer_size", "buf_size", "int"),
    ("agent/learning_rate_critic", "lr_crit", "float"),
    ("agent/learning_rate_actor", "lr_act", "float"),
]


ENV_ABBRV_TO_FULL = {
    "ant": "Ant-v3",
    "hopper": "Hopper-v3",
    "humanoid": "Humanoid-v3",
    "walker": "Walker2d-v3",
    "maze-diverse": "antmaze-large-diverse-v2",
    "maze-play": "antmaze-large-play-v2",
}

ENV_ABBRV_TO_DATASET_URL = {
    "ant": "https://www.dropbox.com/scl/fi/kvzl04g9d3pmhoowvfs75/Ant-v3_demos.npz?rlkey=g2ajzw0l3u3rt436g162e21sl&dl=1",
    "hopper": "https://www.dropbox.com/scl/fi/73ffc8msaky5vc741adr9/Hopper-v3_demos.npz?rlkey=77rm79dd1ppz6mebu0h2w3nvl&dl=1",
    "humanoid": "https://www.dropbox.com/scl/fi/obhdjlbc4pab4v93vflmj/Humanoid-v3_demos.npz?rlkey=tpkz81w8036m1vkzf2556kqsy&dl=1",
    "walker": "https://www.dropbox.com/scl/fi/3i4803t9mz76sqm7k92wy/Walker2d-v3_demos.npz?rlkey=81sb22opy037re044o26bflbh&dl=1",
}


# ---------- Common utility functions ----------
def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def rollout_agent_in_real_env(
    env: gym.Env, agent: nn.Module, num_trajs_to_sample: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Rollout trajectories using a policy and add to replay buffer

    Args:
        env: gym environment
        agent: policy to rollout
        num_trajs_to_sample: number of trajectories to sample

    Returns:
        S_curr: states from the rollouts
        A_curr: actions from the rollouts
        s: total number of steps taken
    """
    S_curr = []
    A_curr = []
    total_trajs = 0
    s = 0
    while total_trajs < num_trajs_to_sample:
        obs = env.reset()
        done = False
        while not done:
            S_curr.append(obs)
            act = agent.predict(obs)[0]
            A_curr.append(act)
            obs, _, done, _ = env.step(act)
            s += 1
            if done:
                total_trajs += 1
                break
    return (
        torch.from_numpy(np.array(S_curr)),
        torch.from_numpy(np.array(A_curr)),
        s,
    )


def setup_plots(use_special_font: bool = True) -> None:
    """
    Setup matplotlib plots in the style of the paper.
    If the font is not found, it will be downloaded.
    """
    if use_special_font:
        font_files = fm.findSystemFonts(".")
        if font_files == []:
            download_font()
        for font_file in fm.findSystemFonts("."):
            fm.fontManager.addfont(font_file)
        matplotlib.rc("font", family="Palatino Linotype")
    fig = plt.figure(dpi=100, figsize=(5.0, 3.0))
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.tick_params(direction="in")
    plt.ticklabel_format(style="sci", axis="x", scilimits=(6, 6), useMathText=True)


def download_font(
    font_url: str = r"https://github.com/dolbydu/font/raw/master/Serif/Palatino/Palatino%20Linotype.ttf",
) -> None:
    """Download the font used in the paper"""
    wget.download(font_url)
