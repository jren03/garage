import os
import random
from typing import Tuple

import gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import wget
from pathlib import Path
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

ENV_ABBRV_TO_FULL = {
    "ant": "Ant-v3",
    "hopper": "Hopper-v3",
    "humanoid": "Humanoid-v3",
    "walker": "Walker2d-v3",
    "maze-diverse": "antmaze-large-diverse-v2",
    "maze-play": "antmaze-large-play-v2",
}

ENV_ABBRV_TO_DATASET_URL = {
    # "ant": "https://drive.google.com/uc?export=download&id=14yZEuRFceeqJ7vLAHkWf6dcg-hRkXWGI",
    # "hopper": "https://drive.google.com/uc?export=download&id=1eLhUBFp6LSrWZc2Ijlz-rIXZcQaDNNYl",
    # "humanoid": "https://drive.google.com/uc?export=download&id=1NeQ56i4g1uh7W0AOSS0rZ3psp48Kn-q5",
    # "walker": "https://drive.google.com/uc?export=download&id=11LuWGVRmgJnPOXIwRekeRN9tHf9i2dYZ",
    "ant": "https://www.dropbox.com/scl/fi/nak6w23k6hinmjd0u7npe/Ant-v3_demos.npz?rlkey=bxc9xtxnzjxfmg3ji26ntppsi&dl=1",
    "hopper": "https://www.dropbox.com/scl/fi/tp9x66ivkjp2n45bcj323/Hopper-v3_demos.npz?rlkey=ta5qtl1yr2xi8su78q3rf54ma&dl=1",
    "humanoid": "https://www.dropbox.com/scl/fi/14vcy8bel4v2hqfrxxr3c/Humanoid-v3_demos.npz?rlkey=rkwks556ty7n0lf4mmjfct9eq&dl=1",
    "walker": "https://www.dropbox.com/scl/fi/s95wk2i87aqi4vjd4b9ae/Walker2d-v3_demos.npz?rlkey=rrvxasd9sgawshv81obi912ih&dl=1",
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
