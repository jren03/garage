from pathlib import Path
import gym
import numpy as np
import torch

from garage.utils.common import PROJECT_ROOT

EPS = 1e-6


def fetch_demos(
    env_name: str, expert_dataset_size: int = -1, silent: bool = False
) -> tuple:
    """
    Fetches the expert demonstrations for the given environment.

    Args:
        env_name (str): Name of the environment.
        expert_dataset_size (int): Number of expert demonstrations to use.
        silent (bool): Whether to print the dataset details.

    Returns:
        tuple: A tuple containing the following elements:
            - dataset (dict): A dictionary containing the expert demonstrations.
            - expert_rewards (np.ndarray): Array containing the rewards obtained by the expert.
            - expert_sa_pairs (torch.Tensor): Tensor containing the state-action pairs of the expert.
            - qpos (np.ndarray): Array containing the qpos of the expert.
            - qvel (np.ndarray): Array containing the qvel of the expert.
            - goals (np.ndarray): Array containing the goals of the expert.
            - expert_reset_states (np.ndarray): Array containing the reset states of the expert.
    """
    if "truncated" in env_name.lower():
        env_name = f"{env_name.split('_')[0].capitalize()}-v3"
        is_truncated = True
    else:
        is_truncated = False

    possible_data_path = Path(
        PROJECT_ROOT, "experts", env_name, f"{env_name}_demos.npz"
    )
    assert possible_data_path.exists(), f"{possible_data_path} does not exist, please run `python experts/download_datasets.py --env <env_name>` to download demos, or `python experts/collect_demos.py --env <env_name>` to collect demos."

    if "maze" in env_name:
        data = np.load(possible_data_path, allow_pickle=True)
        new_dataset = {
            "observations": data["observations"],
            "actions": data["actions"],
            "next_observations": data["next_observations"],
            "rewards": data["rewards"],
            "terminals": data["terminals"],
        }
        new_dataset["rewards"] = np.zeros_like(data["rewards"])
        expert_sa_pairs = torch.from_numpy(data["expert_sa_pairs"])
        expert_reset_states = data["expert_reset_states"]
        qpos = data["qpos"]
        qvel = data["qvel"]
        goals = data["goals"]
        Js = np.zeros_like(qpos)  # no expert rewards
    else:
        dataset = np.load(possible_data_path, allow_pickle=True)
        expert_dataset_size = (
            expert_dataset_size if expert_dataset_size > 0 else len(dataset["rewards"])
        )
        new_dataset = {
            key: np.array(dataset[key])[:expert_dataset_size] for key in dataset.keys()
        }
        if is_truncated and (
            "ant" in env_name.lower() or "humanoid" in env_name.lower()
        ):
            # get qpos and qvel dimensions
            print(f"Old dataset shape: {new_dataset['observations'].shape}")
            env = gym.make(env_name)
            qpos, qvel = (
                env.sim.data.qpos.ravel().copy(),
                env.sim.data.qvel.ravel().copy(),
            )
            qpos_dim, qvel_dim = qpos.shape[0], qvel.shape[0]
            obs_dim = (
                qpos_dim + qvel_dim - 2
            )  # truncated obs ignores first 2 elements of qpos
            new_dataset["observations"] = new_dataset["observations"][:, :obs_dim]
            new_dataset["next_observations"] = new_dataset["next_observations"][
                :, :obs_dim
            ]
            print(f"New dataset shape: {new_dataset['observations'].shape}")

        term = np.argwhere(
            np.logical_or(new_dataset["timeouts"] > 0, new_dataset["terminals"] > 0)
        )
        Js = []
        ranges = []
        start = 0
        for i in range(len(term)):
            ranges.append((start, term[i][0] + 1))
            J = new_dataset["rewards"][start : term[i][0] + 1].sum()
            Js.append(J)
            start = term[i][0] + 1
        Js = np.array(Js)
        exp_ranges = np.array(ranges)

        # Record sequence of obs and actions for each seed to achieve deterministic resets
        traj_obs = np.array(
            [
                new_dataset["observations"][exp_range[0] : exp_range[1]]
                for exp_range in exp_ranges
            ],
        )
        traj_actions = np.array(
            [
                new_dataset["actions"][exp_range[0] : exp_range[1]]
                for exp_range in exp_ranges
            ],
        )
        traj_seeds = np.array(
            [
                new_dataset["seed"][exp_range[0] : exp_range[1]]
                for exp_range in exp_ranges
            ],
        )
        for key in [
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "terminals",
        ]:
            new_dataset[key] = np.concatenate(
                [
                    new_dataset[key][exp_range[0] : exp_range[1]]
                    for exp_range in exp_ranges
                ]
            )
            if key == "actions":
                new_dataset[key] = np.clip(
                    new_dataset[key], -1 + EPS, 1 - EPS
                )  # due to tanh in TD3
        new_dataset["rewards"] = np.zeros_like(
            new_dataset["rewards"]
        )  # zero out rewards

        qpos = np.array(
            [
                new_dataset["qpos"][exp_range[0] : exp_range[1]]
                for exp_range in exp_ranges
            ],
        )
        qvel = np.array(
            [
                new_dataset["qvel"][exp_range[0] : exp_range[1]]
                for exp_range in exp_ranges
            ],
        )
        goals = None
        expert_obs = torch.tensor(new_dataset["observations"])
        expert_acts = torch.tensor(new_dataset["actions"])
        expert_sa_pairs = torch.cat((expert_obs, expert_acts), dim=1)
        expert_reset_states = None

        if not silent:
            print("-" * 80)
            print(f"{possible_data_path=}")
            print(f"{expert_obs.min()=}, {expert_obs.max()=}")
            print(
                f"{expert_sa_pairs.shape=}\t{expert_obs.shape=}\t{expert_acts.shape=}"
            )
            print(f"{qpos.shape=}\t{qvel.shape=}\t{np.mean(Js)=}\t{np.std(Js)=}")
            print(f"{traj_obs.shape=}\t{traj_actions.shape=}\t{traj_seeds.shape=}")
            print("-" * 80)

    return {
        "dataset": new_dataset,
        "expert_rewards": Js,
        "expert_sa_pairs": expert_sa_pairs,
        "qpos": qpos,
        "qvel": qvel,
        "goals": goals,
        "traj_obs": traj_obs,
        "traj_actions": traj_actions,
        "traj_seeds": traj_seeds,
        "expert_reset_states": expert_reset_states,
    }
