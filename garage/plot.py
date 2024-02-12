import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from garage.utils.common import setup_plots
from garage.utils.fetch_demos import fetch_demos


def main(env_name: str, graph_all: bool) -> None:
    # Please uncomment the following line to download Palatino font
    # download_font()

    algorithm_to_color = {
        "expert": "green",
        "mm": "grey",
        "bc": "#1a1919",
        "bc_reg": "#966919",
        "filter": "#724BA1",
        "hype": "#F79646",
        "hyper": "#4BACC6",
        "hype_filter": "#f57c6e",
    }
    env_abbrv_to_full_and_ptremble = {
        "ant": ("Ant-v3", 0.01),
        "hopper": ("Hopper-v3", 0.01),
        "humanoid": ("Humanoid-v3", 0.025),
        "walker": ("Walker2d-v3", 0.01),
        "maze-diverse": ("antmaze-large-diverse-v2", 0),
        "maze-play": ("antmaze-large-play-v2", 0),
    }
    env_abbrv_to_step_size = {
        "ant": 10000,
        "hopper": 10000,
        "humanoid": 10000,
        "walker": 10000,
        "maze-diverse": 5000,
        "maze-play": 5000,
    }

    if graph_all:
        envs = ["ant", "hopper", "humanoid", "walker", "maze-diverse", "maze-play"]
    else:
        envs = [env_name]

    model_free_steps = 20
    mujoco_model_based_steps = 10
    d4rl_model_based_steps = 15

    experiment_results = Path("experiment_results")
    for env_name in envs:
        setup_plots()
        full_env_name, p_tremble = env_abbrv_to_full_and_ptremble[env_name]
        seeds_per_env = defaultdict(int)

        results_files = [f for f in experiment_results.glob(f"{full_env_name}/*.npz")]
        if env_name == "ant":
            results_files += [
                f for f in experiment_results.glob("ant_truncated_obs/*.npz")
            ]
        elif env_name == "humanoid":
            results_files += [
                f for f in experiment_results.glob("humanoid_truncated_obs/*.npz")
            ]
        algorithm_to_means = defaultdict(list)
        for result_file in results_files:
            algorithm_name = result_file.stem.rsplit("_", 1)[0]
            means = np.load(result_file, allow_pickle=True)["means"]
            if (
                algorithm_name == "hyper"
                and "maze" not in env_name
                and len(means) >= d4rl_model_based_steps
            ):
                algorithm_to_means[algorithm_name].append(
                    means[:d4rl_model_based_steps]
                )
            elif (
                algorithm_name == "hyper"
                and "maze" in env_name
                and len(means) >= mujoco_model_based_steps
            ):
                algorithm_to_means[algorithm_name].append(
                    means[:mujoco_model_based_steps]
                )
            elif len(means) >= model_free_steps:
                algorithm_to_means[algorithm_name].append(means[:model_free_steps])
            else:
                continue
            seeds_per_env[algorithm_name] += 1

        model_free_x = (
            np.arange(model_free_steps) * 5 * env_abbrv_to_step_size[env_name]
        )
        for algorithm_name, means in algorithm_to_means.items():
            algorithm_name_upper = algorithm_name.replace("_", "-").upper()
            plot_label = (
                f"{algorithm_name_upper} ({seeds_per_env[algorithm_name]} seeds)"
            )
            means = np.stack(means, axis=0)
            means, stderror = (
                np.mean(means, axis=0),
                np.std(means, axis=0) / np.sqrt(means.shape[0]),
            )
            if algorithm_name == "hyper":
                step_size = 10_000
                # add horizontal extensions if model-based
                initial_exploration_steps = 10000 if "maze" in env_name else 64000
                model_based_x = (
                    np.arange(len(means)) * step_size + initial_exploration_steps
                )
                plt.plot(
                    model_based_x,
                    means,
                    label=plot_label,
                    color=algorithm_to_color[algorithm_name],
                )
                plt.fill_between(
                    model_based_x,
                    means - stderror,
                    means + stderror,
                    alpha=0.1,
                    color=algorithm_to_color[algorithm_name],
                )

                # Extend the last value as a horizontal constant line
                plt.plot(
                    [model_based_x[-1], model_free_x[-1]],
                    [means[-1], means[-1]],
                    linestyle="--",
                    color=algorithm_to_color[algorithm_name],
                )
                plt.fill_between(
                    [model_based_x[-1], model_free_x[-1]],
                    means[-1] - stderror[-1],
                    means[-1] + stderror[-1],
                    alpha=0.1,
                    color=algorithm_to_color[algorithm_name],
                )

                # Add a horizontal line at y=0 from 0 to the first value in model_based_x
                plt.hlines(
                    0, 0, model_based_x[0], color=algorithm_to_color[algorithm_name]
                )
                plt.fill_between(
                    [0, model_based_x[0]],
                    0,
                    color=algorithm_to_color[algorithm_name],
                    alpha=0.1,
                )

                # Add a vertical line at x=exploration_steps from the first value in model_based_x to y=0
                plt.vlines(
                    initial_exploration_steps,
                    means[0],
                    0,
                    color=algorithm_to_color[algorithm_name],
                )
                plt.fill_between(
                    [model_based_x[0], initial_exploration_steps],
                    0,
                    color=algorithm_to_color[algorithm_name],
                    alpha=0.1,
                )
            else:
                step_size = 5 * env_abbrv_to_step_size[env_name]
                plt.plot(
                    np.arange(len(means)) * step_size,
                    means,
                    label=plot_label,
                    color=algorithm_to_color[algorithm_name],
                )
                plt.fill_between(
                    np.arange(len(means)) * step_size,
                    means - stderror,
                    means + stderror,
                    alpha=0.1,
                    color=algorithm_to_color[algorithm_name],
                )

        if "maze" not in env_name:
            # plot experts
            rewards = fetch_demos(full_env_name, 64_000, silent=True)["expert_rewards"]
            alg_name = "expert"
            alg_label = "$\pi_E$"
            rewards = np.stack(rewards, axis=0).reshape(-1, 1)
            means = np.mean(rewards)
            stderrors = np.std(rewards) / np.sqrt(rewards.shape[0])
            plt.plot(
                model_free_x,
                np.ones(model_free_steps) * means,
                label=alg_label,
                color=algorithm_to_color[alg_name],
                linestyle="--",
            )
            plt.fill_between(
                model_free_x,
                means + stderrors,
                means - stderrors,
                color=algorithm_to_color[alg_name],
                alpha=0.1,
            )

        plt.legend(ncol=2, fontsize=8, loc="lower right")
        plt.ylabel("Mean of $J(\\pi)$")
        plt.xlabel("Env. Steps")
        plt.title(full_env_name + ", $p_{tremble}=$" + str(p_tremble))

        fig_save_path = Path("figures", f"{env_name}_results.png")
        fig_save_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving figure to {fig_save_path}")
        plt.savefig(fig_save_path, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--env",
        type=str,
        choices=["ant", "hopper", "humanoid", "walker", "maze-diverse", "maze-play"],
        default="hopper",
    )
    argparser.add_argument(
        "--all", action="store_true", default=False, help="Plot all environments"
    )
    args = argparser.parse_args()
    main(args.env, args.all)
