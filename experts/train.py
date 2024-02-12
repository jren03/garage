import argparse
from pathlib import Path
from typing import Callable, Union

import gym
from tqdm.auto import tqdm


from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from garage.utils.common import PROJECT_ROOT


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train expert policies.")
    parser.add_argument(
        "--env",
        choices=["Ant-v3", "Hopper-v3", "Humanoid-v3", "Walker2d-v3"],
        required=True,
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2_000_000,
    )
    args = parser.parse_args()

    env_name = args.env
    env = gym.make(env_name)

    print(f"Training {env_name} expert for {args.train_steps} steps.")
    if env_name in ["Ant-v3", "Humanoid-v3", "Hopper-v3"]:
        # hyperparams from rl-baselines3-zoo: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            buffer_size=300000,
            batch_size=256,
            gamma=0.98,
            tau=0.02,
            train_freq=64,
            gradient_steps=64,
            ent_coef="auto",
            learning_rate=linear_schedule(7.3e-4),
            learning_starts=10000,
            policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
            use_sde=True,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
        )
    with ProgressBarManager(args.train_steps) as callback:
        model.learn(args.train_steps, callback=callback)

    mean_rewards, std_rewards = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"Expert performance: {mean_rewards=:.4f}, {std_rewards=:.4f}")

    save_path = str(Path(PROJECT_ROOT, "experts", args.env, "expert_spec"))
    model.save(save_path)
    print(f"Saved expert to {save_path}")
