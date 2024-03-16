import argparse
from pathlib import Path
from typing import Callable, Union

import gym
from tqdm.auto import tqdm


from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from garage.utils.common import PROJECT_ROOT, ENV_ABBRV_TO_FULL


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
        choices=["ant", "hopper", "humanoid", "walker"],
        required=True,
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    env_name = ENV_ABBRV_TO_FULL[args.env]
    train_env = gym.make(env_name)
    eval_env = gym.make(env_name)

    save_dir = Path(PROJECT_ROOT, "experts", env_name)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(Path(save_dir, "logs")),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # hyperparms from rl-trained-agents: https://github.com/DLR-RM/rl-trained-agents/blob/master/sac
    if env_name in ["Ant-v3", "Hopper-v3", "Walker2d-v3"]:
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_starts=10_000,
            use_sde=False,
        )
        n_train_steps = 1_000_000 if args.train_steps == -1 else args.train_steps
    elif env_name in ["Humanoid-v3"]:
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_starts=10_000,
            use_sde=True,
        )
        n_train_steps = 2_000_000 if args.train_steps == -1 else args.train_steps
    else:
        raise ValueError(f"Unsupported Environment: {env_name}")
    print(f"Training {env_name} expert for {n_train_steps} steps.")

    with ProgressBarManager(n_train_steps) as progress_callback:
        model.learn(n_train_steps, callback=[progress_callback, eval_callback])

    mean_rewards, std_rewards = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"Expert performance: {mean_rewards=:.4f}, {std_rewards=:.4f}")

    last_ckpt_path = str(Path(save_dir, f"{env_name}_last"))
    model.save(last_ckpt_path)
    print(f"Saved expert to {last_ckpt_path}")
