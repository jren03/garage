import os
import sys
from typing import List, Tuple, Optional

import gym
import numpy as np
import torch
from termcolor import cprint


class RewardWrapper(gym.Wrapper):
    """
    Relabel current reward based on current state and action.
    """

    def __init__(self, env: gym.Env, discriminator: torch.nn.Module) -> None:
        super().__init__(env)
        self.env = env
        self.cur_state = None
        self.discriminator = discriminator
        self.low = env.action_space.low
        self.high = env.action_space.high

        env_name = (
            env.unwrapped.spec.id
            if hasattr(env.unwrapped.spec, "id")
            else env.env_name()
        )
        cprint(f"Adding RewardWrapper to {env_name}", attrs=["bold"])

    def reset(self) -> None:
        obs = self.env.reset()
        self.cur_state = obs
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        next_state, _, done, info = self.env.step(action)
        sa_pair = np.concatenate((self.cur_state, action))
        reward = -(
            self.discriminator(torch.tensor(sa_pair, dtype=torch.float).to("cuda"))
        )
        self.cur_state = next_state
        return next_state, reward, done, info


class TremblingHandWrapper(gym.Wrapper):
    """
    Add noise to action space according to `p_tremble`.
    """

    def __init__(self, env: gym.Env, p_tremble: float = 0.01) -> None:
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble
        self.rng = np.random.default_rng()

        env_name = (
            env.unwrapped.spec.id
            if hasattr(env.unwrapped.spec, "id")
            else env.env_name()
        )
        if float(p_tremble) != 0.0:
            cprint(
                f"Shaking {env_name} with probability {self.p_tremble}", attrs=["bold"]
            )

    def reset(self) -> None:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        if self.rng.random() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)


class GoalWrapper(gym.Wrapper):
    """
    Add goal to observation space.
    """

    def __init__(self, env: gym.Env, goal: np.ndarray) -> None:
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.observation_space.shape[0] + goal.shape[0],),
            dtype=np.float32,
        )
        env_name = (
            env.unwrapped.spec.id
            if hasattr(env.unwrapped.spec, "id")
            else env.env_name()
        )
        cprint(f"Adding GoalWrapper to {env_name}", attrs=["bold"])

    def reset(self) -> np.ndarray:
        with HiddenPrints():
            obs = self.env.reset()
            goal = self.env.target_goal
            return np.concatenate([obs, goal])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        goal = self.env.target_goal
        return np.concatenate([obs, goal]), reward, done, info


class ResetWrapper(gym.Wrapper):
    """
    Reset the environment to a specific state with probability `reset_prob`.
    """

    def __init__(
        self,
        env: gym.Env,
        qpos: List[List[float]],
        qvel: List[List[float]],
        goals: List[List[float]],
        obs: Optional[List[float]] = None,
        actions: Optional[List[float]] = None,
        seeds: Optional[List[float]] = None,
        reset_prob: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.reset_prob = reset_prob
        self.qpos = qpos
        self.qvel = qvel
        self.goals = goals
        self.obs = obs
        self.actions = actions
        self.seeds = seeds
        self.reset_via_qpos = self.obs is None
        self.t = 0
        self.rng = np.random.default_rng()

        env_name = (
            env.unwrapped.spec.id
            if hasattr(env.unwrapped.spec, "id")
            else env.env_name()
        )
        if float(reset_prob) != 0.0:
            cprint(
                f"Adding ResetWrapper to {env_name} with reset probability {reset_prob}",
                attrs=["bold"],
            )

        self.is_maze = "maze" in env_name
        self.T = 1000

    def _rollout_until_timestep(self, traj_num, num_steps):
        obs = self.env.reset(seed=int(self.seeds[traj_num][0]))
        for i in range(num_steps):
            action = self.actions[traj_num][i]
            obs = self.env.step(action)[0]
        return obs

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        if self.rng.random() < self.reset_prob:
            idx = np.random.choice(len(self.qpos))
            t = np.random.choice(min(len(self.qpos[idx]), self.T))
            if self.reset_via_qpos:
                self.env.set_state(self.qpos[idx][t], self.qvel[idx][t])
                obs = self.env.unwrapped._get_obs()
            else:
                obs = self._rollout_until_timestep(traj_num=idx, num_steps=t)
            self.t = t
            if self.is_maze:
                with HiddenPrints():
                    self.env.set_target(tuple(self.goals[idx][t]))
                goal = self.env.target_goal
                obs = np.concatenate([obs, goal])
        else:
            self.t = 0
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.T:
            done = True
        return obs, reward, done, info


class HiddenPrints:
    """
    Suppress print output.
    """

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout
