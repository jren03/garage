from typing import Tuple
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


# Obtained from https://github.com/JannerM/mbpo/blob/master/mbpo/env/ant.py
class AntTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self) -> None:
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)

    def env_name(self) -> str:
        return "AntTruncatedObs"

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self) -> None:
        self.viewer.cam.distance = self.model.stat.extent * 0.5
