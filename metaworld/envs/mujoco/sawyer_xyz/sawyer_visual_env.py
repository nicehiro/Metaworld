import numpy as np
from gymnasium.spaces import Dict, Box
from metaworld.envs.mujoco.sawyer_xyz.sawyer_goal_env import SawyerGoalEnv


class SawyerVisualEnv(SawyerGoalEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_pos_achieve_goal():
        raise NotImplementedError

    def _get_obs(self):
        return self._get_obs_dict()

    def _get_obs_dict(self):
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs

        return dict(
            image_observation=self._get_obs_img(),
            image_desired_goal=self._image_target_goal,
            image_achieved_goal=self._get_obs_img(),
            observation=obs,
            desired_goal=self._get_pos_goal(),
            achieved_goal=self._get_pos_achieve_goal(obs),
        )

    def _get_obs_img(self):
        # Generate image observation of the environment
        image_observation = self.render()
        return image_observation

    def sample_goal(self):
        raise NotImplementedError

    @property
    def sawyer_observation_space(self):
        # return image observation, achieved_goal, desired_goal
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
        goal_low = np.zeros(6) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(6) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0
        self.observation_space = Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,  # 3
                    gripper_low,  # 1
                    obj_low,  # 14
                    self._HAND_SPACE.low,  # 3
                    gripper_low,  # 1
                    obj_low,  # 14
                    goal_low,  # 6
                )
            ),
            np.hstack(
                (
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.image_goal_space = Box(low=0, high=255, shape=(48, 48, 3), dtype=np.uint8)
        return Dict(
            {
                "observation": self.observation_space,
                "desired_goal": self.goal_space,
                "achieved_goal": self.goal_space,
                "image_observation": self.image_goal_space,
                "image_desired_goal": self.image_goal_space,
                "image_achieved_goal": self.image_goal_space,
            }
        )
