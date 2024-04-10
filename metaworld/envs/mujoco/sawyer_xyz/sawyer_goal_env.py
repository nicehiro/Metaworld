import numpy as np
from gymnasium.spaces import Dict, Box
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from gymnasium.utils import RecordConstructorArgs, seeding


class SawyerGoalEnv(SawyerXYZEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_obs(self):
        return self._get_obs_dict()
    
    def _get_pos_achieve_goal():
        raise NotImplementedError

    def _get_obs_dict(self):
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the goal position to form a single flat observation.

        Returns:
            np.ndarray: The flat observation array (39 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        
        return dict(
            observation=obs,
            desired_goal=self._get_pos_goal(),
            achieved_goal=self._get_pos_achieve_goal(obs), # drawer handle and block
        )

    @property
    def sawyer_observation_space(self):
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
                    self._HAND_SPACE.low, # 3
                    gripper_low, # 1
                    obj_low, # 14
                    self._HAND_SPACE.low, # 3
                    gripper_low, # 1
                    obj_low, # 14
                    goal_low, # 6
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

        return Dict({
            'observation': self.observation_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        })

    @_assert_task_is_set
    def evaluate_state(self, achieved_goal, desired_goal, action):
        reward = self.compute_reward(achieved_goal, desired_goal, action)

        info = {}

        return reward, info

    def reset(self, seed=None, options=None):
        self.curr_path_length = 0

        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self._reset_simulation()

        obs = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()

        state_observation = obs["observation"]
        self._prev_obs = state_observation[:18].copy()
        state_observation[18:36] = self._prev_obs
        state_observation = np.float64(state_observation)

        obs["observation"] = state_observation
        return obs, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal.
        handle_dist = np.linalg.norm(achieved_goal[..., :3] - desired_goal[..., :3], axis=-1)
        block_dist = np.linalg.norm(achieved_goal[..., 3:] - desired_goal[..., 3:], axis=-1)

        handle_reward = -(handle_dist > 0.03).astype(np.float32)
        block_reward = -(block_dist > 0.03).astype(np.float32)

        reward = handle_reward + block_reward

        return reward

    @_assert_task_is_set
    def step(self, action):
        assert len(action) == 4, f"Actions should be size 4, got {len(action)}"
        self.set_xyz_action(action[:3])
        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,
                False,  # termination flag always False
                {  # info
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )

        self._last_stable_obs = self._get_obs()

        self._last_stable_obs["observation"] = np.clip(
            self._last_stable_obs["observation"],
            a_max=self.sawyer_observation_space["observation"].high,
            a_min=self.sawyer_observation_space["observation"].low,
            dtype=np.float64,
        )
        reward, info = self.evaluate_state(
            self._last_stable_obs["achieved_goal"],
            self._last_stable_obs["desired_goal"],
            action,
        )
        # step will never return a terminate==True if there is a success
        terminal = False
        # if info["handle_success"] and info["block_success"]:
        #     terminal = True
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True
        return (
            self._last_stable_obs,
            reward,
            False,
            truncate,
            info,
        )