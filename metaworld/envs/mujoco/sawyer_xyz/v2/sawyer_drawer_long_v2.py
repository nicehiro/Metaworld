import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDrawerLongEnvV2(SawyerXYZEnv):
    def __init__(self, tasks=None, render_mode=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.max_path_length = 1000

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_angle": np.array(
                [
                    0.3,
                ],
                dtype=np.float32,
            ),
            "obj_init_pos": np.array([0.0, 0.9, 0.0], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        hand_goal_low = self.hand_low
        hand_goal_high = self.hand_high
        block_goal_low = np.array([-0.2, 0.55, 0.05])
        block_goal_high = np.array([0.2, 0.75, 0.3])
        goal_low = np.concatenate([hand_goal_low, block_goal_low])
        goal_high = np.concatenate([hand_goal_high, block_goal_high])

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer_long.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            handler_target_to_obj,
            block_target_to_obj,
            handler_reach,
            block_in_place,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(reward),
            "handle_target_distance": float(handler_target_to_obj),
            "block_target_distance": float(block_target_to_obj),
            "handle_success": float(handler_reach),
            "block_success": float(block_in_place),
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("objGeom")

    def _get_pos_objects(self):
        # get position of drawer and block
        drawer_pos = self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])
        block_pos = self.get_body_com("my_block")
        return np.concatenate([drawer_pos, block_pos])

    def _get_quat_objects(self):
        # get orientation of drawer and block
        drawer_quat = self.data.body("drawer_link").xquat
        block_quat = self.data.body("my_block").xquat
        return np.concatenate([drawer_quat, block_quat])

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos

        # Set _target_pos to current drawer position (closed) minus an offset
        # self._target_pos = self.obj_init_pos + np.array(
        #     [0.0, -0.16 - self.maxDist, 0.09]
        # )
        drawer_target_pos = self.obj_init_pos + np.array([0.0, -0.16, 0.09])
        # set block target inside drawer
        block_target_pos = self.get_body_com("drawer_link")
        self._target_pos = np.concatenate([drawer_target_pos, block_target_pos])

        mujoco.mj_forward(self.model, self.data)

        self.obj_init_pos = self._get_pos_objects()

        self.grasp_count = 10

        return self._get_obs()

    def compute_reward(self, action, obs):
        return self._compute_sparse_reward(action, obs)

    def _compute_sparse_reward(self, action, obs):
        handler_obj = obs[4:7]
        block_obj = obs[11:14]

        handler_target = self._target_pos[:3]
        block_target = self._target_pos[3:]

        # check if the drawer is close
        handler_target_to_obj = np.linalg.norm(handler_obj - handler_target)
        handler_reach = handler_target_to_obj < 0.02

        # check if the block is inside the drawer
        block_target_to_obj = np.linalg.norm(block_obj - block_target)
        block_in_place = block_target_to_obj < 0.04
        
        reward = 1 if handler_reach and block_in_place else 0

        return (reward, handler_target_to_obj, block_target_to_obj, handler_reach, block_in_place)

    def _compute_dense_reward(self, action, obs):
        gripper = obs[:3]
        handle = obs[4:7]

        handle_error = np.linalg.norm(handle - self._target_pos)

        reward_for_opening = reward_utils.tolerance(
            handle_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
        )

        handle_pos_init = self._target_pos + np.array([0.0, self.maxDist, 0.0])
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3.0, 3.0, 1.0])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid="long_tail",
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        return (
            reward,
            np.linalg.norm(handle - gripper),
            obs[3],
            handle_error,
            reward_for_caging,
            reward_for_opening,
        )
    
    def get_demo_action_(self, act, obs):
        # return the demonstration action for the current observation
        # (used for imitation learning)
        handle_pos = obs[4:7]
        block_pos = obs[11:14]
        gripper_pos = obs[:3] + np.array([0, 0, -0.025])

        handle_open_target = self._target_pos[:3] + np.array([0, -0.2, 0])
        handle_close_target = self._target_pos[:3]
        drawer_center = self.get_body_com("drawer_link")

        print(f"gripper_pos: {gripper_pos}")
        print(f"block_pos: {block_pos}")
        print(f"block_target_pos: {drawer_center}")
        print(f"drawer_center: {drawer_center}")

        block_success = np.linalg.norm(drawer_center - block_pos) < 0.04
        handle_open_succes = np.linalg.norm(handle_pos - handle_open_target) < 0.04
        handle_close_success = np.linalg.norm(handle_pos - handle_close_target) < 0.03

        # check if the drawer is close
        if handle_close_success and block_success:
            # already closed
            return np.array([0, 0, 0, 0])
        
        # check if the gripper is near the handler
        if np.linalg.norm(gripper_pos - handle_pos) < 0.02 and block_success:
            print(6)
            # close the gripper
            grip_action = 0
            # close the drawer
            drawer_action = (handle_close_target - handle_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is on the top of the handler
        handler_pos_xy = handle_pos[:2]
        gripper_pos_xy = gripper_pos[:2]
        if np.linalg.norm(handler_pos_xy - gripper_pos_xy) < 0.02 and block_success:
            print(5)
            # open the gripper
            grip_action = 0
            # move the gripper to the handler
            drawer_action = (handle_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])
        
        # check if the block is in the drawer and the gripper is top of the block
        handle_top_pos = handle_pos + np.array([0, 0, 0.11])
        if block_success and np.linalg.norm(gripper_pos[2] - block_pos[2]) > 0.12:
            print(4)
            # open the gripper
            grip_action = 0
            # move the gripper
            drawer_action = (handle_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])
        
        # check if the block is in the drawer and the gripper is near the block
        block_target_top_pos = drawer_center + np.array([0, 0, 0.12])
        if block_success and np.linalg.norm(gripper_pos[:2] - block_pos[:2]) < 0.02:
            print(3)
            if self.grasp_count > 0:
                self.grasp_count -= 1
                # open the gripper
                grip_action = 0
                # move the block
                drawer_action = [0, 0, 0]
                return np.concatenate([drawer_action, [grip_action]])
            else:
                self.grasp_count = 10
                # open the gripper
                grip_action = 0
                # release
                drawer_action = (block_target_top_pos - drawer_center) * 20
                return np.concatenate([drawer_action, [grip_action]])
            
        # check if the gripper is near the block and gripper is top of the block target
        if np.linalg.norm(gripper_pos - block_pos) < 0.02 and np.linalg.norm(gripper_pos[:2] - block_target_top_pos[:2]) < 0.02:
            print(2)
            # close the gripper
            grip_action = 1
            # move the block to the target
            drawer_action = (drawer_center - block_pos) * 20
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the block and gripper is on the top of the block
        block_top_pos = block_pos.copy()
        block_top_pos[2] = 0.18
        print(f"block_top_pos: {block_top_pos}")
        if np.linalg.norm(gripper_pos - block_pos) < 0.02 and np.linalg.norm(gripper_pos - block_top_pos) < 0.02:
            print(1)
            # close the gripper
            grip_action = 1
            # move the block to the top of the target
            drawer_action = (block_target_top_pos - block_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the block
        if np.linalg.norm(gripper_pos - block_pos) < 0.02:
            print(0)
            if self.grasp_count > 0:
                self.grasp_count -= 1
                # close the gripper
                grip_action = 1
                # move the block
                drawer_action = [0, 0, 0]
                return np.concatenate([drawer_action, [grip_action]])
            else:
                self.grasp_count = 10
                # close the gripper
                grip_action = 1
                # lift the block
                drawer_action = (block_top_pos - gripper_pos) * 30
                return np.concatenate([drawer_action, [grip_action]])
        
        # check if the gripper is on the top of the block
        if np.linalg.norm(gripper_pos[:2] - block_top_pos[:2]) < 0.02:
            print(-1)
            # open the gripper
            grip_action = 0
            # move the gripper to the block
            drawer_action = (block_pos - gripper_pos) * 30
            return np.concatenate([drawer_action, [grip_action]])
        
        # check if drawer is open and gripper is on the top of the handle
        # handle_open_top_pos = handle_open_target + np.array([0, 0, 0.05])
        if handle_open_succes and np.linalg.norm(gripper_pos[2] - handle_pos[2]) > 0.04:
            print(-2)
            # open the gripper
            grip_action = 0
            # move the gripper to the block
            drawer_action = (block_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if drawer is open and gripper is near the handle
        handler_pos_xy = handle_pos[:2]
        gripper_pos_xy = gripper_pos[:2]
        if handle_open_succes and np.linalg.norm(gripper_pos_xy - handler_pos_xy) < 0.02:
            # move the gripper to the top of handle
            print(-3)
            if self.grasp_count > 0:
                self.grasp_count -= 1
                grip_action = 0
                drawer_action = [0, 0, 0]
                return np.concatenate([drawer_action, [grip_action]])
            else:
                self.grasp_count = 10
                grip_action = 0
                drawer_action = (handle_top_pos - gripper_pos) * 10
                return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the handler
        if np.linalg.norm(gripper_pos - handle_pos) < 0.02:
            print(-4)
            # close the gripper
            grip_action = 1
            # open the drawer
            drawer_action = (handle_open_target - handle_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is on the top of the handler
        handler_top_pos = handle_pos + np.array([0, 0, 0.1])
        if np.linalg.norm(handler_pos_xy - gripper_pos_xy) < 0.02:
            print(-5)
            # open the gripper
            grip_action = 0
            # move the gripper to the handler
            drawer_action = (handle_pos - gripper_pos) * 20
            return np.concatenate([drawer_action, [grip_action]])
        else:
            print(-6)
            # block moved
            # move to the top of drawer handle
            grip_action = 0
            drawer_action = (handler_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

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

        self._last_stable_obs = np.clip(
            self._last_stable_obs,
            a_max=self.sawyer_observation_space.high,
            a_min=self.sawyer_observation_space.low,
            dtype=np.float64,
        )
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        # step will never return a terminate==True if there is a success
        terminal = False
        if info["handle_success"] and info["block_success"]:
            terminal = True
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True
        return (
            np.array(self._last_stable_obs, dtype=np.float64),
            reward,
            False,
            truncate,
            info,
        )


class TrainDrawerLongv2(SawyerDrawerLongEnvV2):
    tasks = None

    def __init__(self):
        SawyerDrawerLongEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestDrawerLongv2(SawyerDrawerLongEnvV2):
    tasks = None

    def __init__(self):
        SawyerDrawerLongEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
