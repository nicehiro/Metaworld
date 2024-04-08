import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDrawerCloseLongEnvV2(SawyerXYZEnv):
    _TARGET_RADIUS = 0.04

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

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer_close_long.xml")

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
            "near_object": float(handler_target_to_obj <= 0.03),
            "block_in_place": float(block_target_to_obj <= 0.1),
        }

        return reward, info

    def _get_pos_objects(self):
        # get position of drawer and block
        drawer_pos = self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.05])
        block_pos = self.get_body_com("my_block")
        return np.concatenate([drawer_pos, block_pos])

    def _get_quat_objects(self):
        # get orientation of drawer and block
        drawer_quat = self.data.body("drawer_link").xquat
        block_quat = self.data.body("my_block").xquat
        return np.concatenate([drawer_quat, block_quat])

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed)
        # and current block position
        drawer_target_pos = self.obj_init_pos + np.array([0.0, -0.16, 0.09])
        # set block target inside drawer
        block_target_pos = self.obj_init_pos + np.array([0.0, -0.08, 0.09])
        self._target_pos = np.concatenate([drawer_target_pos, block_target_pos])
        # self._target_pos = self.obj_init_pos + np.array([0.0, -0.16, 0.09])
        # Pull drawer out all the way and mark its starting position
        self._set_obj_xyz(-self.maxDist)
        self.obj_init_pos = self._get_pos_objects()

        # time for grasp blcok
        self.grasp_count = 10

        return self._get_obs()

    @property
    def sawyer_observation_space(self):
        # obj: drawer (7) + block (7)
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
        # goal: position of the handle (3) + position of block (3)
        goal_low = np.zeros(6) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(6) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0
        return Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
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
    
    def compute_reward(self, action, obs):
        return self._compute_sparse_reward(action, obs)
    
    def _compute_sparse_reward(self, action, obs):
        handler_obj = obs[4:7]
        block_obj = obs[11:14]

        handler_target = self._target_pos[:3]
        block_target = self._target_pos[3:]

        # check if the drawer is open
        handler_target_to_obj = np.linalg.norm(handler_obj - handler_target)
        handler_reach = reward_utils.tolerance(
            handler_target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=0,
            sigmoid="long_tail",
        )

        # check if the block is inside the drawer
        block_target_to_obj = np.linalg.norm(block_obj - block_target)
        block_in_place = reward_utils.tolerance(
            block_target_to_obj,
            bounds=(0, 0.1),
            margin=0,
            sigmoid="long_tail",
        )
        
        reward = 1 if handler_reach and block_in_place else 0

        return (reward, handler_target_to_obj, block_target_to_obj, handler_reach, block_in_place)

    def _compute_dense_reward(self, action, obs):
        handler_obj = obs[4:7]
        block_obj = obs[11:14]
        obj = np.concatenate([handler_obj, block_obj])

        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj - target
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = self.obj_init_pos - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        handle_reach_radius = 0.005
        tcp_to_obj = np.linalg.norm(handler_obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos[:3] - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid="gaussian",
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS + 0.015:
            reward = 1.0

        reward *= 10

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)

    def get_demo_action_(self, act, obs):
        # return the demonstration action for the current observation
        # (used for imitation learning)
        handle_pos = obs[4:7]
        block_pos = obs[11:14]
        gripper_pos = obs[:3] + np.array([0, 0, -0.025])

        handle_target_pos = self._target_pos[:3]
        block_target_pos = handle_target_pos
        block_target_pos[2] = 0.0658

        print(f"gripper_pos: {gripper_pos}")
        print(f"block_pos: {block_pos}")
        print(f"block_target_pos: {block_target_pos}")

        block_target_to_obj = np.linalg.norm(block_target_pos - block_pos)

        # check if the drawer is close
        if np.linalg.norm(handle_pos - handle_target_pos) < 0.02:
            # already closed
            return np.array([0, 0, 0, 0])
        
        # check if the gripper is near the handler
        if np.linalg.norm(gripper_pos - handle_pos) < 0.02:
            print(6)
            # close the gripper
            grip_action = 0
            # open the drawer
            drawer_action = (handle_target_pos - handle_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is on the top of the handler
        handler_pos_xy = handle_pos[:2]
        gripper_pos_xy = gripper_pos[:2]
        if np.linalg.norm(handler_pos_xy - gripper_pos_xy) < 0.02:
            print(5)
            # open the gripper
            grip_action = 0
            # move the gripper to the handler
            drawer_action = (handle_pos - gripper_pos)
            return np.concatenate([drawer_action, [grip_action]])
        
        # check if the block is in the drawer and the gripper is near the block
        block_target_top_pos = block_target_pos + np.array([0, 0, 0.12])
        if np.linalg.norm(block_target_pos - block_pos) < 0.04 and np.linalg.norm(gripper_pos[:2] - block_pos[:2]) < 0.02:
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
                drawer_action = (block_target_top_pos - block_pos) * 5
                return np.concatenate([drawer_action, [grip_action]])

        # check if the block is in the drawer and the gripper is top of the block
        handle_top_pos = handle_pos + np.array([0, 0, 0.1])
        if np.linalg.norm(block_target_pos - block_pos) < 0.04:
            print(4)
            # open the gripper
            grip_action = 0
            # move the gripper
            drawer_action = (handle_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])
            
        # check if the gripper is near the block and gripper is top of the block target
        if np.linalg.norm(gripper_pos - block_pos) < 0.02 and np.linalg.norm(gripper_pos[:2] - block_target_top_pos[:2]) < 0.02:
            print(2)
            # close the gripper
            grip_action = 1
            # move the block to the target
            drawer_action = (block_target_pos - block_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the block and gripper is on the top of the block
        block_top_pos = block_pos.copy()
        block_top_pos[2] = 0.2
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
                drawer_action = (block_top_pos - gripper_pos) * 10
                return np.concatenate([drawer_action, [grip_action]])
        
        # check if the gripper is on the top of the block
        if np.linalg.norm(gripper_pos[:2] - block_top_pos[:2]) < 0.02:
            print(-1)
            # open the gripper
            grip_action = 0
            # move the gripper to the block
            drawer_action = (block_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])
        else:
            # move the gripper to the top of block
            print(-2)
            grip_action = 0
            drawer_action = (block_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])


class TrainDrawerCloseLongv2(SawyerDrawerCloseLongEnvV2):
    tasks = None

    def __init__(self):
        SawyerDrawerCloseLongEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestDrawerCloseLongv2(SawyerDrawerCloseLongEnvV2):
    tasks = None

    def __init__(self):
        SawyerDrawerCloseLongEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
