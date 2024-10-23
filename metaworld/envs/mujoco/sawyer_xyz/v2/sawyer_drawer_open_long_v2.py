import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)
from metaworld.envs.mujoco.sawyer_xyz.sawyer_goal_env import SawyerGoalEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_visual_env import SawyerVisualEnv


class SawyerDrawerOpenLongEnvV2(SawyerGoalEnv):
    def __init__(self, tasks=None, render_mode=None):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        drawer_low = np.array((-0.1, 0.9, 0.0))
        drawer_high = np.array((0.1, 0.9, 0.0))

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.max_path_length = 300

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
            "hand_init_pos": np.array([0, 0.5, 0.4], dtype=np.float32),
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

        self._random_reset_drawer_space = Box(drawer_low, drawer_high)

        # set block random positio in front of drawer
        block_low = drawer_low.copy() + np.array([-0.1, 0.2, 0.0])
        block_high = drawer_high.copy() + np.array([0.1, 0.2, 0.0])
        self._random_reset_block_space = Box(block_low, block_high)

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer_open_long.xml")

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

    def _get_pos_achieve_goal(self, obs):
        handle_pos = obs[4:7]
        block_pos = obs[11:14]
        return np.concatenate([handle_pos, block_pos])

    def reset_model(self):
        self._freeze_rand_vec = False
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos, self.block_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "my_block")
        ] = self.block_init_pos

        # Set _target_pos to current drawer position (closed) minus an offset
        drawer_target_pos = self.obj_init_pos + np.array(
            [0.0, -0.16 - self.maxDist, 0.09]
        )
        # set block target away from drawer
        block_pos = self.get_body_com("my_block")
        if np.random.rand() > 0.5:
            block_target_pos = np.array(
                [self.obj_init_pos[0] - 0.2, block_pos[1], block_pos[2]]
            )
        else:
            block_target_pos = np.array(
                [self.obj_init_pos[0] + 0.2, block_pos[1], block_pos[2]]
            )
        self._target_pos = np.concatenate([drawer_target_pos, block_target_pos])
        # self._target_pos = self.obj_init_pos + np.array(
        #     [0.0, -0.16 - self.maxDist, 0.09]
        # )
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def get_demo_action_(self, obs):
        # return the demonstration action for the current observation
        # (used for imitation learning)
        state_observation = obs["observation"]

        drawer_handle_pos = state_observation[4:7]
        block_pos = state_observation[11:14]
        gripper_pos = state_observation[:3]

        # print(block_pos)

        # drawer_handle_target_pos = obs['desired_goal'][:3]
        # block_target_pos = obs['desired_goal'][3:]
        drawer_handle_target_pos = self._target_pos[:3]
        block_target_pos = self._target_pos[3:]

        # check block target is left or right of the block
        if block_target_pos[0] < block_pos[0]:
            # left
            block_hold_pos = block_pos + np.array([0.06, 0, 0])
            block_hold_top_pos = block_hold_pos + np.array([0, 0, 0.1])
        else:
            # right
            block_hold_pos = block_pos + np.array([-0.06, 0, 0])
            block_hold_top_pos = block_hold_pos + np.array([0, 0, 0.1])

        # check if the drawer is open
        if np.linalg.norm(drawer_handle_pos - drawer_handle_target_pos) < 0.02:
            # print(7)
            # already opened
            return np.array([0, 0, 0, 0])

        # check if the gripper is near the handler
        if np.linalg.norm(gripper_pos - drawer_handle_pos) < 0.02:
            # print(6)
            # close the gripper
            grip_action = 1
            # open the drawer
            drawer_action = (drawer_handle_target_pos - drawer_handle_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is on the top of the handler
        handler_pos_xy = drawer_handle_pos[:2]
        gripper_pos_xy = gripper_pos[:2]
        if np.linalg.norm(handler_pos_xy - gripper_pos_xy) < 0.02:
            # print(5)
            # open the gripper
            grip_action = 0
            # move the gripper to the handler
            drawer_action = (drawer_handle_pos - gripper_pos) * 20
            return np.concatenate([drawer_action, [grip_action]])

        # check if the block is moved
        handler_top_pos = drawer_handle_pos + np.array([0, 0, 0.1])
        if np.linalg.norm(block_pos - block_target_pos) < 0.02:
            # print(4)
            # block moved
            # move to the top of drawer handle
            grip_action = 0
            drawer_action = (handler_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the block
        # block_left_pos = block_pos + np.array([-0.06, 0, 0])
        if np.linalg.norm(gripper_pos - block_hold_pos) < 0.02:
            # print(3)
            # close the gripper
            grip_action = 1
            # move the block
            block_action = (block_target_pos - block_pos) * 10
            return np.concatenate([block_action, [grip_action]])

        # check if the gripper is on the top of the block
        # block_left_top_pos = block_left_pos + np.array([0, 0, 0.1])
        if np.linalg.norm(gripper_pos[:2] - block_hold_top_pos[:2]) < 0.02:
            # print(2)
            # close the gripper
            grip_action = 1
            # move the gripper to the block
            block_action = (block_hold_pos - gripper_pos) * 5
            return np.concatenate([block_action, [grip_action]])
        else:
            # move the gripper to the top of block
            # print(1)
            grip_action = 1
            block_action = (block_hold_top_pos - gripper_pos) * 5
            return np.concatenate([block_action, [grip_action]])

        return np.array([0, 0, 0, 0])


class TrainDrawerOpenLongv2(SawyerDrawerOpenLongEnvV2):
    tasks = None

    def __init__(self):
        SawyerDrawerOpenLongEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestDrawerOpenLongv2(SawyerDrawerOpenLongEnvV2):
    tasks = None

    def __init__(self):
        SawyerDrawerOpenLongEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)