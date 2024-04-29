import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)

from metaworld.envs.mujoco.sawyer_xyz.sawyer_visual_env import SawyerVisualEnv


class SawyerDrawerLongEnvV3(SawyerVisualEnv):
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

        self._random_reset_drawer_space = Box(drawer_low, drawer_high)

        # set block random positio in front of drawer
        block_low = drawer_low.copy() + np.array([0.2, 0.15, 0.0])
        block_high = drawer_high.copy() + np.array([0.3, 0.2, 0.0])
        self._random_reset_block_space = Box(block_low, block_high)

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer_long.xml")

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

    def _get_state_rand_vec(self):
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else:
            drawer_rand_vec = np.random.uniform(
                self._random_reset_drawer_space.low,
                self._random_reset_drawer_space.high,
                size=self._random_reset_drawer_space.low.size,
            ).astype(np.float64)
            # set _random_reset_block_space here
            # since the block position is dependent on the drawer position
            self._random_reset_block_space = Box(
                low=drawer_rand_vec + np.array([0.15, -0.25, 0.05]),
                high=drawer_rand_vec + np.array([0.2, -0.2, 0.05]),
            )
            block_rand_vec = np.random.uniform(
                self._random_reset_block_space.low,
                self._random_reset_block_space.high,
                size=self._random_reset_block_space.low.size,
            ).astype(np.float64)
            # self._last_rand_vec = None
            return (drawer_rand_vec, block_rand_vec)

    def reset_model(self):
        self._freeze_rand_vec = False
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos, self.block_init_pos = self._get_state_rand_vec()

        self._image_target_goal = self.sample_goal()

        self.grasp_count = 10

        return self._get_obs()

    def sample_goal(self):
        # Set mujoco body to computed position
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos
        qpos, qvel = self.get_env_state()
        qpos[10:13] = self.block_init_pos
        self.set_env_state((qpos, qvel))

        # update
        mujoco.mj_forward(self.model, self.data)

        # Set _target_pos to current drawer position (closed) minus an offset
        drawer_target_pos = self.obj_init_pos + np.array([0.0, -0.16, 0.09])
        # set block target inside drawer
        block_target_pos = self.get_body_com("drawer_link")
        self._target_pos = np.concatenate([drawer_target_pos, block_target_pos])

        # set mujoco body to target position
        qpos, qvel = self.get_env_state()
        qpos[10:13] = block_target_pos
        self.set_env_state((qpos, qvel))

        mujoco.mj_forward(self.model, self.data)

        image_desired_goal = self.render()

        # back to original position
        qpos, qvel = self.get_env_state()
        qpos[10:13] = self.block_init_pos
        self.set_env_state((qpos, qvel))

        # update
        mujoco.mj_forward(self.model, self.data)

        return image_desired_goal

    def get_demo_action_(self, obs):
        # return the demonstration action for the current observation
        # (used for imitation learning)
        state_observation = obs["observation"]

        handle_pos = state_observation[4:7]
        block_pos = state_observation[11:14]
        gripper_pos = state_observation[:3] + np.array([0, 0, -0.025])

        handle_open_target = self._target_pos[:3] + np.array([0, -0.2, 0])
        handle_close_target = self._target_pos[:3]
        # block target is drawer center
        drawer_center = self.get_body_com("drawer_link")

        # print(f"gripper_pos: {gripper_pos}")
        # print(f"block_pos: {block_pos}")
        # print(f"block_target_pos: {drawer_center}")
        # print(f"drawer_center: {drawer_center}")

        block_success = np.linalg.norm(drawer_center - block_pos) < 0.04
        handle_open_succes = np.linalg.norm(handle_pos - handle_open_target) < 0.04
        handle_close_success = np.linalg.norm(handle_pos - handle_close_target) < 0.03

        # check if the drawer is close
        if handle_close_success and block_success:
            # already closed
            return np.array([0, 0, 0, 0])

        # check if the gripper is near the handler
        if np.linalg.norm(gripper_pos - handle_pos) < 0.02 and block_success:
            # print(6)
            # close the gripper
            grip_action = 0
            # close the drawer
            drawer_action = (handle_close_target - handle_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is on the top of the handler
        handler_pos_xy = handle_pos[:2]
        gripper_pos_xy = gripper_pos[:2]
        if np.linalg.norm(handler_pos_xy - gripper_pos_xy) < 0.02 and block_success:
            # print(5)
            # open the gripper
            grip_action = 0
            # move the gripper to the handler
            drawer_action = (handle_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the block is in the drawer and the gripper is top of the block
        handle_top_pos = handle_pos + np.array([0, 0, 0.11])
        if block_success and np.linalg.norm(gripper_pos[2] - block_pos[2]) > 0.12:
            # print(4)
            # open the gripper
            grip_action = 0
            # move the gripper
            drawer_action = (handle_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the block is in the drawer and the gripper is near the block
        block_target_top_pos = drawer_center + np.array([0, 0, 0.12])
        if block_success and np.linalg.norm(gripper_pos[:2] - block_pos[:2]) < 0.02:
            # print(3)
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
        if (
            np.linalg.norm(gripper_pos - block_pos) < 0.02
            and np.linalg.norm(gripper_pos[:2] - block_target_top_pos[:2]) < 0.02
        ):
            # print(2)
            # close the gripper
            grip_action = 1
            # move the block to the target
            drawer_action = (drawer_center - block_pos) * 20
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the block and gripper is on the top of the block
        block_top_pos = block_pos.copy()
        block_top_pos[2] = 0.19
        # print(f"block_top_pos: {block_top_pos}")
        if (
            np.linalg.norm(gripper_pos - block_pos) < 0.02
            and np.linalg.norm(gripper_pos - block_top_pos) < 0.02
        ):
            # print(1)
            # close the gripper
            grip_action = 1
            # move the block to the top of the target
            drawer_action = (block_target_top_pos - block_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is near the block
        if np.linalg.norm(gripper_pos - block_pos) < 0.02:
            # print(0)
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
            # print(-1)
            # open the gripper
            grip_action = 0
            # move the gripper to the block
            drawer_action = (block_pos - gripper_pos) * 30
            return np.concatenate([drawer_action, [grip_action]])

        # check if drawer is open and gripper is on the top of the handle
        # handle_open_top_pos = handle_open_target + np.array([0, 0, 0.05])
        if handle_open_succes and np.linalg.norm(gripper_pos[2] - handle_pos[2]) > 0.04:
            # print(-2)
            # open the gripper
            grip_action = 0
            # move the gripper to the block
            drawer_action = (block_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if drawer is open and gripper is near the handle
        handler_pos_xy = handle_pos[:2]
        gripper_pos_xy = gripper_pos[:2]
        if (
            handle_open_succes
            and np.linalg.norm(gripper_pos_xy - handler_pos_xy) < 0.02
        ):
            # move the gripper to the top of handle
            # print(-3)
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
            # print(-4)
            # close the gripper
            grip_action = 1
            # open the drawer
            drawer_action = (handle_open_target - handle_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])

        # check if the gripper is on the top of the handler
        handler_top_pos = handle_pos + np.array([0, 0, 0.1])
        if np.linalg.norm(handler_pos_xy - gripper_pos_xy) < 0.02:
            # print(-5)
            # open the gripper
            grip_action = 0
            # move the gripper to the handler
            drawer_action = (handle_pos - gripper_pos) * 20
            return np.concatenate([drawer_action, [grip_action]])
        else:
            # print(-6)
            # block moved
            # move to the top of drawer handle
            grip_action = 0
            drawer_action = (handler_top_pos - gripper_pos) * 5
            return np.concatenate([drawer_action, [grip_action]])


class TrainDrawerLongv2(SawyerDrawerLongEnvV3):
    tasks = None

    def __init__(self):
        SawyerDrawerLongEnvV3.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestDrawerLongv2(SawyerDrawerLongEnvV3):
    tasks = None

    def __init__(self):
        SawyerDrawerLongEnvV3.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
