"""
Basic Robot Environment Wrapper
Robot Specific Functions: self._update_pose(), self.get_ee_pos(), self.get_ee_angle()
Camera Specific Functions: self.render_obs()
Experiment Specific Functions: self.get_info(), self.get_reward(), self.get_observation()
"""
import numpy as np
import time
import gym
import torch
import os
import json

import sys
import os
sys.path.append('/home/weirdlab/Downloads/reaching/environments/franka')
from transformations import add_angles, angle_diff

from gym.spaces import Box, Dict
from franka_simple import FrankaMujocoEnv

class RobotEnv(gym.Env):
    """
    Main interface to interact with the robot.
    """

    def __init__(
        self,
        # robot
        sim=True,
        hz=10,
        DoF=3,
        use_gripper=False,
        # pass IP if not running on NUC
        ip_address=None,
        # observation space
        front_camera=False,
        side_camera=False,
        full_state=False,
        qpos=False,
        ee_pos=False,
        sphere_pos=False,
        sphere_vel=False,
        normalize_obs=False,
        flat_obs=False,
        # task
        max_episode_steps=None,
        goal=None,
        # rendering
        has_renderer=False,
        has_offscreen_renderer=True,
        depth_camera=False,
        img_height=480,
        img_width=480,
        **kwargs,
    ):
        super().__init__()

        # system
        self.max_lin_vel = 0.5
        self.max_rot_vel = 1.5
        self.DoF = DoF
        self.hz = hz
        self.sim = sim

        self._episode_count = 0
        self._max_episode_steps = max_episode_steps
        self._curr_path_length = 0

        if self.DoF == 2:
            self._up_joint_qpos = np.array(
                [
                    -0.01895611,
                    0.3541462,
                    -0.02401299,
                    -1.76869237,
                    0.01435507,
                    1.97038139,
                    0.05868276,
                ]
            )
            self._reset_joint_qpos = np.array(
                [
                    0.02013862,
                    0.50847548,
                    -0.09224909,
                    -2.36841345,
                    0.1598147,
                    2.88097692,
                    0.63428867,
                ]
            )
        else:
            self._up_joint_qpos = None
            self._reset_joint_qpos = np.array([0, 0.423, 0, -1.944, 0.013, 2.219, 0.1])

        # observation space config
        self._flat_obs = flat_obs
        self._normalize_obs = normalize_obs

        self._front_camera = front_camera
        self._side_camera = side_camera
        self._img_height = img_height
        self._img_width = img_width
        self._depth_camera = depth_camera

        self._full_state = full_state
        if self._full_state:
            qpos, ee_pos, sphere_pos, sphere_vel = False, False, False, False
        self._qpos = qpos
        self._ee_pos = ee_pos
        self._sphere_pos = sphere_pos
        self._sphere_vel = sphere_vel

        # action space
        self.use_gripper = use_gripper
        if self.use_gripper:
            self.action_space = Box(
                np.array([-1] * (self.DoF + 1)),  # dx_low, dy_low, dz_low, dgripper_low
                np.array(
                    [1] * (self.DoF + 1)
                ),  # dx_high, dy_high, dz_high, dgripper_high
            )
        else:
            self.action_space = Box(
                np.array([-1] * (self.DoF)),  # dx_low, dy_low, dz_low
                np.array([1] * (self.DoF)),  # dx_high, dy_high, dz_high
            )

        # EE position (x, y, z) + gripper width
        if self.DoF == 2:
            self.ee_space = Box(
                np.array([0.38, -0.25, 0.00]),
                np.array([0.70, 0.28, 0.085]),
            )
        if self.DoF == 3:
            self.ee_space = Box(
                np.array([0.38, -0.25, 0.15, 0.00]),
                np.array([0.70, 0.28, 0.35, 0.085]),
            )
        elif self.DoF == 4:
            # EE position (x, y, z) + gripper width
            self.ee_space = Box(
                np.array([0.55, -0.06, 0.15, -1.57, 0.00]),
                np.array([0.73, 0.28, 0.35, 0.0, 0.085]),
            )

        # joint limits + gripper
        self._jointmin = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045],
            dtype=np.float32,
        )
        self._jointmax = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085],
            dtype=np.float32,
        )
        # joint space + gripper
        self.qpos_space = Box(self._jointmin, self._jointmax)

        # observation space
        env_obs_spaces = {
            "front_camera_obs": Box(0, 255, (100, 100, 3), np.uint8),
            "side_camera_obs": Box(0, 255, (100, 100, 3), np.uint8),
            "lowdim_ee": self.ee_space,
            "lowdim_qpos": self.qpos_space,
            "lowdim_sphere": Box(-np.inf, np.inf, (3,), np.float32),
            "lowdim_sphere_vel": Box(-np.inf, np.inf, (3,), np.float32),
            "full_state": Box(-np.inf, np.inf, (27,), np.float32),
        }
        if not self._front_camera:
            env_obs_spaces.pop("front_camera_obs", None)
        if not self._side_camera:
            env_obs_spaces.pop("side_camera_obs", None)
        if not self._qpos:
            env_obs_spaces.pop("lowdim_qpos", None)
        if not self._ee_pos:
            env_obs_spaces.pop("lowdim_ee", None)
        if not self._sphere_pos:
            env_obs_spaces.pop("lowdim_sphere", None)
        if not self._sphere_vel:
            env_obs_spaces.pop("lowdim_sphere_vel", None)
        if not self._full_state:
            env_obs_spaces.pop("full_state", None)
        self.observation_space_dict = Dict(env_obs_spaces)

        low, high = None, None
        for key in self.observation_space_dict.keys():
            low_tmp = self.observation_space_dict[key].low
            high_tmp = self.observation_space_dict[key].high
            low = low_tmp if low is None else np.concatenate((low, low_tmp))
            high = high_tmp if high is None else np.concatenate((high, high_tmp))
        self.observation_space = Box(low, high, dtype=np.float32)

        # robot configuration
        if self.sim:
            # cameras
            self._has_renderer = has_renderer
            self._has_offscreen_renderer = has_offscreen_renderer
            # robot

            self._robot = FrankaMujocoEnv(
                has_renderer=self._has_renderer,
                has_offscreen_renderer=self._has_offscreen_renderer,
                img_height=self._img_height,
                img_width=self._img_width,
                control_hz=self.hz,
                init_qpos=self._reset_joint_qpos,
                DoF=self.DoF,
                goal=goal,
                **kwargs,
            )
        else:
            # cameras
            from camera_utils.realsense_camera import gather_realsense_cameras
            from camera_utils.multi_camera_wrapper import MultiCameraWrapper

            cameras = gather_realsense_cameras()
            self._camera_reader = MultiCameraWrapper(specific_cameras=cameras)
            # wait for exposure to adjust
            for _ in range(10):
                time.sleep(0.1)
                self.get_images()
            # robot
            from server.robot_interface import RobotInterface

            self._robot = RobotInterface(ip_address=ip_address)

        # load camera calibration
        path = "/home/weirdlab/Downloads/reaching/environments/franka/"
        cam_front_file = "calibration_july14_cam_back.json"
        cam_front_json = json.load(open(os.path.join(path, cam_front_file)))
        cam_left_file = "calibration_july14_cam_side.json"
        cam_left_json = json.load(open(os.path.join(path, cam_left_file)))

        self.camera_intrinsic = {
            "front": self.compute_camera_intrinsic(
                cam_front_json[0]["intrinsics"]["fx"],
                cam_front_json[0]["intrinsics"]["fy"],
                cam_front_json[0]["intrinsics"]["ppx"],
                cam_front_json[0]["intrinsics"]["ppy"]
            ),
            "left": self.compute_camera_intrinsic(
                cam_left_json[0]["intrinsics"]["fx"],
                cam_left_json[0]["intrinsics"]["fy"],
                cam_left_json[0]["intrinsics"]["ppx"],
                cam_left_json[0]["intrinsics"]["ppy"]
            ),
        }

        self.camera_extrinsic = {
            "front": self.compute_camera_extrinsic(
                cam_front_json[0]["camera_base_ori"], cam_front_json[0]["camera_base_pos"]
            ),
            "left": self.compute_camera_extrinsic(
                cam_left_json[0]["camera_base_ori"], cam_left_json[0]["camera_base_pos"]
            ),
        }

        self.data_dict = {
            "actions": [],
            "act_clip": [],
            "observations": [],
            "ee_pose": [],
            "rew": [],
            "done": [],
            "rgb": [],
            "first_rgb": [],
            "last_rgb": [],
            "depth": [],
            "first_depth": [],
            "last_depth": [],
        }

    def compute_camera_intrinsic(self, fx, fy, ppx, ppy):
        return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]])

    def compute_camera_extrinsic(self, ori_mat, pos):
        cam_mat = np.eye(4)
        cam_mat[:3, :3] = ori_mat
        cam_mat[:3, 3] = pos
        return cam_mat

    def step(self, action):
        if not self.use_gripper:
            action = np.append(action, -1.0)

        self.data_dict["actions"] += [action]

        start_time = time.time()

        assert len(action) == (self.DoF + 1)
        action = np.clip(action, -1, 1)

        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        # clip position
        desired_pos, gripper = self._get_valid_pos_and_gripper(
            self._curr_pos + lin_vel, gripper
        )
        desired_angle = add_angles(rot_vel, self._curr_angle)
        # clip angle
        if self.DoF == 4:
            desired_angle[2] = desired_angle[2].clip(
                self.ee_space.low[3], self.ee_space.high[3]
            )

        # ensure control frequency
        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)

        # enforce z constraints
        if self.DoF == 2:
            # when using cylinder EE
            # make sure franka gripper is removed in /desk
            # height cylinder: 0.115 franka gripper: 0.14
            desired_pos[2] = 0.115

        self._update_robot(desired_pos, desired_angle, gripper)

        # ensure control frequency
        comp_time = time.time() - start_time
        sleep_left = max(0, (1 / self.hz) - comp_time)
        if not self.sim:
            time.sleep(sleep_left)

        reward, success = self.compute_reward(action)

        self._curr_path_length += 1
        done = False
        if (
            self._max_episode_steps is not None
            and self._curr_path_length >= self._max_episode_steps
        ):
            done = True

        obs = self.get_observation()
        if self._flat_obs:
            obs = self.flatten_obs(obs)

        # store data
        if not self.sim:
            current_images = self.get_images()
            rgb = np.concatenate(
                (current_images[0]["color_image"], current_images[2]["color_image"]),
                axis=-1,
            )
            depth = np.concatenate(
                (
                    current_images[1]["depth_aligned"][..., np.newaxis],
                    current_images[3]["depth_aligned"][..., np.newaxis],
                ),
                axis=-1,
            )
            self.data_dict["rgb"] += [rgb]
            self.data_dict["depth"] += [depth]
        current_images = self.get_images()

        print(self.camera_intrinsic)
     
        self.data_dict["act_clip"] += [action]
        self.data_dict["observations"] += [obs]
        self.data_dict["ee_pose"] += [self._robot.get_ee_pos()]
        self.data_dict["rew"] += [reward]
        self.data_dict["done"] += [done]

        info = {}
        if self.sim:
            if self._has_renderer:
                self.render(mode="human")
            info = {"zeta": self._robot.get_parameters().copy(), "action": [action], "current_images": [current_images], "obs": [obs]}
          
        if success:
            info["success"] = True
        
            
        return obs, reward, done, info

    def dump_data(self):
        import joblib
        from datetime import datetime

        dt = datetime.now()
        joblib.dump(
            self.data_dict,
            f"dump_{'sim' if self.sim else 'real'}_date_{dt.month}_{dt.day}_time_{dt.hour}_{dt.minute}",
        )

    def flatten_obs(self, obs):
        if self._qpos or self._ee_pos or self._sphere_pos or self._sphere_vel or self._full_state:
            return np.concatenate(list(obs.values()))
        elif self._front_camera or self._side_camera:
            return np.concatenate(list(obs.values()), -1)

    def normalize_ee_obs(self, obs):
        """Normalizes low-dim obs between [-1,1]."""
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2 + min(x)
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        if self._normalize_obs:
            normalized_obs = (
                2 * (obs - self.ee_space.low) / (self.ee_space.high - self.ee_space.low)
                - 1
            )
            return normalized_obs
        else:
            return obs

    def unnormalize_ee_obs(self, obs):
        if self._normalize_obs:
            return (obs + 1) * (
                self.ee_space.high - self.ee_space.low
            ) / 2 + self.ee_space.low
        else:
            return obs

    def normalize_qpos(self, qpos):
        """Normalizes qpos between [-1,1]."""
        # The ranges for the joint limits are taken from
        # the franka emika page: https://frankaemika.github.io/docs/control_parameters.html
        # if self._normalize_obs:
        norm_qpos = (
            2
            * (qpos - self.qpos_space.low)
            / (self.qpos_space.high - self.qpos_space.low)
            - 1
        )
        return norm_qpos
        # else:
        # return qpos

    def reset_gripper(self):
        if self.use_gripper:
            self._robot.update_gripper(0)

    def reset_up_gripper(self):
        if self.use_gripper:
            self._robot.update_gripper(-1)

    def get_parameters(self):
        if self.sim:
            return self._robot.get_parameters()
        else:
            return np.zeros(2)

    def set_parameters(self, params):
        return self._robot.set_parameters(params)

    def set_sphere_pos(self, pos):
        self._robot.set_sphere_pos(pos)

    def set_sphere_radius(self, radius):
        self._robot.set_sphere_radius(radius)

    def get_camera_intrinsic(self, camera_name):
        if self.sim:
            return self._robot.get_camera_intrinsic(camera_name=camera_name)
        else:
            return self.camera_intrinsic[camera_name]

    def get_camera_extrinsic(self, camera_name):
        if self.sim:
            return self._robot.get_camera_extrinsic(camera_name=camera_name)
        else:
            return self.camera_extrinsic[camera_name]

    def reset_task(self, task=None):
        if self.sim:
            self._robot.reset_task()

    def seed(self, seed):
        np.random.seed(seed)
        if self.sim:
            self._robot.seed(seed)

    @property
    def parameter_dim(self):
        return self._robot.parameter_dim

    def _reset_up(self):
        self.reset_up_gripper()

        if self._up_joint_qpos is not None:
            for _ in range(5):
                self._robot.update_joints(self._up_joint_qpos)
                if self.is_robot_init():
                    break
                else:
                    print("moving up failed, trying again")

        obs = self.get_observation()
        if self._flat_obs:
            obs = self.flatten_obs(obs)

        return obs

    def _reset_down(self):
        self.reset_gripper()
        for _ in range(5):
            self._robot.update_joints(self._reset_joint_qpos)
            if self.is_robot_reset():
                break
            else:
                print("moving down, trying again")

        obs = self.get_observation()
        if self._flat_obs:
            obs = self.flatten_obs(obs)

        return obs

    def render_up(self):
        self._reset_up()
        imgs = self.get_images() # self.images_to_array(self.get_images())
        self._reset_down()
        return imgs

    def reset(self):
        if self.sim:
            self._robot.reset()

        self._reset_down()

        # fix default angle at first joint reset
        if self._episode_count == 0:
            self._default_angle = self._robot.get_ee_angle()

        # take actual observation and return
        obs = self.get_observation()
        if self._flat_obs:
            obs = self.flatten_obs(obs)

        self._curr_path_length = 0
        self._episode_count += 1

        return obs

    def _format_action(self, action):
        """Returns [x,y,z], [yaw, pitch, roll], close_gripper"""
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 2:
            delta_pos, delta_angle, gripper = (
                np.append(action[:2], 0.0),
                default_delta_angle,
                action[-1],
            )
        elif self.DoF == 3:
            delta_pos, delta_angle, gripper = (
                action[:-1],
                default_delta_angle,
                action[-1],
            )
        elif self.DoF == 4:
            delta_pos, delta_angle, gripper = (
                action[:3],
                [default_delta_angle[0], default_delta_angle[1], action[3]],
                action[-1],
            )
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
        lin_vel, rot_vel = (
            lin_vel * self.max_lin_vel / self.hz,
            rot_vel * self.max_rot_vel / self.hz,
        )
        return lin_vel, rot_vel

    def _get_valid_pos_and_gripper(self, pos, gripper):
        """To avoid situations where robot can break the object / burn out joints,
        allowing us to specify (x, y, z, gripper) where the robot cannot enter. Gripper is included
        because (x, y, z) is different when gripper is open/closed.

        There are two ways to do this: (a) reject action and maintain current pose or (b) project back
        to valid space. Rejecting action works, but it might get stuck inside the box if no action can
        take it outside. Projection is a hard problem, as it is a non-convex set :(, but we can follow
        some rough heuristics."""

        # clip commanded position to satisfy box constraints
        x_low, y_low, z_low = self.ee_space.low[:3]
        x_high, y_high, z_high = self.ee_space.high[:3]
        pos[0] = pos[0].clip(x_low, x_high)  # new x
        pos[1] = pos[1].clip(y_low, y_high)  # new y
        pos[2] = pos[2].clip(z_low, z_high)  # new z

        return pos, gripper

    def _update_robot(self, pos, angle, gripper):
        """input: the commanded position (clipped before).
        feasible position (based on forward kinematics) is tracked and used for updating,
        but the real position is used in observation."""
        feasible_pos, feasible_angle = self._robot.update_pose(pos, angle)
        if self.use_gripper:
            self._robot.update_gripper(gripper)

    @property
    def _curr_pos(self):
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        return self._robot.get_ee_angle()

    def get_images(self):
        camera_feed = []
        if self.sim and self._has_offscreen_renderer:
            camera_feed.extend(self._robot.render(mode="rgb_array"))
        if not self.sim:
            camera_feed.extend(self._camera_reader.read_cameras())
        return camera_feed

    def get_state(self):
        state_dict = {}
        if self.use_gripper:
            gripper_state = self._robot.get_gripper_state()
        else:
            gripper_state = 0

        state_dict["control_key"] = "current_pose"

        if self.sim:
            gripper_state = np.array(0)
        state_dict["current_pose"] = np.concatenate(
            [self._robot.get_ee_pos(), self._robot.get_ee_angle(), [gripper_state]]
        )

        state_dict["joint_positions"] = self._robot.get_joint_positions()
        state_dict["joint_velocities"] = self._robot.get_joint_velocities()
        # don't track gripper velocity
        state_dict["gripper_velocity"] = 0

        return state_dict

    def _randomize_reset_pos(self):
        """takes random action along x-y plane, no change to z-axis / gripper"""
        random_xy = np.random.uniform(-0.5, 0.5, (2,))
        random_z = np.random.uniform(-0.2, 0.2, (1,))
        if self.DoF == 4:
            random_rot = np.random.uniform(-0.5, 0.0, (1,))
            act_delta = np.concatenate(
                [random_xy, random_z, random_rot, np.zeros((1,))]
            )
        else:
            act_delta = np.concatenate([random_xy, random_z, np.zeros((1,))])
        for _ in range(10):
            self.step(act_delta)

    def get_observation(self):
        # get state and images
        current_state = self.get_state()
        # set gripper width
        gripper_width = current_state["current_pose"][-1:]
        # compute and normalize ee/qpos state
        if self.DoF == 2:
            ee_pos = np.concatenate([current_state["current_pose"][:2], gripper_width])
        elif self.DoF == 3:
            ee_pos = np.concatenate([current_state["current_pose"][:3], gripper_width])
        elif self.DoF == 4:
            ee_pos = np.concatenate(
                [
                    current_state["current_pose"][:3],
                    current_state["current_pose"][5:6],
                    gripper_width,
                ]
            )
        qpos = np.concatenate([current_state["joint_positions"], gripper_width])
        normalized_ee_pos = self.normalize_ee_obs(ee_pos)
        normalized_qpos = self.normalize_qpos(qpos)

        sphere_qpos, sphere_qvel = None, None
        if self.sim:
            sphere_qpos = self.normalize_ee_obs(self._robot.get_sphere_joint_position())
            sphere_qvel = self.normalize_ee_obs(
                self._robot.get_sphere_joint_velocities()
            )

        if self._side_camera or self._front_camera:
            current_images = self.get_images()
            imgs = self.images_to_array(current_images)
        else:
            current_images = []

        obs_dict = {
            "front_camera_obs": imgs[0][..., : 4 if self._depth_camera else 3] if len(current_images) else [],
            "side_camera_obs": imgs[1][..., : 4 if self._depth_camera else 3] if len(current_images) else [],
            "lowdim_ee": normalized_ee_pos,
            "lowdim_sphere": sphere_qpos,
            "lowdim_sphere_vel": sphere_qvel,
            "lowdim_qpos": normalized_qpos,
            "full_state": self._robot.get_state() if self._full_state else None,
        }

        if not self._front_camera:
            obs_dict.pop("front_camera_obs", None)
        if not self._side_camera:
            obs_dict.pop("side_camera_obs", None)
        if not self._qpos:
            obs_dict.pop("lowdim_qpos", None)
        if not self._ee_pos:
            obs_dict.pop("lowdim_ee", None)
        if not self._sphere_vel:
            obs_dict.pop("lowdim_sphere_vel", None)
        if not self._sphere_pos:
            obs_dict.pop("lowdim_sphere", None)
        if not self._full_state:
            obs_dict.pop("full_state", None)
        return obs_dict

    def render(self, mode="rgb_array"):
        if mode == "rgb_array" and self._has_offscreen_renderer:
            return self.get_images()[0]["color_image"]
        elif mode == "human" and self.sim and self._has_renderer:
            self._robot.render(mode="human")
        else:
            self._robot.render(mode="human")
            
    def images_to_array(self, imgs_list):
        # stack images
        imgs = []
        
        if len(imgs_list) >= 4:
            imgs_list[0].update(imgs_list[1])
            imgs_list[2].update(imgs_list[3])
            imgs_list = [imgs_list[0], imgs_list[2]]

        for img in imgs_list:
            img_tmp = img["color_image"]
            if "depth_aligned" in img.keys():
                depth = img["depth_aligned"]
                if len(depth.shape) == 2:
                    depth = depth[..., np.newaxis]
                img_tmp = np.concatenate((img_tmp, depth), axis=-1)
                
            imgs += [img_tmp]
               

        return imgs

    def is_robot_reset(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self._reset_joint_qpos)
        return joint_dist < epsilon

    def is_robot_init(self, epsilon=0.1):
        curr_joints = self._robot.get_joint_positions()
        joint_dist = np.linalg.norm(curr_joints - self._up_joint_qpos)
        return joint_dist < epsilon

    @property
    def num_cameras(self):
        return len(self.get_images())

    def compute_reward(self, action):
        if self.sim:
            return self._robot.compute_reward(action=action)
        return 0.0, False

    def init_demo(self, batch_size=1):
        pass

    def reset_demo(self):
        pass

    def get_action_list(self):
        action_lists = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([-1, 0]),
            np.array([1, 1]),
            np.array([1, -1]),
            np.array([-1, 1]),
            np.array([-1, -1]),
        ]
        return np.asarray(action_lists)

    def convert_act_2_idx(self, action):
        action_lists = self.get_action_list()

        dists = np.linalg.norm(action_lists - action, axis=-1)
        best_idx = dists.argmin()

        return np.array([int(best_idx)])

    def convert_idx_2_act(self, idx):
        action_lists = self.get_action_list()

        return action_lists[idx]

    def get_demo_action(self, obs):
        assert self.DoF == 2, "Only 2 DoF is supported for now"

        # [('lowdim_ee', (3,)), ('lowdim_sphere', (3,)), ('lowdim_sphere_vel', (3,)), ('lowdim_qpos', (8,))]
        ee_pos = obs[: self.DoF]
        sphere_pos = obs[3 : 3 + self.DoF]

        action = np.sign(sphere_pos - ee_pos)  # * action

        # wall x_pos (0.75) - 4 * sphere_radius (0.02) to avoid bounce out of bounds
        if sphere_pos[0] > 0.67 or sphere_pos[0] < 0.25:
            action[0] = action[0] * -1
        if sphere_pos[1] > 0.28 or sphere_pos[1] < -0.28:
            action[1] = action[1] * -1

        action_idx = self.convert_act_2_idx(action)
        action = self.convert_idx_2_act(action_idx)
        return action

    def get_stupid_demo_action(self, obs, t):
        if t < 3:
            action = np.array([1, 0])
        else:
            action = np.array([-1, 1])
        action_idx = self.convert_act_2_idx(action)
        action = self.convert_idx_2_act(action_idx)
        
        return action

        assert self.DoF == 2, "Only 2 DoF is supported for now"

        # [('lowdim_ee', (3,)), ('lowdim_sphere', (3,)), ('lowdim_sphere_vel', (3,)), ('lowdim_qpos', (8,))]
        ee_pos = obs[: self.DoF]
        sphere_pos = obs[3 : 3 + self.DoF]

        action = np.sign(sphere_pos - ee_pos)  # * action

        # wall x_pos (0.75) - 4 * sphere_radius (0.02) to avoid bounce out of bounds
        if sphere_pos[0] > 0.67 or sphere_pos[0] < 0.25:
            action[0] = action[0] * -1
        if sphere_pos[1] > 0.28 or sphere_pos[1] < -0.28:
            action[1] = action[1] * -1

        action_idx = self.convert_act_2_idx(action)
        action = self.convert_idx_2_act(action_idx)
        return action
