import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.path.append('/home/weirdlab/Downloads/reaching/environments/franka')
import numpy as np
import math
from gym import utils
from gym import spaces
import mujoco
from collections import OrderedDict
import gym
from real_robot_ik.robot_ik_solver import RobotIKSolver
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import glfw
import time
import mujoco_viewer
from copy import deepcopy


from transformations import euler_to_quat, quat_to_euler
from rewards import tolerance


class FrankaMujocoEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        max_episode_length=50,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_depth=True,
        img_height=480,
        img_width=480,
        camera_names=["front", "left"],
        control_hz=10,
        DoF=2,
        task_reward=False,
        safety_reward=False,
        reward="reach",
        goal=None,
        init_qpos=None,
        init_pos_noise=0.0,  # sphere position noise
        domain_randomization=False,
        param_keys=[],  # ["mass", "stiffness", "damping", "rolling_friction"],
        seed=0,
        xml_path=None,
    ):
        # rendering
        assert (
            has_renderer and has_offscreen_renderer
        ) is False, "both has_renderer and has_offscreen_renderer not supported"
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.viewer = None
        self.use_depth = use_depth
        self.img_width = img_width
        self.img_height = img_height
        self.camera_names = camera_names

        # timestep
        self.max_episode_length = max_episode_length
        self.frame_skip = control_hz * 10

        self.reward = reward
        self.goal_pos = goal

        self.task_reward = task_reward
        self.safety_reward = safety_reward

        # setup model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model_backup = deepcopy(self.model)
        self.data = mujoco.MjData(self.model)

        # get ids
        self.sphere_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "sphere_body"
        )
        self.sphere_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere_geom"
        )
        self.sphere_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "sphere_freejoint"
        )
        self.sphere_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "sphere_site"
        )

        self.structure_geom_ids = []
        for i in range(4):
            self.structure_geom_ids += [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"wall{i+1}")
            ]
        for i in range(2):
            self.structure_geom_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle{i+1}"
                )
            ]

        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_geom"
        )
        self.goal_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_freejoint"
        )
        self.goal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_body"
        )

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "EEF")
        self.franka_joint_ids = []
        for i in range(7):
            self.franka_joint_ids += [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot:joint{i+1}"
                )
            ]

        # setup IK
        self.DoF = DoF
        self.ik = RobotIKSolver(robot=self, control_hz=control_hz)

        # set robot init position
        self.obs_noise = None
        if init_qpos is not None:
            self.update_joints(init_qpos)
        self.init_qpos = self.get_joint_positions().copy()
        self.init_ee_pos = self.get_ee_pos().copy()
        self.init_ee_angle = self.get_ee_angle().copy()

        # domain randomization
        self.act_drop = None
        self._last_pos = self.init_ee_pos
        self._last_angle = self.init_ee_angle
        self.domain_randomization = domain_randomization

        # sphere
        self.init_pos_noise = init_pos_noise
        self.init_sphere_qpos = self.get_sphere_joint_position()
        self.curr_sphere_qpos = None

        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=restitution#restitution

        # https://github.com/deepmind/dm_control/blob/330c91f41a21eacadcf8316f0a071327e3f5c017/dm_control/locomotion/soccer/soccer_ball.py#L32
        # FIFA regulation parameters for a size 5 ball.
        # _REGULATION_RADIUS = 0.117  # Meters.
        # _REGULATION_MASS = 0.45  # Kilograms.

        # _DEFAULT_FRICTION = (0.7, 0.05, 0.04)  # (slide, spin, roll).
        # _DEFAULT_DAMP_RATIO = 0.4
        # friction="0.7 0.05 0.04" solref="0.02 1.0"
        # The first number is the sliding friction, acting along both axes of the tangent plane. The second number is the torsional friction, acting around the contact normal. The third number is the rolling friction, acting around both axes of the tangent plane.

        # task randomization
        self.reset_parameters()
        self.param_keys = param_keys
        if self.param_keys is not None:
            self.zeta_dict = {
                # key: [value, min, max]
                "mass": [None, 0.03, 0.06],
                "sliding_friction": [None, 0.1, 0.4],
                "rolling_friction": [None, 5e-6, 2e-4],
                "stiffness": [None, -1200, -700],
                "damping": [None, -5.0, -0.5],
                "radius": [None, 0.03, 0.04],
                # TODO -> use init_pos_noise
                "x_offset": [None, -0.1, 0.1],
                "y_offset": [None, -0.2, 0.2],
            }
            for key in self.zeta_dict.copy().keys():
                if key not in self.param_keys:
                    del self.zeta_dict[key]
        else:
            self.zeta_dict = {}

        if "free" not in xml_path:
            self.reset_task()

        # gym interface
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=[self.DoF], dtype=np.float32
        )
        self.action_space.flat_dim = len(self.action_space.low)
        observation = self._get_obs()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(observation),), dtype=observation.dtype
        )
        self.observation_space.flat_dim = len(self.observation_space.low)
        self.spec = self

        self.seed(seed)
        utils.EzPickle.__init__(self)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def update_gripper(self, gripper):
        pass

    def apply_action_drop(self, pos, angle):
        # with 50% chance use last action to simulate delay on real system
        if self.act_drop is not None and np.random.choice(
            [0, 1], p=[self.act_drop, 1 - self.act_drop]
        ):
            return self._last_pos, self._last_angle
        self._last_pos = pos.copy()
        self._last_angle = angle.copy()
        return self._last_pos, self._last_angle

    def update_pose(self, pos, angle):
        pos, angle = self.apply_action_drop(pos, angle)
        desired_qpos, success = self.ik.compute(pos, euler_to_quat(angle))

        self.data.ctrl[: len(desired_qpos)] = desired_qpos
        # advance simulation, use control callback to obtain external force and control.
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        if self.has_renderer:
            self.render()

        return self.get_ee_pos(), self.get_ee_angle()

    def update_joints(self, qpos):
        self.data.qpos[: len(qpos)] = qpos
        # forward dynamics: same as mj_step but do not integrate in time.
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        # angle_euler = R.from_quat(action[3:][[3, 0, 1, 2]]).as_euler('xyz', degrees=False)

        delta_ee_pos = np.zeros(3)
        delta_ee_angle = np.zeros(3)

        curr_ee_pos = self.get_ee_pos()

        # fix roll, pitch, yaw
        desired_ee_angle = self.init_ee_angle.copy()

        if self.DoF == 2:
            delta_ee_pos[:2] = action[:2]
            desired_ee_pos = curr_ee_pos + delta_ee_pos
            # fix z
            desired_ee_pos[2] = 0.12  # self.init_ee_pos[2].copy()

        if self.DoF == 3:
            delta_ee_pos[:3] = action[:3]
            desired_ee_pos = curr_ee_pos + delta_ee_pos

        if self.DoF == 4:
            delta_ee_pos[:3] = action[:3]
            desired_ee_pos = curr_ee_pos + delta_ee_pos
            desired_ee_angle[3] += action[3]

        # desired_ee_angle = self.get_ee_angle() + delta_ee_angle
        ee_pos, ee_angle = self.update_pose(desired_ee_pos, desired_ee_angle)

        next_obs = self._get_obs()

        reward = -np.linalg.norm(np.array([0.3, 0.3]) - ee_pos[:2])

        done = bool(reward < 0.1)

        return next_obs, reward, done, {}

    def _get_obs(self):
        # not being used when going through RobotEnv
        return np.concatenate([self.get_qpos(), self.get_qvel()], dtype=np.float32)

    def _value_to_range(self, value, r_max, r_min, t_max, t_min):
        """scales value in range [r_max, r_min] to range [t_max, t_min]"""
        return (value - r_min) / (r_max - r_min) * (t_max - t_min) + t_min

    def set_sphere_pos(self, qpos):
        self.init_sphere_qpos = qpos.copy()
        self.update_sphere(qpos)

    def set_sphere_radius(self, radius):
        self.model.geom_size[self.sphere_geom_id][0] = radius
        mujoco.mj_resetData(self.model, self.data)

    def update_sphere(self, qpos):
        self.data.qpos[self.sphere_joint_id : self.sphere_joint_id + 3] = qpos
        mujoco.mj_forward(self.model, self.data)

    def update_goal(self, qpos):
        self.model.body_pos[self.goal_body_id] = qpos
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *args, **kwargs):
        if self.domain_randomization:
            self.reset_domain()

        # modify sphere parameters
        self.reset_task()

        # propagate model changes to data (active sim)
        mujoco.mj_resetData(self.model, self.data)

        if self.curr_sphere_qpos is None:
            sphere_qpos = self.init_sphere_qpos.copy()
        else:
            sphere_qpos = self.curr_sphere_qpos.copy()
        self.update_sphere(sphere_qpos)

        # TODO update joints to upright position and render to video!

        if self.goal_pos:
            self.update_goal(self.goal_pos)

        # runs mujoco.mj_forward(self.model, self.data)
        self.update_joints(self.init_qpos)

        # mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        return obs  # , {}

    def viewer_setup(self):
        if self.has_renderer:
            return mujoco_viewer.MujocoViewer(
                self.model, self.data, height=720, width=720
            )
        if self.has_offscreen_renderer:
            return mujoco.Renderer(
                self.model, width=self.img_width, height=self.img_height
            )

    def render(self, mode="rgb_array"):
        assert mode in ["rgb_array", "human"], "mode not in ['rgb_array', 'human']"
        assert (
            self.has_renderer or self.has_offscreen_renderer
        ), "no renderer available."

        imgs = []

        if not self.viewer:
            self.viewer = self.viewer_setup()

        if self.has_renderer and mode == "human":
            self.viewer.render()
        elif self.has_offscreen_renderer and mode == "rgb_array":
            
            for camera in self.camera_names:
            
                img_dict = {}
                self.viewer.update_scene(self.data, camera=camera)
                img_dict["color_image"] = self.viewer.render().copy()
                if self.use_depth:
                    self.viewer.enable_depth_rendering()
                    img_dict["depth_aligned"] = self.viewer.render().copy()
                    self.viewer.disable_depth_rendering()
                imgs += [img_dict.copy()]

        return imgs

    def get_camera_intrinsic(self, camera_name):
        """
        Obtains camera intrinsic matrix.

        Args:
            camera_name (str): name of camera
        Return:
            K (np.array): 3x3 camera matrix
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self.model.cam_fovy[cam_id]

        # Compute intrinsic parameters
        fy = self.img_height / (2 * np.tan(np.radians(fovy / 2)))
        fx = fy
        cx = self.img_width / 2
        cy = self.img_height / 2

        # Camera intrinsic matrix
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return K

    def get_camera_extrinsic(self, camera_name):
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame. MuJoCo has a weird convention for how it sets up the
        camera body axis, so we also apply a correction so that the x and y
        axis are along the camera view and the z axis points along the
        viewpoint.
        Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

        https://github.com/ARISE-Initiative/robosuite/blob/de64fa5935f9f30ce01b36a3ef1a3242060b9cdb/robosuite/utils/camera_utils.py#L39

        Args:
            sim (MjSim): simulator instance
            camera_name (str): name of camera
        Return:
            R (np.array): 4x4 camera extrinsic matrix
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        camera_pos = self.data.cam_xpos[cam_id]
        camera_rot = self.data.cam_xmat[cam_id].reshape(3, 3)

        R = np.eye(4)
        R[:3, :3] = camera_rot
        R[:3, 3] = camera_pos

        camera_axis_correction = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        R = R @ camera_axis_correction

        return R

    def seed(self, seed=0):
        np.random.seed(seed)

    def apply_obs_noise(self, obs):
        if self.obs_noise is not None:
            obs += np.random.normal(loc=0.0, scale=self.obs_noise, size=obs.shape)
        return obs

    def get_state(self):
        return np.concatenate([self.get_qpos(), self.get_qvel()])
    
    def set_state(self, state):
        qpos = state[: self.model.nq]
        qvel = state[self.model.nq :]
        self.set_qpos(qpos)
        self.set_qvel(qvel)

    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def set_qpos(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
    
    def set_qvel(self, qvel):
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def get_sphere_joint_position(self):
        sphere_pos = self.data.qpos[
            self.sphere_joint_id : self.sphere_joint_id + 3
        ].copy()
        return sphere_pos

    def get_sphere_joint_velocities(self):
        sphere_vel = self.data.qvel[
            self.sphere_joint_id : self.sphere_joint_id + 3
        ].copy()
        return sphere_vel

    def get_joint_positions(self):
        qpos = self.data.qpos[self.franka_joint_ids].copy()
        return self.apply_obs_noise(qpos)

    def get_joint_velocities(self):
        qvel = self.data.qvel[self.franka_joint_ids].copy()
        return self.apply_obs_noise(qvel)

    def get_ee_pose(self):
        return self.get_ee_pos(), self.get_ee_angle(quat=True)

    def get_ee_pos(self):
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        return self.apply_obs_noise(ee_pos)

    def get_ee_angle(self, quat=False):
        ee_mat = self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3)
        ee_mat = self.apply_obs_noise(ee_mat)

        ee_angle = R.from_matrix(ee_mat)
        ee_angle = ee_angle.as_euler("xyz")
        if quat:
            return euler_to_quat(ee_angle)
        else:
            return ee_angle

    def get_gripper_state(self):
        return np.array(0)

    def reset_task(self, task=None):
        for key in self.param_keys:
            min_value, max_value = self.zeta_dict[key][1], self.zeta_dict[key][2]

            # if parameters are set from outside, don't draw random set
            if not self.params_set:
                value = np.random.uniform(low=min_value, high=max_value)
                value = np.around(value, decimals=2)
            else:
                value = self.zeta_dict[key][0]
            if key == "mass":
                # https://mujoco.readthedocs.io/en/stable/XMLreference.html?highlight=geom#body-geom
                # mass: real, optional
                # If this attribute is specified, the density attribute below is ignored and the geom
                # density is computed from the given mass, using the geom shape and the assumption of
                # uniform density. The computed density is then used to obtain the geom inertia. Recall
                # that the geom mass and inertia are only used during compilation, to infer the body
                # mass and inertia if necessary. At runtime only the body inertial properties affect
                # the simulation; the geom mass and inertia are not even saved in mjModel.

                # https://github.com/deepmind/mujoco/issues/764#issuecomment-1463193638
                value_tmp = self.model.body_mass[self.sphere_body_id].copy()
                self.model.body_mass[self.sphere_body_id] = value
                self.model.body_inertia[self.sphere_body_id] = value / value_tmp
            elif key == "sliding_friction":
                self.model.geom_friction[self.sphere_geom_id][0] = value
            elif key == "rolling_friction":
                self.model.geom_friction[self.sphere_geom_id][2] = value
            elif key == "stiffness":
                self.model.geom_solref[self.sphere_geom_id][0] = value
                self.model.geom_solref[self.structure_geom_ids][0] = value
            elif key == "damping":
                self.model.geom_solref[self.sphere_geom_id][1] = value
                self.model.geom_solref[self.structure_geom_ids][1] = value
            elif key == "radius":
                self.model.geom_size[self.sphere_geom_id][0] = value

            self.zeta_dict[key][0] = value

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)

        # check env has sphere
        if self.sphere_geom_id != -1:
            # perturb sphere position
            qpos = self.init_sphere_qpos.copy()
            if self.init_pos_noise:
                qpos[0] += np.random.uniform(
                    -0.1, 0.1, size=1
                )
                qpos[1] += np.random.uniform(
                    -0.2, 0.2, size=1
                )

            if "x_offset" in self.param_keys and self.params_set:
                qpos[0] += self.zeta_dict[key][0]
            if "y_offset" in self.param_keys and self.params_set:
                qpos[1] += self.zeta_dict[key][0]
            # set height = radius to avoid bouncing behavior
            qpos[2] = self.model.geom_size[self.sphere_geom_id][0]

            self.curr_sphere_qpos = qpos.copy()
        # self.data.qpos[self.sphere_joint_id+2] = self.model.geom_size[self.sphere_geom_id][0]
        # friction_static, friction_dynamic, slip_threshold
        # coefficient of static friction is the maximum amount of friction that can be generated between two objects that are not moving relative to each other
        # coefficient of dynamic friction is the amount of friction that is generated between two objects that are moving relative to each other
        # slip_threshold is the threshold velocity at which static friction transitions to dynamic friction

    def reset_domain(self):
        # reset mujoco model because we apply multiplicative noise
        self.model = deepcopy(self.model_backup)
        # reset viewer because viewer = f(model)
        del self.viewer
        self.viewer = self.viewer_setup()

        # Reinforcement and imitation learning for diverse visuomotor skills. Zhu et al. 2018. Appendix A
        self.act_drop = 0.3  # 0.5
        self.obs_noise = 0.01

        # Reinforcement and Imitation Learning for Diverse Visuomotor Skills
        # lighting
        # sphere + floor/background + wall color

        self.model.light_pos += np.random.normal(0.0, 0.2, size=3)
        self.model.light_dir += np.random.normal(0.0, 0.2, size=3)

        self.model.light_castshadow = np.random.choice([0, 1])

        self.model.light_ambient += np.random.normal(0.0, 0.2, size=3)
        self.model.light_diffuse += np.random.normal(0.0, 0.2, size=3)
        self.model.light_specular += np.random.normal(0.0, 0.2, size=3)

        self.model.geom_rgba[:, :3] *= np.random.uniform(
            0.8, 1.2, (self.model.geom_rgba.shape[0], 3)
        )

        # DeXtreme: Transfer of Agile In-Hand Manipulation from Simulation to Reality
        # mass, scale, friction, armature, effort, , joint damping, restitution

        self.franka_dof_ids = self.model.dof_jntid[self.franka_joint_ids]

        size = len(self.franka_dof_ids)
        # joint damping
        self.model.dof_damping[self.franka_dof_ids] *= np.exp(
            np.random.uniform(0.8, 1.2, size)
        )

        # joint stiffness
        self.model.dof_frictionloss[self.franka_dof_ids] *= np.random.uniform(
            0.8, 1.2, size
        )

        # joint armature
        self.model.dof_armature[self.franka_dof_ids] *= np.random.uniform(
            0.8, 1.2, size
        )

        # joint mass / diag inertia
        self.model.dof_M0[self.franka_dof_ids] *= np.random.uniform(0.8, 1.2, size)

        # joint stiffness + damping + restitution (?)
        # self.model.dof_solimp *= np.random.uniform(0.8, 1.2, size)
        # self.model.dof_solref[self.franka_dof_ids,1] *= np.exp(np.random.uniform(0.3, 3.0, size))

        mujoco.mj_resetData(self.model, self.data)

    @property
    def parameter_dim(self):
        return len(self.get_parameters())

    def reset_parameters(self):
        self.params_set = False

    def get_parameters(self):
        params = []
        for key, value in self.zeta_dict.items():
            # print("get zeta_dict", key, value)
            params += [self._value_to_range(value[0], value[2], value[1], 1.0, -1.0)]
        # print("get params", params)
        return np.array(params)

    def set_parameters(self, params):

        # print("set params", params)
        params = np.around(params, decimals=2)
        # print("set round params", params)
        for i, (key, value) in enumerate(self.zeta_dict.items()):
            self.zeta_dict[key][0] = self._value_to_range(
                params[i], 1.0, -1.0, value[2], value[1]
            )
            # print("set zeta_dict", self.zeta_dict[key][0])
        # self.zeta = self._value_to_range(params, 1., -1., self.zeta_max, self.zeta_min)
        self.params_set = True
        self.reset_task()
        return self.get_parameters()

    def seed(self, seed=0):
        np.random.seed(seed)
        return super().seed(seed)

    def compute_reward(self, action=None):
        if self.task_reward:

            if self.reward == "reach":
                
                ee_pos = self.get_ee_pos()
                goal_pos = np.array(self.goal_pos)

                # reaching reward
                ee_to_goal = np.linalg.norm(ee_pos - goal_pos)
                reach_reward = - ee_to_goal

                # success reward
                success_reward = 0.
                success = ee_to_goal < 5e-2
                if success:
                    success_reward = 1.

                reward = reach_reward + success_reward
                return reward, success
            
            elif self.reward == "strike":

                ee_pos = self.get_ee_pos()
                sphere_pos = self.get_sphere_joint_position()
                sphere_vel = self.get_sphere_joint_velocities()
                goal_pos = np.array(self.goal_pos)
                
                ee_to_sphere = np.linalg.norm(ee_pos[:2] - sphere_pos[:2])
                sphere_to_goal = np.linalg.norm(sphere_pos[:2] - goal_pos[:2])
                
                # reaching reward
                reach_reward = 0.
                if np.any(np.abs(sphere_vel) > 1e-2):    
                    reach_reward = - 0.1 * ee_to_sphere
                # ee_to_sphere = np.linalg.norm(ee_pos[:2] - sphere_pos[:2])
                # reach_reward = - 0.1 * ee_to_sphere
                
                # strike reward
                strike_reward = 0.
                if ee_to_sphere < 2e-5:
                    strike_reward += np.sum(np.abs(action))

                # distance to goal reward
                goal_reward = - 5. * sphere_to_goal**2

                # success reward
                success_reward = 0.
                success = sphere_to_goal < 5e-2
                if success:
                    success_reward = 1.

                reward = reach_reward + goal_reward + success_reward
                return reward, success
            
        elif self.safety_reward:
            safety_reward = -np.linalg.norm(np.array(self.goal_pos) - self.get_sphere_joint_position())
            return safety_reward, False
        else:
            return 0.0, False
