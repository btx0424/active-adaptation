import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from typing import Dict
import random

# env related
from envs.base_task import BaseTask

# utils
from utils.terrain import Terrain
from utils.math import quat_apply_yaw, wrap_to_pi, get_scale_shift
from utils.helpers import class_to_dict
import torchvision
import cv2

# config
from configs import LeggedRobotCfg
from global_config import ROOT_DIR


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

def copysign_new(a, b):

    a = torch.tensor(a, device=b.device, dtype=torch.float)
    a = a.expand_as(b)
    return torch.abs(a) * torch.sign(b)

def get_euler_rpy(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[..., qw] * q[..., qx] + q[..., qy] * q[..., qz])
    cosr_cosp = q[..., qw] * q[..., qw] - q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] + q[..., qz] * q[..., qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[..., qw] * q[..., qy] - q[..., qz] * q[..., qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign_new(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[..., qw] * q[..., qz] + q[..., qx] * q[..., qy])
    cosy_cosp = q[..., qw] * q[..., qw] + q[..., qx] * \
        q[..., qx] - q[..., qy] * q[..., qy] - q[..., qz] * q[..., qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def get_euler_rpy_tensor(quat):
    r, p, w = get_euler_rpy(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=-1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz

class LeggedRobot2Leg_Tinker_Gait(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True #doghome show
        self.init_done = False
    
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.resize_transform = torchvision.transforms.Resize((self.cfg.depth.resized[1], self.cfg.depth.resized[0]), 
                                                              interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_cost_function()
        self.init_done = True
        self.global_counter = 0

        # self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.post_physics_step()

    #------------ enviorment core ----------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
       
        """
        print("Init buffer---------------------")#cushihua
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_rpy_tensor(self.feet_quat)

        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_feet_z = torch.zeros((self.num_envs, 2), device=self.device)#0.05
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_rpy_tensor(self.feet_quat)
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 2, 6) # for feet only, see create_env()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.action_avg = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)#doghome

        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.leg_phase = torch.zeros((self.num_envs, 2), device=self.device)
        self.target_height= torch.zeros((self.num_envs, 2), device=self.device)
        self.feet_height_g= torch.zeros((self.num_envs, 2), device=self.device)

        str_rng = self.cfg.domain_rand.motor_strength_range
        kp_str_rng = self.cfg.domain_rand.kp_range
        kd_str_rng = self.cfg.domain_rand.kd_range

        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        self.kp_factor = (kp_str_rng[1] - kp_str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + kp_str_rng[0]
        self.kd_factor = (kd_str_rng[1] - kd_str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + kd_str_rng[0]

        if self.cfg.env.history_encoding:
             self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.num_dofs, device=self.device, dtype=torch.float)
        self.contact_buf = torch.zeros(self.num_envs, self.cfg.env.contact_buf_len, 2, device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time1 = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time2 = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time3 = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time4 = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts1 = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts2 = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts3 = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.base_height_points = self._init_base_height_points()

        self.measured_heights = 0
        self.feet_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_st = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
       
        print("Dof Name list:")
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            print(name)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            angle = self.cfg.init_state.default_joint_angles_st[name]
            self.default_dof_pos_st[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        print("Default_dof_pos:",self.default_dof_pos)
        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.num_envs,  
                                            self.cfg.depth.buffer_len, 
                                            self.cfg.depth.resized[1], 
                                            self.cfg.depth.resized[0]).to(self.device)
            
        self.lag_buffer = torch.zeros(self.num_envs,self.cfg.domain_rand.lag_timesteps,self.num_actions,device=self.device,requires_grad=False)
        #----lag buffer
        if self.cfg.domain_rand.add_dof_lag:
            self.dof_lag_buffer = torch.zeros(self.num_envs,self.num_actions * 2,self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,device=self.device)
            if self.cfg.domain_rand.randomize_dof_lag_timesteps:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                    self.last_dof_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.dof_lag_timesteps_range[1]
            else:
                self.dof_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.dof_lag_timesteps_range[1]

        if self.cfg.domain_rand.add_imu_lag:
            self.imu_lag_buffer = torch.zeros(self.num_envs, 6, self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,device=self.device)
            if self.cfg.domain_rand.randomize_imu_lag_timesteps:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.imu_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                    self.last_imu_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
            else:
                self.imu_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.imu_lag_timesteps_range[1]
                
        if self.cfg.domain_rand.add_dof_pos_vel_lag:
            self.dof_pos_lag_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1,device=self.device)
            self.dof_vel_lag_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1,device=self.device)
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                    self.last_dof_pos_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]
            else:
                self.dof_pos_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0],
                                                        self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1, (self.num_envs,),device=self.device)
                if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                    self.last_dof_vel_lag_timestep = torch.ones(self.num_envs,device=self.device,dtype=int) * self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]
            else:
                self.dof_vel_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]

        if self.cfg.domain_rand.randomize_init_joint_offset:
            self.init_joint_offset = torch_rand_float(self.cfg.domain_rand.init_joint_offset[0], self.cfg.domain_rand.init_joint_offset[1], (self.num_envs, self.num_dofs), device=self.device)
        else:
            self.init_joint_offset = torch.zeros(self.num_envs,self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            
        if self.cfg.domain_rand.randomize_init_joint_scale:
            self.init_joint_scale = torch_rand_float(self.cfg.domain_rand.init_joint_scale[0], self.cfg.domain_rand.init_joint_scale[1],(self.num_envs, self.num_dofs), device=self.device)
        else:
            self.init_joint_scale = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,requires_grad=False)
         
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.init_randomize_props()#cushihua
        self.randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print("body_names:",body_names)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print("dof_names:",self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]

        print("feet_names:",feet_names)
        for s in ["L4_Link_ankle", "R4_Link_ankle"]:        
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        print("Creating env...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.attach_camera(i, env_handle, actor_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            friction_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].friction
            self.friction_coeffs_tensor = friction_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs_tensor = self.restitution_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            restitution_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].restitution
            self.restitution_coeffs_tensor = restitution_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [random.randint(0,self.cfg.domain_rand.lag_timesteps-1) for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False
        else:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [self.cfg.domain_rand.lag_timesteps-1 for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def init_randomize_props(self):
        ''' Initialize torch tensors for random properties
        '''
        if self.cfg.domain_rand.randomize_base_mass:
            self.payload_masses = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
            
        if self.cfg.domain_rand.randomize_com:
            self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                        requires_grad=False)
            
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False) 

        if self.cfg.domain_rand.randomize_motor_strength:
            self.torque_multi = torch_rand_float(self.cfg.domain_rand.motor_strength[0], self.cfg.domain_rand.motor_strength[1], (self.num_envs, self.num_actions), device=self.device)
        else:
            self.torque_multi = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,requires_grad=False)
        
        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
            self.randomized_d_gains = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        else:
            self.randomized_p_gains = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.randomized_d_gains = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
                        
        if self.cfg.domain_rand.add_action_lag:   
            self.action_lag_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.action_lag_timesteps_range[1]+1,device=self.device)
            self.action_lag_timestep = torch.randint(self.cfg.domain_rand.action_lag_timesteps_range[0],
                                                  self.cfg.domain_rand.action_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device) 
        else:
            self.action_lag_timestep = torch.ones(self.num_envs,device=self.device) * self.cfg.domain_rand.action_lag_timesteps_range[0]

        self.action_filterd = torch.zeros(self.num_envs, self.num_actions,
                                        dtype=torch.float,
                                        device=self.device, requires_grad=False)

    def randomize_rigid_body_props(self, env_ids):
        ''' Randomise some of the rigid body properties of the actor in the given environments, i.e.
            sample the mass, centre of mass position, friction and restitution.'''
  
    def randomize_dof_props(self, env_ids):
        # Randomise the motor strength:
 
        # rand motor position offset
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch_rand_float(min_offset, max_offset, (len(env_ids),self.num_actions), device=self.device)

    def refreshable_randomize_lag(self,env_ids):
        # action lag
        if self.cfg.domain_rand.add_action_lag:   
            self.action_lag_buffer[env_ids,:,:] = 0.0
            self.action_lag_timestep[env_ids] = torch.randint(self.cfg.domain_rand.action_lag_timesteps_range[0],
                                                       self.cfg.domain_rand.action_lag_timesteps_range[1]+1,
                                                       (len(env_ids),),device=self.device) 
 
    
    def refreshable_randomize_props(self,env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:
            self.torque_multi = torch_rand_float(self.cfg.domain_rand.motor_strength[0], self.cfg.domain_rand.motor_strength[1], (self.num_envs, self.num_actions), device=self.device)
     
        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
            self.randomized_d_gains = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)

        # if self.cfg.domain_rand.randomize_rfi:
        #     self.rfi_ep_offest = torch_rand_float(self.cfg.domain_rand.rfi_ep[0], self.cfg.domain_rand.rfi_ep[1],(self.num_envs, self.num_dofs), device=self.device)
        #     self.rfi_ep_offest *= self.torque_limits

    def reindex(self,tensor):#shunxu
        #sim2real purpose
        return tensor[:,[0,1,2,3,4,5,6,7,8,9]]
    
    def reindex_feet(self,tensor):
        return tensor[:,[0,1]]

    def exp_avg_filter(self, x, avg, alpha=0.8):
        """
        Simple exponential average filter
        """
        avg = alpha*x + (1-alpha)*avg
        return avg
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)#历史指令 直接存储原始值

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) #doghome bug
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            actions_scaled = self.actions * self.cfg.control.action_scale
 
            if self.cfg.domain_rand.add_action_lag:
                self.action_lag_buffer[:,:,1:] = self.action_lag_buffer[:,:,:self.cfg.domain_rand.action_lag_timesteps_range[1]].clone()
                self.action_lag_buffer[:,:,0] = actions_scaled.clone()
                lagged_actions_scaled = self.action_lag_buffer[torch.arange(self.num_envs),:,self.action_lag_timestep.long()]
            else:
                lagged_actions_scaled = actions_scaled
                
            if self.cfg.control.use_filter:
                self.action_filterd = self.exp_avg_filter(lagged_actions_scaled, self.action_filterd,self.cfg.control.exp_avg_decay) 
                self.torques = self._compute_torques(self.action_filterd).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(lagged_actions_scaled).view(self.torques.shape)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            if self.cfg.domain_rand.add_dof_lag:
                q = self.dof_pos
                dq = self.dof_vel
                self.dof_lag_buffer[:,:,1:] = self.dof_lag_buffer[:,:,:self.cfg.domain_rand.dof_lag_timesteps_range[1]].clone()
                self.dof_lag_buffer[:,:,0] = torch.cat((q, dq), 1).clone()
            if self.cfg.domain_rand.add_dof_pos_vel_lag:
                q = self.dof_pos
                self.dof_pos_lag_buffer[:,:,1:] = self.dof_pos_lag_buffer[:,:,:self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]].clone()
                self.dof_pos_lag_buffer[:,:,0] = q.clone()
                dq = self.dof_vel
                self.dof_vel_lag_buffer[:,:,1:] = self.dof_vel_lag_buffer[:,:,:self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]].clone()
                self.dof_vel_lag_buffer[:,:,0] = dq.clone()
            if self.cfg.domain_rand.add_imu_lag:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.base_quat[:] = self.root_states[:, 3:7]
                self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
                self.imu_lag_buffer[:,:,1:] = self.imu_lag_buffer[:,:,:self.cfg.domain_rand.imu_lag_timesteps_range[1]].clone()
                self.imu_lag_buffer[:,:,0] = torch.cat((self.base_ang_vel, self.base_euler_xyz ), 1).clone()
                
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.depth.use_camera and self.global_counter % self.cfg.depth.update_interval == 0:
            self.extras["depth"] = self.depth_buffer[:, -2]  # have already selected last one
        else:
            self.extras["depth"] = None
 
        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras
 
    def compute_observations(self):#guance
        sin_pos = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)
        diff = self.dof_pos - self.ref_dof_pos

               # random add dof_pos and dof_vel same lag
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :self.num_actions, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_lag_buffer[torch.arange(self.num_envs), -self.num_actions:, self.dof_lag_timestep.long()]  
        # random add dof_pos and dof_vel different lag
        elif self.cfg.domain_rand.add_dof_pos_vel_lag:
            if self.cfg.domain_rand.randomize_dof_pos_lag_timesteps_perstep:
                self.dof_pos_lag_timestep = torch.randint(self.cfg.domain_rand.dof_pos_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_pos_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_pos_lag_timestep > self.last_dof_pos_lag_timestep + 1
                self.dof_pos_lag_timestep[cond] = self.last_dof_pos_lag_timestep[cond] + 1
                self.last_dof_pos_lag_timestep = self.dof_pos_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_pos_lag_buffer[torch.arange(self.num_envs), :, self.dof_pos_lag_timestep.long()]
                
            if self.cfg.domain_rand.randomize_dof_vel_lag_timesteps_perstep:
                self.dof_vel_lag_timestep = torch.randint(self.cfg.domain_rand.dof_vel_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_vel_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_vel_lag_timestep > self.last_dof_vel_lag_timestep + 1
                self.dof_vel_lag_timestep[cond] = self.last_dof_vel_lag_timestep[cond] + 1
                self.last_dof_vel_lag_timestep = self.dof_vel_lag_timestep.clone()
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_vel_lag_timestep.long()]
        # dof_pos and dof_vel has no lag
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

                # imu lag, including rpy and omega
        if self.cfg.domain_rand.add_imu_lag:    
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.int()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz[:,-3:]

        obs_buf =torch.cat((self.lagged_base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            #self.lagged_base_euler_xyz,
                            self.commands[:, :3]* self.commands_scale,
                            self.reindex((self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
                            self.reindex(self.lagged_dof_vel * self.obs_scales.dof_vel),
                            self.actions,#self.action_history_buf[:,-1],
                            sin_pos,# 
                            cos_pos,# 
                            ),

                            dim=-1,
                            )#列表最后一项 [:-1]也就是上一次的

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.ang_vel * noise_level,
                               torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.zeros(3),# 
                               torch.ones(
                                   10) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   10) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               torch.zeros(2),
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

        # priv_latent = torch.cat((#

        #     self.base_ang_vel  * self.obs_scales.ang_vel,
        #     self.projected_gravity,
        #     self.commands[:, :3] * self.commands_scale,
        #     (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #     self.dof_vel * self.obs_scales.dof_vel,
        #     self.actions,
        #     sin_pos,
        #     cos_pos,
        #     self.base_lin_vel * self.obs_scales.lin_vel,
        #     self.randomized_lag_tensor,
        #     self.contact_mask,
        #     self.stance_mask,
        #     diff,
        #     self.mass_params_tensor,
        #     self.friction_coeffs_tensor,
        #     self.restitution_coeffs_tensor,
        #     self.motor_strength, 
        #     self.kp_factor,
        #     self.kd_factor), dim=-1)
    
        priv_latent = torch.cat((   self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    #self.base_euler_xyz,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_pos,
                                    cos_pos,
                                    self.friction_coeffs_tensor,
                                    self.contact_mask,
                                    self.stance_mask,
                                    diff,
                                    self.rand_push_force[:,:2]
                                    ),dim=-1)    
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            priv_latent = torch.cat([priv_latent,self.feet_heights.view(self.num_envs, -1)],dim=-1)
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        # update buffer 历史 压入 obs_buf
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

        if self.cfg.terrain.include_act_obs_pair_buf:#unuse
            # add to full observation history and action history to obs
            pure_obs_hist = self.obs_history_buf[:,:,:-self.num_actions].reshape(self.num_envs,-1)
            act_hist = self.action_history_buf.view(self.num_envs,-1)
            self.obs_buf = torch.cat([self.obs_buf,pure_obs_hist,act_hist], dim=-1)
    
    #------------- Callbacks --------------
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        #print(  self.projected_gravity[0])
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_euler_xyz = get_euler_rpy_tensor(self.feet_quat)

        #self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = self.contact_forces[:, self.feet_indices, 2] > 5
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_cost()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
        #     self._push_robots()

        self.update_depth_buffer()
        self.compute_observations()
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts = contact

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    #------------- Cameras --------------
    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg.depth.use_camera:
            config = self.cfg.depth
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.depth.original[0]
            camera_props.height = self.cfg.depth.original[1]
            camera_props.enable_tensors = True
            camera_horizontal_fov = self.cfg.depth.horizontal_fov
            camera_props.horizontal_fov = camera_horizontal_fov

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(config.position)
            camera_angle = np.random.uniform(config.angle[0],config.angle[1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return 
        # not meet the requirement of update
        if self.global_counter % self.cfg.depth.update_interval != 0:
            return 
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)
        
        self.gym.end_access_image_tensors(self.sim)

    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image

    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        self.phase = self._get_phase()
        self.compute_ref_state()
        self.stance_mask = self._get_gait_phase()
        self.contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        if self.cfg.commands.heading_command and self.cfg.lession.stop==False:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.) #航向模式下产生速度控制量hangxiang
            self.commands[:, 2] *= (torch.abs(self.commands[:, 4]) > self.cfg.rewards.stop_rate)
            if self.cfg.lession.stop:
                self.commands[:, :2] *=(torch.abs(self.commands[:, 4]) > self.cfg.rewards.stop_rate)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.feet_heights = self._get_feet_heights()
            
        if self.cfg.domain_rand.push_robots:
            self._push_robots_d()
    
       
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
     
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare friction randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]
     
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def _process_rigid_body_props(self, props, env_id):
     
        if self.cfg.domain_rand.randomize_base_mass:
            rng_mass = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass += rand_mass
        else:
            rand_mass = np.zeros((1, ))
        
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        mass_params = np.concatenate([rand_mass, rand_com])


        # randomize mass of all link
        if self.cfg.domain_rand.randomize_all_mass:
            for s in range(len(props)):
                rng = self.cfg.domain_rand.rd_mass_range
                rd_num = np.random.uniform(rng[0], rng[1])
                #self.mass_mask[env_id, s] = rd_num
                props[s].mass *= rd_num
        
        # randomize com of all link other than base link
        if self.cfg.domain_rand.randomize_com:
            for s in range(len(props)-1):
                
                rng = self.cfg.domain_rand.rd_com_range
                rd_num = np.random.uniform(rng[0], rng[1])
                #self.com_diff_x[env_id, s] = rd_num
                props[s].com.x += rd_num
                rd_num = np.random.uniform(rng[0], rng[1])
                #self.com_diff_y[env_id, s] = rd_num
                props[s].com.y += rd_num
                rd_num = np.random.uniform(rng[0], rng[1])
                #self.com_diff_z[env_id, s] = rd_num
                props[s].com.z += rd_num

        # randomize inertia of all body
        if self.cfg.domain_rand.random_inertia:
            rng = self.cfg.domain_rand.inertia_range
            for s in range(len(props)):
                rd_num = np.random.uniform(rng[0], rng[1])
                #self.inertia_mask_xx[env_id, s] = rd_num
                props[s].inertia.x.x *= rd_num
                
                rd_num = np.random.uniform(rng[0], rng[1])
                #self.inertia_mask_xy[env_id, s] = rd_num
                props[s].inertia.x.y *= rd_num
                props[s].inertia.y.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                #self.inertia_mask_xz[env_id, s] = rd_num
                props[s].inertia.x.z *= rd_num
                props[s].inertia.z.x *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                #self.inertia_mask_yy[env_id, s] = rd_num
                props[s].inertia.y.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                #self.inertia_mask_yz[env_id, s] = rd_num
                props[s].inertia.y.z *= rd_num
                props[s].inertia.z.y *= rd_num

                rd_num = np.random.uniform(rng[0], rng[1])
                #self.inertia_mask_zz[env_id, s] = rd_num
                props[s].inertia.z.z *= rd_num


        return props, mass_params
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # actions_scaled = actions * self.cfg.control.action_scale
 
        # if self.cfg.domain_rand.add_action_lag:
        #     self.action_lag_buffer[:,:,1:] = self.action_lag_buffer[:,:,:self.cfg.domain_rand.action_lag_timesteps_range[1]].clone()
        #     self.action_lag_buffer[:,:,0] = actions_scaled.clone()
        #     lagged_actions_scaled = self.action_lag_buffer[torch.arange(self.num_envs),:,self.action_lag_timestep.long()]
        # else:
        #     lagged_actions_scaled = actions_scaled
        
        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.p_gains*self.randomized_p_gains
            d_gains = self.d_gains*self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        torques = p_gains*(actions + self.default_dof_pos - self.dof_pos + self.motor_offsets) - d_gains*self.dof_vel

        if self.cfg.domain_rand.randomize_motor_strength:
            torques = torques*self.torque_multi
            
        # if self.cfg.domain_rand.randomize_rfi:
        #     torques += self.torque_limits*torch_rand_float(self.cfg.domain_rand.rfi_st[0], self.cfg.domain_rand.rfi_st[1],(self.num_envs, self.num_dofs), device=self.device) + self.rfi_ep_offest
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)   

    def check_termination(self):#jieshu  fuwei
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        #print(self.reset_buf[0])
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs

        # Termination for velocities, orientation, and low height
        self.reset_buf |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)

        self.reset_buf |= torch.any(
          torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 10., dim=1)

        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.4, dim=1)

        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.4, dim=1)
        
        # self.base_pos = self.root_states[:, 0:3]
        # self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.1, dim=1)


        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_cost(self):
        self.cost_buf[:] = 0
        for i in range(len(self.cost_functions)):
            name = self.cost_names[i]
            cost = self.cost_functions[i]() * self.dt #self.cost_scales[name]
            self.cost_buf[:,i] += cost
            self.cost_episode_sums[name] += cost
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.refreshable_randomize_props(env_ids)
        self.refreshable_randomize_lag(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            self.cost_episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # for i in range(len(self.lag_buffer)):
        #     self.lag_buffer[i][env_ids, :] = 0
        self.lag_buffer[env_ids,:,:] = 0
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs,_,_, _, _,_= self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * self.init_joint_scale[env_ids] + self.init_joint_offset[env_ids]
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # # base position
        # if self.custom_origins:
        #     self.root_states[env_ids] = self.base_init_state
        #     self.root_states[env_ids, :3] += self.env_origins[env_ids]
        #     if self.cfg.terrain.curriculum:
        #         platform = self.cfg.terrain.platform
        #         self.root_states[env_ids, :2] += torch_rand_float(-platform/3, platform/3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        #     else:
        #         terrain_length = self.cfg.terrain.terrain_length
        #         self.root_states[env_ids, :2] += torch_rand_float(-terrain_length/2, terrain_length/2, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        # else:
        #     self.root_states[env_ids] = self.base_init_state
        #     self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel doghome
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        if self.cfg.depth.use_camera:
            self.graphics_device_id = self.sim_device_id # required in headless mode
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        
    def _prepare_cost_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.cost_scales.keys()):
            scale = self.cost_scales[key]
            if scale==0:
                self.cost_scales.pop(key) 
            # else:
            #     self.cost_scales[key] *= self.dt

        self.cost_functions = []
        self.cost_names = []
        self.cost_k_values = []
        self.cost_d_values_tensor = []

        for name,scale in self.cost_scales.items():
            self.cost_names.append(name)
            name = '_cost_' + name
            print('cost name:',name)
            print('cost k value:',scale)
            self.cost_functions.append(getattr(self, name))
            self.cost_k_values.append(float(scale))

        for name,value in self.cost_d_values.items():
            print('cost name:',name)
            print('cost d value:',value)
            self.cost_d_values_tensor.append(float(value))

        self.cost_k_values = torch.FloatTensor(self.cost_k_values).view(1,-1).to(self.device)
        self.cost_d_values_tensor = torch.FloatTensor(self.cost_d_values_tensor).view(1,1,-1).to(self.device)

        self.cost_episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                  for name in self.cost_scales.keys()}

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.cost_scales = class_to_dict(self.cfg.costs.scales)
        self.cost_d_values = class_to_dict(self.cfg.costs.d_values)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        # global counter 是否该类似这个
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # if not self.terrain.cfg.measure_heights:
        #     return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(1):#self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        # draw depth image with window created by cv2
        if self.cfg.depth.use_camera:
            window_name = "Depth Image"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cup().numpy() + 0.5)
            cv2.waitKey(1) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height
    #----------------------phase---------------------
    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        #phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos > 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0

        return stance_mask

    def  _get_phase_single(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase_s = (self.episode_length_buf * self.dt / cycle_time) % 1
        return phase_s
 
    def compute_ref_state(self):
        #phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * self.phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        scale_3 = 0.8 * scale_1
        # left swing
        # 'J_L0':   0.0,   # [rad]
        # 'J_L1':  0.08,   # [rad]
        # 'J_L2':  0.56,   # [rad]
        # 'J_L3':  -1.12,   # [rad]
        # 'J_L4_ankle': -0.57,   # [rad]

        # 'J_R0':   0.0,   # [rad]
        # 'J_R1':  -0.08,   # [rad]
        # 'J_R2':  -0.56,   # [rad]
        # 'J_R3':  1.12,   # [rad]
        # 'J_R4_ankle': 0.57,   # [rad] 
        sin_pos_l[sin_pos_l > 0] = 0 #-left doghome urdf
        self.ref_dof_pos[:, 2] = sin_pos_l * -scale_1 #大腿 
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2 #小腿
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_3
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0#+  left
        self.ref_dof_pos[:, 7] = sin_pos_r * -scale_1 #大腿 
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2 #小腿
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_3

        self.ref_dof_pos[torch.abs(sin_pos) < 0.05] = 0.
        
        self.ref_action = 2 * self.ref_dof_pos
        
        self.ref_dof_pos += self.default_dof_pos

        phase_s = self._get_phase_single().unsqueeze(-1)  # 形状变为 (num_envs, 1)
        phase2 = torch.clamp(phase_s / 0.5, max=1.0)  # 当 phase_s 在 0 到 0.5 之间时，phase1 线性增加
        phase1 = torch.clamp((phase_s - 0.5) / 0.5, min=0.0)  # 当 phase_s 在 0.5 到 1 之间时，phase2 线性增加
        self.leg_phase[:, 0] = phase1.squeeze(-1)  # 更新 swing phase1 0～1
        self.leg_phase[:, 1] = phase2.squeeze(-1)  # 更新 swing phase2 0～1

    #------------ curriculum ----------------
    def _push_robots_d(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _resample_commands(self, env_ids):#zhiling
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.commands[env_ids, 4] = torch_rand_float(0,1, (len(env_ids), 1), device=self.device).squeeze(1)
        # set small commands to zero
        if self.cfg.lession.stop:
            self.commands[env_ids, :2] *=(torch.abs(self.commands[env_ids, 4]) > self.cfg.rewards.stop_rate).unsqueeze(1) 
        else:
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.rewards.command_dead).unsqueeze(1)
            if self.common_step_counter> 800*10000/300 and 0:
                self.commands[env_ids, :2] *=(torch.abs(self.commands[env_ids, 4]) > self.cfg.rewards.stop_rate).unsqueeze(1) 
            
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height

    #------------ reward functions----------------jiangli
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        # print("foot_pos:",foot_pos.size())
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_body_states[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        #print(stance_mask[0],contact[0])
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 550), dim=1)

    #{'J_hip_l_roll': 0, 'J_hip_l_yaw': 1, 'J_hip_l_pitch': 2, 'J_knee_l_pitch': 3, 'J_ankle_l_pitch': 4, 'J_ankle_l_roll': 5, 'J_hip_r_roll': 6, 'J_hip_r_yaw': 7, 'J_hip_r_pitch': 8, 'J_knee_r_pitch': 9, 'J_ankle_r_pitch': 10, 'J_ankle_r_roll': 11}
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, [0,1]]
        right_yaw_roll = joint_diff[:, [5,6]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_no_jump(self):
        # Penalize feet in the air at zero commands(static)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5
        double_contact = torch.sum(1.*contacts, dim=1)==2
        single_contact = torch.sum(1.*contacts, dim=1)==1
        no_contact = torch.sum(1.*contacts, dim=1)==0
        reward_out1= 1.* (single_contact) * (torch.norm(self.commands[:, :3], dim=1) >  self.cfg.rewards.command_dead)
        reward_out2= -2.* (no_contact) * (torch.norm(self.commands[:, :3], dim=1) >  self.cfg.rewards.command_dead)
        reward_out3= -0.5* (double_contact) * (torch.norm(self.commands[:, :3], dim=1) >  self.cfg.rewards.command_dead)
        return reward_out2#+reward_out2+reward_out3

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        
        stance_mask = self._get_gait_phase()
        # print("stance_mask_shape,feet_shape:",stance_mask.size(),self.rigid_body_states[:, self.feet_indices, 2].size())
        measured_heights = torch.sum(
            self.rigid_body_states[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # get new feet height in local frame
        #feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05
        # self.feet_height
        # delta_z = feet_z - self.last_feet_z
        # self.feet_height += delta_z
        # self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should larger than target feet height at the peak
        rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height)
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speedx(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 2.0
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward #* (self.commands[:, 0].abs() > 0.1)

    def _reward_low_speedy(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 1])
        absolute_command = torch.abs(self.commands[:, 1])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 1]) != torch.sign(self.commands[:, 1])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 1])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 2.0
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward #* (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_simple_feet_min_distance(self):
        # get local frame feet pos
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        # compute y distance
        y_dist = torch.abs(footpos_in_body_frame[:,0,1] - footpos_in_body_frame[:,1,1]) - 0.16 #0.20
        return torch.clamp(y_dist,max=0.0)
    
    def _reward_simple_knee_min_distance(self):
        cur_kneepos_translated = self.knee_pos - self.root_states[:, 0:3].unsqueeze(1)
        kneepos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            kneepos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_kneepos_translated[:, i, :])
        # compute y distance
        y_dist = torch.abs(kneepos_in_body_frame[:,0,1] - kneepos_in_body_frame[:,1,1]) - 0.16
        return torch.clamp(y_dist,max=0.0)
    
    def _reward_feet_rotation(self):
        rotation = torch.sum(torch.square(self.feet_euler_xyz[:,:,1]),dim=[1])
        # rotation = torch.sum(torch.square(feet_euler_xyz[:,:,1]),dim=1)
        r = torch.exp(-rotation*15)
        return r
    
    def _reward_feet_yaw_phase(self):
 
        # reward to align the yaw of the feet and the base of the swing feet        
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        #is_swing = (self.leg_phase > self.stance_ratio_fil.unsqueeze(1)*1.1) & (self.leg_phase<0.9)
        swing_mask = 1 - self._get_gait_phase()
        no_contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) < 5
        error = self.base_euler_xyz.unsqueeze(1).expand(-1, 2, -1) - self.feet_euler_xyz
        res = torch.sum(torch.exp(-torch.square(error[:,:,2]))*swing_mask*no_contact,dim=1)
        #print("_reward_feet_yaw=",error[0])
        return res
    
    def _reward_exp_action_smooothness(self):
        # 动作越发顺滑越好
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return torch.exp(-1e-2*(term_1 + term_2 + term_3))
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_power_dist(self):
        # Penalize power dist
        return torch.var(self.torques*self.dof_vel, dim=1)
    
    def _reward_power(self):
        return torch.sum(self.torques*self.dof_vel, dim=1)
    
    def _reward_exp_energy(self):
        return torch.exp(-1e-6*torch.sum(torch.square(self.dof_vel * self.torques),dim=1))
    
    def _reward_foot_normal_reward(self):
        # 获取原始接触力向量 [num_envs, num_feet, 3]
        contact_vectors = self.contact_forces[:, self.feet_indices, :3]
        
        # 计算接触力模长并添加极小值防止除零
        force_magnitude = torch.norm(contact_vectors, dim=-1, keepdim=True) + 1e-6
        
        # 归一化得到接触力方向（即推断的接触面法线）
        inferred_normals = contact_vectors / force_magnitude
        
        # 计算与理想地面法线（z轴方向）的对齐程度
        # 直接取z分量的值，等价于与[0,0,1]的点积
        vertical_alignment = inferred_normals[..., 2]  # shape: [num_envs, num_feet]
        
        # 创建接触掩码（排除非接触状态的脚）
        contact_mask = (force_magnitude.squeeze(-1) > 5.0)  # 5N力阈值，避免噪声干扰
        
        # 计算掩码加权后的对齐奖励
        alignment_reward = torch.sum(vertical_alignment * contact_mask, dim=-1) / (
            torch.sum(contact_mask, dim=-1) + 1e-6)
        
        # 可选：添加非线性增强
        alignment_reward = torch.where(
            alignment_reward > 0.8, 
            alignment_reward * 2.0,  # 高对齐区域奖励加倍
            alignment_reward * 0.5    # 低对齐区域奖励减半
        )
        
        return alignment_reward
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_hip_roll_default(self):
        hip_roll_differ = torch.mean(torch.square(self.dof_pos[:,[0,6]]),dim=-1)
        return torch.exp(-hip_roll_differ)
    
    def _reward_hip_roll_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[0,6]] * self.torques[:,[0,6]]),dim=1)
        return  torch.exp(-(1e-6*energy))
    
    def _reward_ankle_energy(self):
        energy = torch.sum(torch.square(self.dof_vel[:,[4,9]] * self.torques[:,[4,9]]),dim=1)
        return torch.exp(-(1e-6*energy))
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation_pen(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_exp_action_smooothness(self):
        # 动作越发顺滑越好
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return torch.exp(-1e-2*(term_1 + term_2 + term_3))

    def _reward_feet_swing_trajectory_z(self):
        # reward to the feet to track the given trajector in z axis
        # Initialize result tensor for each environment
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_body_states[:, self.feet_indices, 2] - self.cfg.rewards.feet_off_z
        ground_height = self._get_heights()
        for i in range(2):
            # Check if the leg is in swing phase (leg_phase >= 0.55)
            is_swing = self.leg_phase[:, i] >= 0.05       
            # Calculate pos_error for legs in swing phase
            self.target_height[:,i] = self.cfg.rewards.target_feet_height_swing *torch.sin((self.leg_phase[:, i])  * torch.pi)
            self.feet_height_g[:,i] = torch.mean(self.feet_pos[:, i, 2] .unsqueeze(1)- ground_height - self.cfg.rewards.feet_off_z, dim=1) 
            swing_pos_error = torch.square(self.feet_height_g[:,i] - self.target_height[:,i] )            
            # Set pos_error to a fixed value for legs not in swing phase
            not_swing_pos_error = torch.full_like(swing_pos_error, 10.0)            
            # Combine the errors based on the swing condition
            pos_error = torch.where(is_swing, swing_pos_error, not_swing_pos_error)            
            # Apply Gaussian reward function, and accumulate the error for this leg
            res += torch.exp(-pos_error/ 0.02)
        #print("trajectory_z:=",self.leg_phase[0],self.target_height[0])
        #print("trajectory_z:=",self.feet_height_g[0],self.target_height[0,0])
        return res

    def _reward_feet_swing_height(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ground_height = self._get_heights()
        for i in range(2):
            is_apex = (self.leg_phase[:, i] > 0.3) & (self.leg_phase[:, i] < 0.7)
            feet_height = torch.mean(self.feet_pos[:, i, 2] .unsqueeze(1) - ground_height - self.cfg.rewards.feet_off_z, dim=1)
            is_higher = feet_height>self.cfg.rewards.target_feet_height_swing*0.3
            res += is_higher*is_apex
        #print(self.feet_pos[0])
        return res

    def _reward_foot_landing_vel(self):
        z_vels = self.feet_vel[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 5
        ground_height = self._get_heights()
        feet_height = self.feet_pos[:, :, 2]  - self.cfg.rewards.feet_off_z
        about_to_land = (feet_height< self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
    #------------ cost functions----------------
    def _cost_torque_limit(self):
        # constaint torque over limit
        return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
    
    def _cost_pos_limit(self):
        upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        return out_limit
   
    def _cost_dof_vel_limits(self):
        return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
    
    def _cost_vel_smoothness(self):
        return torch.mean(torch.max(torch.zeros_like(self.dof_vel),torch.abs(self.dof_vel) - (self.dof_vel_limits/2.)),dim=1)
    
    def _cost_acc_smoothness(self):
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        acc_limit = self.dof_vel_limits/(2.*self.dt)
        return torch.mean(torch.max(torch.zeros_like(acc),torch.abs(acc) - acc_limit),dim=1)
    
    def _cost_collision(self):
        return  1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) > 0.0)
    
    def _cost_feet_contact_forces(self):
        # penalize high contact forces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > self.cfg.rewards.max_contact_force), dim=1) > 0.0)
        # return torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1))
    
    def _cost_stumble(self):
        # Penalize feet hitting vertical surfaces
        return 1.*(torch.sum(1.*(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2])), dim=1) > 0.0)

    def _cost_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        #print(base_height[0])
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _cost_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - self.cfg.rewards.cycle_time) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > self.cfg.rewards.command_dead #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return torch.max(torch.zeros_like(rew_airTime),-1.*rew_airTime)#1.*(rew_airTime < 0.0)
    
    def _cost_ang_vel_xy(self):
        ang_vel_xy = 0.01*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return ang_vel_xy
    
    def _cost_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _cost_torques(self):
        # Penalize torques
        torque_squres = 0.0001*torch.sum(torch.square(self.torques),dim=1)
        return torque_squres
    
    def _cost_action_rate(self):
        action_rate = 0.01*torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_rate
    
    def _cost_walking_style(self):
        # number of contact must greater than 2 at each frame
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        return 1.*(torch.sum(1.*contact_filt,dim=-1) < 3.)
    
    def _cost_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos_st), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < self.cfg.rewards.command_dead)
    
    def _cost_hip_pos(self):
        vel_y=self.cfg.commands.ranges.lin_vel_y[1]
        temp=self.dof_pos[:, [1,2,  7,8]] - self.default_dof_pos[:, [1,2,  7,8]]
        temp[:,0]*=2
        temp[:,2]*=2
        # temp[:,1]*= (vel_y-torch.abs(self.commands[:,1]))/vel_y
        # temp[:,4]*= (vel_y-torch.abs(self.commands[:,1]))/vel_y
        return  torch.sum(torch.square(temp), dim=1)
    
    def _cost_feet_height(self):
        # Reward high steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        foot_heights_cost = torch.sum(torch.square(self.dof_pos[:,[2,5,8,11]] - (-2.0)) * (~contact_filt),dim=1)
 
        return foot_heights_cost
    
    def _cost_contact_force_xy(self):
        contact_xy_force_norm = torch.mean(torch.norm(self.contact_forces[:, self.feet_indices, :2],dim=-1),dim=-1)
        return contact_xy_force_norm

    def _cost_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _cost_default_pos(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _cost_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.mean(contact_filt * foot_velocities, dim=1)
        return rew_slip
    
    def _cost_feet_contact_velocity(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        foot_velocities = torch.square(self.foot_velocities[:, :, 2].view(self.num_envs, -1))
        rew_contact_force = torch.mean(contact_filt * foot_velocities, dim=1)
        return rew_contact_force
    
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = 2.0
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    def _cost_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _cost_foot_slide(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)

        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.touch_thr
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        
        cost_slide = torch.sum(contact_filt * foot_leteral_vel, dim=1)
        return cost_slide

    def _cost_foot_regular(self):#摆动约束
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        #height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        height_error = torch.clamp(torch.exp(footpos_in_body_frame[:, :, 2]/(0.025*self.cfg.rewards.base_height_target)).view(self.num_envs, -1),0,1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _cost_trot_contact(self):#相序约束
        contact_filt = 1.*self.contact_filt
        pattern_match1 = torch.mean(torch.abs(contact_filt - self.trot_pattern1),dim=-1)
        pattern_match2 = torch.mean(torch.abs(contact_filt - self.trot_pattern2),dim=-1)
        pattern_match_flag = 1.*(pattern_match1*pattern_match2 > 0)
        return pattern_match_flag*(torch.norm(self.commands[:, :3], dim=1) > self.cfg.rewards.command_dead)