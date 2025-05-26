# active-adaptation

## Features
* Automatic shape handling for observation.
* Clean and efficient single-file RL implementation.
* Easy symmetry augmentation.
* Seamless Mujoco sim2sim.

## Current Limitations
* TorchRL stores redundant information and therefore the rollout buffer consumes more GPU memory.

## Installation

1. Install [Isaac Sim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) by downloading the latest release and unzip it to a desired location `$ISAACSIM_PATH`.
2. Install [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and setup a conda environment:
   ```bash
   conda create -n lab python=3.10
   conda activate lab
   # install IsaacLab to the exisiting conda environment
   # git clone https://github.com/isaac-sim/IsaacLab.git
   git clone git@github.com:isaac-sim/IsaacLab.git # SSH recommended
   cd IsaacLab
   ln -s $ISAACSIM_PATH _isaac_sim
   ./isaaclab.sh -c lab
   ./isaaclab.sh -i none # install without additional RL libraries
   # reactivate the environment
   conda activate lab
   echo $PYTHONPATH 
   ```
   You should see the isaac-sim related dependencies are added to `$PYTHONPATH`.
3. [**Recommended**] Isaac Sim comes with its cumstom Python environment which may lead to conflicts with our conda environment.
   To avoid Python environment conflicts, try the following steps:
   ```bash
   cd $ISAACSIM_PATH/exts/omni.isaac.ml_archive
   mv pip_prebundle pip_prebundle.back # backup the packages shipped with Isaac Sim
   ln -s $CONDA_PREFIX/lib/python3.10/site-packages
   ```
4. [**Optional**] VSCode setup. This enables the Python extension for code analysis to provide auto-completiong and linting.
5. `pip install -U torch torchvision tensordict torchrl`
6. Install this repo:
   ```bash
   git clone git@github.com:btx0424/active-adaptation.git # SSH recommended
   cd active-adaptation
   pip install -e . 
   ```


## Basic Usage

We use Hydra for configuration management. Each task is specified by a yaml file placed under `cfg/task` or `cfg/task/{subfolder}`, for example:

```yaml
# @package task
name: Go2Flat
task: Quadruped

robot: go2
terrain: plane
payload: false
homogeneous: false

action:
  _target_: active_adaptation.envs.mdp.action.JointPosition
  joint_names: .*_joint
  action_scaling: {.*_joint: 0.5}
  max_delay: 2
  alpha: [0.5, 1.0]

command:
  _target_: active_adaptation.envs.mdp.Command2
  linvel_x_range: [-1.0, 2.0]
  linvel_y_range: [-0.7, 0.7]
  angvel_range:   [-2.0, 2.0]
  yaw_stiffness_range: [0.5, 0.7]
  use_stiffness_ratio: 0.99
  aux_input_range: [.5, 1.]
  resample_prob: 0.5
  stand_prob: 0.02
  target_yaw_range: 
    - [-0.3927,  0.3927]
    - [ 1.1781,  1.9635]
    - [ 2.7489,  3.5343]
    - [ 4.3197,  5.1051]
  adaptive: true

observation:
  policy:
    command:
    projected_gravity_b: {noise_std: 0.05}
    joint_pos:    {noise_std: 0.05, joint_names: .*_joint}
    joint_vel:    {noise_std: 0.4, joint_names: .*_joint}
    prev_actions: {steps: 3}
  priv:
    applied_action:
    root_linvel_b:    {yaw_only: true}
    root_angvel_b:
    feet_pos_b:
    feet_vel_b:
    feet_height_map:  {feet_names: .*foot }
    applied_torques:  {actuator_name: base_legs}
    joint_forces:     {joint_names: .*_joint}
    external_forces:  {body_names: ["base"]}
    contact_indicator:  {body_names: [".*_foot", ".*_calf"], timing: true}

reward:
  loco:
    linvel_exp:         {weight: 1.5, enabled: true, dim: 3, yaw_only: true}
    angvel_z_exp:       {weight: 0.75, enabled: true}
    angvel_xy_l2:       {weight: 0.02, enabled: true}
    linvel_z_l2:        {weight: 2.0, enabled: true}
    base_height_l1:     {weight: 0.5, enabled: true, target_height: 0.35}
    energy_l1:          {weight: 0.0002, enabled: true}
    joint_acc_l2:       {weight: 2.5e-7, enabled: true}
    joint_torques_l2:   {weight: 2.0e-4, enabled: true}
    quadruped_stand:    {weight: 0.5}
    survival:           {weight: 1.0, enabled: true}
    action_rate_l2:     {weight: 0.01, enabled: true}
    feet_air_time:      {weight: 0.4, enabled: true, body_names: .*_foot, thres: 0.4}
  debug:
    feet_slip:          {weight: 1.0, enabled: false, body_names: .*_foot}
    feet_contact_count: {weight: 1.0, enabled: false, body_names: .*_foot}
    undesired_contact:  {body_names: [.*_calf, .*thigh, Head.*], weight: 0.25, enabled: true}

termination: # terminate upon any of the following checks being satisfied
  crash: {body_names_expr: [Head.*, "base"], t_thres: 0.5, z_thres: 0.}
  # joint_acc_exceeds: {thres: 5000}
  cum_error: {thres: 1.0}

randomization:
  random_scale: [1.0, 1.0]
  perturb_body_mass:
    (?!(payload|base|Head.*)).*: [0.8, 1.2]
    base: [0.8, 1.4]
```

Observations are grouped by keys and the observation of the same group is concatenated.

Rewards are grouped by keys and the rewards of the same group is summed up, excluding those marked with `enabled=false`. However, rewards with `enabled=false` will still be computed and logged as metrics for debugging purposes.

## Guide

### 1. Asset Specification

We borrow the asset specification and management of [IsaacLab](): each robot specification is defined with an `ArticulationCfg` in `active_adaptation/assets/` and stored in `active_adaptation.assets.ROBOTS`.

For Mujoco sim2sim verification (optional), provide a MJCF file that aligns with the USD and put the MJCF under `active_adaptation/assets_mjcf/`. We may implement USD-to-MJCF exporting in the future.

For symmetry augmentation, provide joint- and cartesian-space mappings that specify the left-right symmetry of a robot. See `assets.quadruped` for examples.

### 2. Task Definition

All components, including action, command, observation, reward, termination condition are defined by subclassing the base class. The base classes have a series of callbacks that will be called at each environment step:

```python
class Observation:
    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor
    
    def startup(self):
        """Called once upon initialization of the environment"""
        pass
    
    def post_step(self, substep: int):
        """Called after each physics substep"""
        pass

    def update(self):
        """Called after all physics substeps are completed"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass
```

The stepping logic is defined in `active_adaptation.envs.base._Env.step`.


### Training

Examples:

```bash
python test_env.py task=Go2/Go2Flat algo=ppo
```

### Evaluation and Visualization

Examples:

```bash
python eval_run.py --run_path ${wandb_run_path} -p # p for play
```

