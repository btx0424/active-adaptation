# active-adaptation

## Installation

* [Isaac Sim 4.1.0]()
* [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
* [TensorDict](https://github.com/btx0424/tensordict) from GitHub source.
* [TorchRL](https://github.com/btx0424/rl) from GitHub source.

**DO NOT** install tensordict and torchrl using `pip install`. They are under active development so the release versions on PyPi might have bugs and lack new functionalities.

## Basic Usage

Each task is specified by a yaml file under `cfg/task`, for example:

```yaml
# @package task
name: Go2Flat
task: Quadruped

defaults:
  # see https://hydra.cc/docs/advanced/overriding_packages/
  - /task/Velocity@_here_
  - override /task/action@action: null
  - _self_

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


### Training

Examples:

```bash
python test_env.py task=Go2/Go2Flat algo=ppo
python test_env.py task=ORCA/CY1Flat algo=ppo_adapt_train total_frames=250000000
```

### Evaluation and Visualization

Examples:

```bash
python eval_run.py --run_path ${wandb_run_path} -p # p for play
```

## Adding New Tasks

All of observation, reward, termination, randomization and command follow a similar protocol, for example:

```python

class Observation:
    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def update(self):
        """Called at each step **after** simulation"""

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""

```

Inherit from and extend the classes in `envs/mdp/xxx.py` to implement environment logic.
