# Reward Specification Related To EE

command at `t` is the desired pos/vel in the next timestep `t+1`.
desired pos/vel at `t` is the desired pos/vel in the current timestep `t`.

## reset

`desired_lin_acc_ee_w_{t:t+future} = 0.0`
`desired_linvel_ee_w_{t:t+future} = real_linvel_ee_w_{t}`
`desired_pos_ee_w_{t:t+future} = real_pos_ee_w_{t}`

## integrate

<!-- currently not this step -->

compute desired pos/vel in the world frame at `t+1`:

`desired_pos_ee_w_{t+1}`

`desired_linvel_ee_w_{t+1}`

## update command

smooth the desired quantities to get command pos/vel in world frame:

`command_pos_ee_w_{t} = desired_pos_ee_w_{t+1}.mean(1)`

`command_linvel_ee_w_{t} = desired_linvel_ee_w_{t+1}.mean(1)`


compute command pos/vel of ee in base frame:

`commmand_pos_ee_b_{t} =  yaw_rotate(-command_yaw_{t}, command_pos_ee_w_{t} - command_pos_base_w_{t})`

`command_linvel_ee_b_{t} = yaw_rotate(-command_yaw_{t}, command_linvel_ee_w_{t} - desired_coriolis_ee_w_{t+1})`

where `desired_coriolis_ee_w_{t+1} = command_linvel_base_w_{t} + cross(command_angvel_w_{t}, command_pos_ee_w_{t} - command_pos_base_w_{t})`


set command_hidden:

`command_pos_ee_diff_b_{t} = command_pos_ee_b_{t} - real_pos_ee_b_{t}`
where `real_pos_ee_b = yaw_rotate(-real_yaw_{t}, real_ee_pos_w_{t} - real_base_pos_w_{t})`

## env step

policy takes command and takes action

## compute reward

the diff that is used to compute the reward is:

`rew_ee_pos_diff_{t} = command_pos_ee_b_{t} - real_pos_ee_b_{t+1}`

## integrate

compute desired pos/vel in the world frame at `t+2`:

`desired_pos_ee_w_{t+2}`

`desired_linvel_ee_w_{t+2}`

`desired_lin_acc_ee_w_{t+2}`

## update command

smooth the desired quantities to get command pos/vel in world frame:

`command_pos_ee_w_{t+1} = desired_pos_ee_w_{t+2}.mean(1)`
`command_linvel_ee_w_{t+1} = desired_linvel_ee_w_{t+2}.mean(1)`
`command_lin_acc_ee_w_{t+1} = desired_lin_acc_ee_w_{t+2}.mean(1)`

compute command pos/vel of ee in base frame:

`commmand_pos_ee_b_{t+1} =  yaw_rotate(-command_yaw_{t+1}, command_pos_ee_w_{t+1} - command_pos_base_w_{t+1})`

`command_linvel_ee_b_{t+1} = yaw_rotate(-command_yaw_{t+1}, command_linvel_ee_w_{t+1} - desired_coriolis_ee_w_{t+2})`

where `desired_coriolis_ee_w_{t+2} = command_linvel_base_w_{t+1} + cross(command_angvel_w_{t+1}, command_pos_ee_w_{t+1} - command_pos_base_w_{t+1})`

set command_hidden:

`command_pos_ee_diff_b_{t+1} = command_pos_ee_b_{t+1} - real_pos_ee_b_{t+1}`
where `real_pos_ee_b = yaw_rotate(-real_yaw_{t+1}, real_ee_pos_w_{t+1} - real_base_pos_w_{t+1})`

## env step

policy takes command and takes action

## compute reward

the diff that is used to compute the reward is:

`rew_ee_pos_diff_{t+1} = command_pos_ee_b_{t+1} - real_pos_ee_b_{t+2}`