import torch

class Buffer:
    def __init__(self, shape, size, device):
        self.data = torch.zeros(*shape, size, device=device)

    def reset(self, env_ids: torch.Tensor, value=0.):
        self.data[env_ids] = value
    
    def update(self, value: torch.Tensor):
        self.data[..., :-1] = self.data[..., 1:]
        self.data[..., -1] = value


class Observation:
    def __init__(self, env):
        self.env = env

    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass


class linvel_b_buffer(Observation):
    def __init__(self, env, size: int=4):
        super().__init__(env)
        self.size = size
        self.asset = env.scene["robot"]
        self.buffer = Buffer(self.asset.data.root_lin_vel_b.shape, size, env.device)

    def __call__(self):
        return self.buffer.data.reshape(self.env.num_envs, -1)

    def reset(self, env_ids):
        self.buffer.reset(env_ids)
    
    def update(self):
        self.buffer.update(self.asset.data.root_linvel_b)


class joint_pos_buffer(Observation):
    def __init__(self, env, size: int=4):
        super().__init__(env)
        self.size = size
        self.asset = env.scene["robot"]
        self.buffer = Buffer(self.asset.data.joint_pos.shape, size, env.device)

    def __call__(self):
        return self.buffer.data.reshape(self.env.num_envs, -1)

    def reset(self, env_ids):
        self.buffer.reset(env_ids)
    
    def update(self):
        self.buffer.update(self.asset.data.joint_pos)

