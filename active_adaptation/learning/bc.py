import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.envs.transforms.transforms import TensorDictPrimer
from torchrl.data import UnboundedContinuousTensorSpec

from .ppo.common import make_batch, make_mlp
from .ppo.ppo_adapt import GRUModule


class BCPolicy(TensorDictModuleBase):
    def __init__(
        self, 
        observation_spec,
        action_spec,
        teacher,
        device,
    ):
        super().__init__()
        self.observation_spec = observation_spec
        self.action_dim = action_spec.shape[-1]
        self.device = device
        self.teacher = teacher

        fake_input = observation_spec.zero()
        fake_input["hx"] = torch.zeros((fake_input.shape[0], 128), device=self.device)
        self.actor = TensorDictSequential(
            TensorDictModule(GRUModule(256), ["policy", "is_init", "hx"], ["_feature", ("next", "hx")]),
            TensorDictModule(
                nn.Sequential(make_mlp([256]), nn.LazyLinear(self.action_dim)),
                ["_feature"], ["action"]
            )
        ).to(self.device)

        self.teacher(fake_input)
        self.actor(fake_input)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer({
            "hx": UnboundedContinuousTensorSpec((num_envs, 128))
        })
    
    def forward(self, tensordict: TensorDictBase):
        return self.actor(tensordict)
    
    def train_op(self, tensordict: TensorDictBase):
        with torch.no_grad():
            action_expert = self.teacher(tensordict.reshape(-1).to_tensordict())["action"]
        tensordict.set("action_expert", action_expert.reshape_as(tensordict["action"]))
        
        losses = []
        for epoch in range(4):
            for minibatch in make_batch(tensordict, 8, 32):
                loss = self.loss(minibatch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        return {"loss": sum(losses) / len(losses)}
    
    def loss(self, tensordict: TensorDictBase):
        action = self(tensordict)["action"]
        action_expert = tensordict["action_expert"]
        return F.mse_loss(action, action_expert)

