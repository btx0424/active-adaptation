import torch
import torch.distributions as D
import torch.nn.functional as F

from torchrl.data import TensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
from torchrl.envs.transforms import Transform
from torchrl.objectives.utils import hold_out_net
from torchrl.modules import ProbabilisticActor

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from typing import Sequence

class AdaptationModule(Transform):

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        reward_spec.set(
            "reward_adaptation", 
            reward_spec[("agents", "reward")].clone()
        )
        return reward_spec
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec["stats"].set(
            "return_adaptation", 
            UnboundedContinuousTensorSpec(
                (observation_spec.shape[0], 1), 
                device=observation_spec.device
            )
        )
        return observation_spec
    
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        state_spec = input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = CompositeSpec(shape=input_spec.shape, device=input_spec.device)
        state_spec.set(
            ("stats", "return_adaptation"), 
            UnboundedContinuousTensorSpec(
                (input_spec.shape[0], 1), 
                device=input_spec.device
            )
        )
        input_spec["full_state_spec"] = state_spec
        return input_spec
        
    def forward(self, tensordict: TensorDictBase):
        raise NotImplementedError
    
    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        _reset = tensordict.get("_reset", None)
        if _reset is None or _reset.any():
            value = tensordict.get(("stats", "return_adaptation"), None)
            if value is not None:
                if _reset is None:
                    tensordict.set(
                        ("stats", "return_adaptation"),
                        torch.zeros_like(value)
                    )
                else:
                    tensordict.set(
                        ("stats", "return_adaptation"),
                        value * (1. - _reset.reshape(value.shape).float())
                    )
            else:
                tensordict.set(
                    ("stats", "return_adaptation"),
                    torch.zeros(tensordict.shape[0], 1, device=tensordict.device)
                )
        return tensordict

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        reward_adaptation = tensordict.get("reward_adaptation")
        return_adaptation = tensordict.get(("stats", "return_adaptation"), None)
        # if return_adaptation is None:
        #     return_adaptation = torch.zeros(
        #         tensordict.shape[0], 1, device=tensordict.device
        #     )
        # return_adaptation = return_adaptation * (1. - next_tensordict.get("is_init").float())
        next_tensordict.set(
            ("stats", "return_adaptation"),
            return_adaptation + reward_adaptation.squeeze(1)
        )
        next_tensordict.set("reward_adaptation", reward_adaptation)
        return next_tensordict

    def update(self, tensordict: TensorDictBase):
        raise NotImplementedError


class MSE(AdaptationModule):
    def __init__(self, adaptation_module: TensorDictModule, keys: Sequence[str]):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.keys = keys
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict):
        target = tensordict.select(*self.keys)
        pred = self.adaptation_module(tensordict).select(*self.keys)
        loss = sum([
            F.mse_loss(pred[k], target[k], reduction="none") 
            for k in self.keys
        ])
        return loss

    def update(self, tensordict):
        info = []
        for epoch in range(4):
            for batch in make_batch(tensordict, 8):
                loss = self(batch).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


class Action(AdaptationModule):
    def __init__(
        self,
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        actor: ProbabilisticActor,
        closed_kl: bool = False
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.closed_kl = closed_kl
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        target = self.actor.get_dist(tensordict)
        td = self.adaptation_module(tensordict)
        if self.closed_kl:
            pred = self.actor.get_dist(td)
            loss = D.kl_divergence(pred, target)
        else:
            pred = self.actor(td)[("agents", "action")]
            loss = -target.log_prob(pred)
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
        with hold_out_net(self.actor):
            for epoch in range(4):
                for batch in make_batch(tensordict, 8):
                    loss = self(batch).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


class Value(AdaptationModule):
    def __init__(
        self,
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        critic: TensorDictModule,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.critic = critic
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        target = self.critic(tensordict).get("state_value")
        td = self.adaptation_module(tensordict)
        pred = self.critic(td).get("state_value")
        loss = F.mse_loss(pred, target, reduction="none")
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
        with hold_out_net(self.critic):
            for epoch in range(4):
                for batch in make_batch(tensordict, 8):
                    loss = self(batch).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


class ActionValue(AdaptationModule):
    def __init__(
        self,
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        actor: ProbabilisticActor,
        critic: TensorDictModule
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.critic = critic
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        value_target = self.critic(td).get("state_value")
        action_target = self.actor.get_dist(td)
        
        td = self.adaptation_module(tensordict)
        value_pred = self.critic(td).get("state_value")
        action_pred = self.actor.get_dist(td)

        loss = (
            D.kl_divergence(action_pred, action_target)
            + F.mse_loss(value_pred, value_target, reduction="none")
        )
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
        with hold_out_net(self.critic), hold_out_net(self.actor):
            for epoch in range(4):
                for batch in make_batch(tensordict, 8):
                    loss = self(batch).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]