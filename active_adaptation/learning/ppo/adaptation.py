import torch
import torch.nn as nn
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
    def __init__(self, adaptation_module: TensorDictModule):
        super().__init__()
        self.adaptation_module = adaptation_module
    
    def forward(self, tensordict: TensorDictBase, out: TensorDictBase, mean: bool=False):
        self.adaptation_module(tensordict)
        pred = tensordict["context_adapt"]
        target = tensordict["context_expert"]
        loss = F.mse_loss(pred, target, reduction="none")
        if mean:
            loss = loss.mean()
        out["adaptation_loss"] = loss.mean(-1, keepdim=True)
        return out


class Action(AdaptationModule):
    def __init__(
        self,
        adaptation_module: TensorDictModule,
        actor_expert: ProbabilisticActor,
        actor_adapt: ProbabilisticActor,
        closed_kl: bool = False
    ) -> None:
        super().__init__()
        self.adaptation_module = adaptation_module
        self.actor_expert = actor_expert
        self.actor_adapt = actor_adapt
        self.closed_kl = closed_kl
    
    def forward(self, tensordict: TensorDictBase, out: TensorDictBase, mean: bool=False):
        target_dist = self.actor_expert.get_dist(tensordict)
        self.adaptation_module(tensordict)
        if self.closed_kl:
            pred_dist = self.actor_adapt.get_dist(tensordict)
            loss = D.kl_divergence(pred_dist, target_dist)
            # loss = D.kl_divergence(target_dist, pred_dist)
        else:
            pred_dist = self.actor_adapt.get_dist(tensordict)
            pred_action = pred_dist.rsample()
            loss = pred_dist.log_prob(pred_action)-target_dist.log_prob(pred_action)
            # loss = -target_dist.log_prob(pred_action)
        if mean:
            loss = loss.mean()
        out.set("adaptation_loss", loss.unsqueeze(-1))
        return out


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
            for epoch in range(2):
                for batch in make_batch(tensordict, 8):
                    loss = self(batch).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    info.append(loss)
        return {"adapt_loss": torch.stack(info).mean().item()}


class Discriminator(AdaptationModule):
    def __init__(
        self, 
        encoder: TensorDictModule,
        adaptation_module: TensorDictModule,
        actor: ProbabilisticActor,
    ):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.LazyLinear(256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.LazyLinear(1),
            nn.Tanh()
        )
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.opt_dis = torch.optim.Adam(self.discriminator.parameters())
        self.opt = torch.optim.Adam(self.adaptation_module.parameters(), lr=1e-4)
    
    def forward(self, tensordict: TensorDictBase):
        obs = tensordict[("agents", "observation")]
        td = self.adaptation_module(tensordict)
        false_action = self.actor(td)[("agents", "action")]
        false = self.discriminator(torch.cat([obs, false_action], dim=-1))
        loss = F.mse_loss(false, torch.ones_like(false), reduction="none")
        return loss

    def loss_discriminator(self, tensordict: TensorDictBase):
        obs = tensordict[("agents", "observation")]
        true_action = self.actor(tensordict)[("agents", "action")]
        true = self.discriminator(torch.cat([obs, true_action], dim=-1))
        td = self.adaptation_module(tensordict.clone())
        false_action = self.actor(td)[("agents", "action")].detach()
        false = self.discriminator(torch.cat([obs, false_action], dim=-1))
        loss = (
            0.5 * F.mse_loss(true, torch.ones_like(true)) 
            + 0.5 * F.mse_loss(false, torch.zeros_like(false))
        )
        return loss

    def update(self, tensordict: TensorDictBase):
        info = {"adapt_loss": [], "dis_loss": []}
        with hold_out_net(self.actor):
            for epoch in range(4):
                for batch in make_batch(tensordict, 8):
                    loss_discriminator = self.loss_discriminator(batch)
                    self.opt_dis.zero_grad()
                    loss_discriminator.backward()
                    self.opt_dis.step()
                    
                    loss = self(batch).mean()
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    info["adapt_loss"].append(loss)
                    info["dis_loss"].append(loss_discriminator)
        return {
            "adapt_loss": torch.stack(info["adapt_loss"]).mean().item(), 
            "dis_loss": torch.stack(info["dis_loss"]).mean().item()
        }
    

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]