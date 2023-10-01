import torch
import torch.distributions as D
import torch.nn.functional as F

from torchrl.data import TensorSpec
from torchrl.envs.transforms import Transform
from torchrl.objectives.utils import hold_out_net

from tensordict import TensorDict, TensorDictBase


class AdaptationModule(Transform):

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        reward_spec.set("reward_adaptation", reward_spec.clone())
        return reward_spec
    
    def forward(self, tensordict: TensorDictBase):
        raise NotImplementedError
    
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        loss = self(tensordict)
        tensordict.set("reward_adaptation", -loss)
        return tensordict

    def update(self, tensordict: TensorDictBase):
        raise NotImplementedError


class MSE(AdaptationModule):
    def __init__(self, adaptation_module: TensorDictModule, key: str):
        super().__init__()
        self.adaptation_module = adaptation_module
        self.key = key
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict):
        target = tensordict.get(self.key)
        pred = self.adaptation_module(tensordict).get(self.key)
        loss = F.mse_loss(pred, target, reduction="none")
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
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adaptation_module = adaptation_module
        self.actor = actor
        self.opt = torch.optim.Adam(self.adaptation_module.parameters())
    
    def forward(self, tensordict: TensorDictBase):
        with torch.no_grad():
            td = self.encoder(tensordict.exclude("context"))
            target = self.actor.get_dist(td)
        with hold_out_net(self.actor):
            td = self.adaptation_module(tensordict.exclude("context"))
            pred = self.actor.get_dist(td)
        loss = D.kl_divergence(pred, target)
        # loss = D.kl_divergence(target, pred).mean()
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
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
        if tensordict.get("context", None) is None:
            with torch.no_grad():
                td = self.encoder(tensordict)
                target = self.critic(td).get("state_value")
        
        with hold_out_net(self.critic):
            td = self.adaptation_module(tensordict.exclude("context"))
            pred = self.critic(td).get("state_value")
        loss = F.mse_loss(pred, target, reduction="none")
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
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
        if tensordict.get("context", None) is None:
            with torch.no_grad():
                td = self.encoder(tensordict.exclude("context"))
                value_target = self.critic(td).get("state_value")
                action_target = self.actor.get_dist(td)
        
        with hold_out_net(self.critic), hold_out_net(self.actor):
            td = self.adaptation_module(tensordict.exclude("context"))
            value_pred = self.critic(td).get("state_value")
            action_pred = self.actor.get_dist(td)
        loss = (
            D.kl_divergence(action_pred, action_target)
            + F.mse_loss(value_pred, value_target, reduction="none")
        )
        return loss
    
    def update(self, tensordict: TensorDictBase):
        info = []
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