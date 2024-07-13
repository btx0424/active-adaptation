import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math

from torchrl.data import CompositeSpec, TensorSpec, TensorDictReplayBuffer, LazyTensorStorage, ListStorage
from torchrl.objectives import hold_out_net
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
from torchrl.modules import ProbabilisticActor, TanhNormal, TruncatedNormal
from torchrl.envs.transforms import CatTensors, ExcludeTransform, VecNorm, MultiStepTransform
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Mapping, Union
from copy import deepcopy

from .ppo.common import *
from .modules.distributions import IndependentNormal

@dataclass
class SACConfig:
    _target_: str = "active_adaptation.learning.sac.SAC"

    name: str = "sac"
    train_every: int = 32
    buffer_size: int = 5000000
    warm_up_steps: int = 100000
    lr: float = 5e-4

    checkpoint_path: Union[str, None] = None
    context_dim: int = 128

cs = ConfigStore.instance()
cs.store("sac", node=SACConfig, group="algo")

class SAC(TensorDictModuleBase):
    def __init__(
        self,
        cfg: SACConfig,
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        self.action_spec = action_spec

        fake_input = observation_spec.zero()
        self.action_dim = self.action_spec.shape[-1]
        self.target_entropy = 0 # - self.action_dim

        self.encoder_priv = TensorDictModule(
            make_mlp([self.cfg.context_dim]),
            [OBS_PRIV_KEY],
            ["context_expert"]
        ).to(self.device)

        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=TensorDictSequential(
                CatTensors([OBS_KEY, "context_expert"], "actor_input", del_keys=False),
                TensorDictModule(make_mlp([256, 256, 256]), ["actor_input"], ["actor_feature"]),
                TensorDictModule(Actor(self.action_dim, True), ["actor_feature"], ["loc", "scale"]),
            ),
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=TanhNormal,
            distribution_kwargs={"min": -torch.pi, "max": torch.pi},
            return_log_prob=True
        ).to(self.device)

        def make_critic():
            return nn.Sequential(make_mlp([512, 256, 256]), nn.LazyLinear(1))
        
        self.qs = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, ACTION_KEY], "q_input", del_keys=False),
            TensorDictModule(make_critic(), ["q_input"], ["Q1"]),
            TensorDictModule(make_critic(), ["q_input"], ["Q2"]),
        ).to(self.device)
        self.v = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY], "v_input", del_keys=False),
            TensorDictModule(make_critic(), ["v_input"], ["V"]),
        ).to(self.device)
        self.gae = GAE(0.99, 0.95)

        self.dynamics = TensorDictSequential(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, ACTION_KEY], "dyn_input", del_keys=False),
            TensorDictModule(
                nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(fake_input[OBS_KEY].shape[-1])),
                ["dyn_input"], [("next", OBS_KEY)]
            )
        ).to(self.device)

        self.encoder_priv(fake_input)
        self.actor(fake_input)
        self.qs(fake_input)
        self.v(fake_input)
        self.dynamics(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.)

        self.encoder_priv.apply(init_)
        self.actor.apply(init_)
        self.qs.apply(init_)
        self.v.apply(init_)

        self.qs_ema = deepcopy(self.qs)
        self.qs_ema.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(0., device=self.device))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=1e-2)

        self.opt_actor = torch.optim.Adam([
            {"params": self.actor.parameters()},
            {"params": self.encoder_priv.parameters()}
        ], lr=self.cfg.lr)
        self.opt_critic = torch.optim.Adam(self.qs.parameters(), lr=self.cfg.lr)
        self.opt_value = torch.optim.Adam(self.v.parameters(), lr=self.cfg.lr)
        self.opt_dyn = torch.optim.Adam(self.dynamics.parameters(), lr=cfg.lr)

        self.rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=self.cfg.buffer_size),
            batch_size=8192,
            prefetch=2,
        )
        self.multi_step = MultiStepTransform(3, gamma=0.99)


    def get_rollout_policy(self, mode: str="train"):
        def add_noise(x: torch.Tensor):
            return x + torch.randn_like(x).clamp(-1., 1.) * 0.1
        
        policy = TensorDictSequential(
            self.encoder_priv,
            self.actor,
            TensorDictModule(add_noise, [ACTION_KEY], [ACTION_KEY]),
            ExcludeTransform("actor_feature", "loc", "scale")
        )
        return policy

    def train_op(self, tensordict: TensorDictBase):
        tensordict = tensordict.copy()
        with torch.no_grad():
            self.v(tensordict)
            self.v(tensordict["next"])
            adv, ret = self.gae(
                tensordict[REWARD_KEY] ,
                tensordict[DONE_KEY],
                tensordict["V"],
                tensordict["next", "V"]
            )
            tensordict["ret"] = ret

            encoder_dr = cal_dormant_ratio(self.encoder_priv, tensordict, percentage=0.1)
            actor_dr = cal_dormant_ratio(self.actor, tensordict, percentage=0.1)
            q_dr = cal_dormant_ratio(self.qs, tensordict, percentage=0.1)

        # on-policy updates
        for _ in range(4):
            for minibatch in make_batch(tensordict, 2):
                value_loss = (minibatch["ret"] - self.v(minibatch)["V"]).square()
                value_loss = (value_loss * (~minibatch["is_init"]).float()).mean()
                self.opt_value.zero_grad()
                value_loss.backward()
                self.opt_value.step()
        with torch.no_grad():
            explained_var_v = (1. - F.mse_loss(ret, self.v(tensordict)["V"]) / ret.var()).item()
            
        self.rb.extend(tensordict.view(-1).cpu())
        if len(self.rb) < self.cfg.warm_up_steps:
            return {"rb_size": len(self.rb)}
        
        infos = []

        # off-policy updates
        for _ in range(self.cfg.train_every * 2):
            batch = self.rb.sample().to(self.device)
            infos.append(self.update(batch))
        
        soft_copy_(self.qs, self.qs_ema, 0.04)

        infos = {k: v.float().mean().item() for k, v in sorted(torch.stack(infos).items())}
        infos["value_loss/critic_v"] = value_loss.item()
        infos["value_loss/explained_var_v"] = explained_var_v
        infos["value_priv"] = tensordict["ret"].mean().item()
        infos["rb_size"] = len(self.rb)
        infos["alpha"] = self.log_alpha.exp().item()

        infos["dormant/encoder_dr"] = encoder_dr
        infos["dormant/actor_dr"] = actor_dr
        infos["dormant/q_dr"] = q_dr
        return infos
    
    def update(self, tensordict: TensorDictBase):
        losses = {}

        losses["dyn_loss"] = self._compute_dyn_loss(tensordict.copy())
        self.opt_dyn.zero_grad()
        losses["dyn_loss"].backward()
        self.opt_dyn.step()

        losses["value_loss/critic_q"] = self._compute_critic_loss(tensordict)
        self.opt_critic.zero_grad()
        losses["value_loss/critic_q"].backward()
        losses["qs_grad_norm"] = nn.utils.clip_grad_norm_(self.qs.parameters(), 2.)
        self.opt_critic.step()

        losses["actor_loss"] = self._compute_actor_loss(tensordict)
        self.opt_actor.zero_grad()
        losses["actor_loss"].backward()
        losses["actor_grad_norm"] = nn.utils.clip_grad_norm_(self.actor.parameters(), 2.)
        losses["encoder_grad_norm"] = nn.utils.clip_grad_norm_(self.encoder_priv.parameters(), 2.)
        self.opt_actor.step()

        losses["alpha_loss"] = -(self.log_alpha.exp() * (tensordict["sample_log_prob"].detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        losses["alpha_loss"].backward()
        self.opt_alpha.step()

        losses["entropy"] = -tensordict["sample_log_prob"].mean()
        losses["q_taken"] = tensordict["q_taken"].mean()

        return TensorDict(losses, [])
    
    def _compute_actor_loss(self, tensordict: TensorDictBase):
        self.encoder_priv(tensordict)
        dist = self.actor.get_dist(tensordict)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        tensordict[ACTION_KEY] = action
        tensordict["sample_log_prob"] = log_prob
        with hold_out_net(self.qs):
            self.qs(tensordict)
        q = 0.5 * (tensordict["Q1"] + tensordict["Q2"])
        tensordict["q_taken"] = q
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q).mean()

        return actor_loss
    
    def _compute_critic_loss(self, tensordict: TensorDictBase):
        with torch.no_grad():
            self.encoder_priv(tensordict["next"])
            self.actor(tensordict["next"])
            self.qs_ema(tensordict["next"])
            entropy_bonus = - self.log_alpha.exp() * tensordict["next", "sample_log_prob"].unsqueeze(-1)
            next_q = 0.5 * (tensordict["next", "Q1"] + tensordict["next", "Q2"])
            next_v = self.v(tensordict["next"])["V"]
            next_q = torch.where(next_v > next_q, next_q.lerp(next_v, 0.2), next_q)
            gamma = 0.99 * (1 - tensordict[DONE_KEY].float())
            td_target = (tensordict[REWARD_KEY] + gamma * (next_q + entropy_bonus)).detach()
        
        self.qs(tensordict)
        
        critic_loss = (
            F.mse_loss(tensordict["Q1"], td_target)
            + F.mse_loss(tensordict["Q2"], td_target)
        )
        return critic_loss

    def _compute_dyn_loss(self, tensordict: TensorDictBase):
        pred = self.dynamics(tensordict.copy())["next", OBS_KEY]
        target = tensordict["next", OBS_KEY]
        dyn_loss = F.mse_loss(pred, target)
        return dyn_loss

    def state_dict(self):
        state_dict = super().state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=strict)


class LinearOutputHook:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)


def cal_dormant_ratio(model, *inputs, percentage=0.025):
    hooks = []
    hook_handlers = []
    total_neurons = 0
    dormant_neurons = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Mish):
            hook = LinearOutputHook()
            hooks.append(hook)
            hook_handlers.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(*inputs)

    for module, hook in zip(
        (module
         for module in model.modules() if isinstance(module, nn.Linear)),
            hooks):
        with torch.no_grad():
            for output_data in hook.outputs:
                mean_output = output_data.abs().mean(0)
                avg_neuron_output = mean_output.mean()
                dormant_indices = (mean_output < avg_neuron_output *
                                   percentage).nonzero(as_tuple=True)[0]
                total_neurons += module.weight.shape[0]
                dormant_neurons += len(dormant_indices)         

    for hook in hooks:
        hook.outputs.clear()

    for hook_handler in hook_handlers:
        hook_handler.remove()

    return dormant_neurons / total_neurons