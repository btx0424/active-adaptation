import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    TensorDictModuleBase,
)
from torchrl.envs import EnvBase
from active_adaptation.learning.ppo.common import *
from torch.utils._pytree import tree_map
from collections import OrderedDict
import warnings


@dataclass
class TD3Config:
    _target_: str = "active_adaptation.learning.td3.td3.TD3"
    name: str = "td3"
    train_every: int = 1
    gamma: float = 0.99
    max_grad_norm: float = 2.0
    learning_starts: int = 10

    buffer_size: int = 1024 * 5
    batch_size: int = 32768
    num_updates: int = 2 # number of updates per step
    policy_frequency: int = 2
    n_steps: int = 4
    backtrack: bool = False


cs = ConfigStore.instance()
cs.store("td3", node=TD3Config, group="algo")


class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
        )
        self.act = nn.LazyLinear(action_dim)
    
    def forward(self, obs):
        return F.tanh( self.act(self.layers(obs)) / 2. ) * 2.


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_net_1 = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(1),
        )
        self.q_net_2 = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.LazyLinear(1),
        )
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        obs_act = torch.cat([obs, act], dim=-1)
        q1 = self.q_net_1(obs_act)
        q2 = self.q_net_2(obs_act)
        return torch.cat([q1, q2], dim=-1)


class Noise(nn.Module):
    def __init__(self, batch_size, action_dim):
        super().__init__()
        self.noise_scales: torch.Tensor
        self.noise: torch.Tensor
        self.theta: float = 0.15
        self.register_buffer("noise_scales", torch.ones(batch_size, action_dim))
        self.register_buffer("noise", torch.zeros(batch_size, action_dim))

    def forward(self, action: torch.Tensor, is_init: torch.Tensor):
        if is_init.any():
            is_init = is_init
            new_scales = torch.rand_like(self.noise_scales) * 0.8 + 0.2
            self.noise_scales = torch.where(is_init, new_scales, self.noise_scales)
        # Ornstein-Uhlenbeck process
        sigma = torch.randn_like(action).clamp(-3, 3.) 
        self.noise += self.theta * -self.noise + sigma * self.noise_scales
        return (action + self.noise).clamp(-2., 2.)


class TD3(TensorDictModuleBase):
    def __init__(
        self,
        cfg: TD3Config,
        observation_spec,
        action_spec,
        reward_spec,
        device,
        env: EnvBase,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.gamma = cfg.gamma
        self.n_steps = cfg.n_steps
        self.max_grad_norm = cfg.max_grad_norm
        self.num_updates = cfg.num_updates

        fake_input = observation_spec.zero()
        action_dim = action_spec.shape[-1]

        self.actor = Mod(Actor(action_dim), [OBS_KEY], [ACTION_KEY]).to(self.device)
        self.critic = Mod(Critic(), [OBS_KEY, ACTION_KEY], ["qs"]).to(self.device)
        self.noise = Mod(
            Noise(fake_input.shape[0], action_dim),
            [ACTION_KEY, "is_init"],
            [ACTION_KEY],
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        def init(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Actor):
                nn.init.normal_(m.act.weight, 0.0, 0.01)
        
        self.actor.apply(init)
        self.critic.apply(init)

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=2e-4, weight_decay=0.1)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=5e-4, weight_decay=0.1)

        self.global_step = 0
        self.buffer_ptr = 0
        fake_tensordict = env.fake_tensordict()
        buffer_shape = (fake_tensordict.shape[0], cfg.buffer_size)
        self.buffer: TensorDict = (
            fake_tensordict.exclude(("next", "stats"))
            .unsqueeze(1)
            .expand(buffer_shape)
            .clone()
        )
        self.batch_size = cfg.batch_size // buffer_shape[0]
        
    def get_rollout_policy(self, mode: str="train"):
        if mode == "train":
            return Seq(self.actor, self.noise)
        else:
            return Seq(self.actor, self.noise)
    
    def train_op(self, tensordict: TensorDictBase):

        infos = {"buffer": min(self.buffer_ptr, self.buffer.shape[1])}
        tensordict = tensordict.exclude(("next", "stats"), "collector")
        
        self.buffer[:, self.buffer_ptr % self.buffer.shape[1]] = tensordict.squeeze(1).to(self.buffer.device)
        self.buffer_ptr += 1
        self.global_step += 1

        if self.global_step < self.cfg.learning_starts:
            return infos
        
        critic_infos = []
        actor_infos = []

        indices_start = None
        for i in range(self.num_updates):
            batch, indices_start = self.sample_batch(indices_start)
            critic_infos.append(self.update_critic(batch))
            if i % self.cfg.policy_frequency == 1:
                actor_infos.append(self.update_actor(batch))
            soft_copy_(self.actor, self.actor_target, tau=0.1)
            soft_copy_(self.critic, self.critic_target, tau=0.1)
        
        critic_infos = tree_map(lambda *xs: sum(xs).item() / len(xs), *critic_infos)
        actor_infos = tree_map(lambda *xs: sum(xs).item() / len(xs), *actor_infos)
        infos.update(critic_infos)
        infos.update(actor_infos)
        return {k: v for k, v in sorted(infos.items())}
    
    def sample_batch(self, indices_start = None) -> TensorDictBase:
        shape = (self.buffer.shape[0], self.batch_size)
        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_ptr, self.buffer.shape[1]),
                shape,
                device=self.device,
            )
            samples: TensorDictBase = self.buffer.gather(dim=1, index=indices).reshape(*shape)
            return samples, None
        else:            
            indices = torch.randint(
                0,
                min(self.buffer_ptr, self.buffer.shape[1] - self.n_steps),
                shape,
                device=self.device,
            )
            seq_offsets = torch.arange(self.n_steps, device=self.device)
            all_indices = indices.unsqueeze(-1) + seq_offsets
            samples: TensorDictBase = self.buffer.gather(
                dim=1, index=all_indices.flatten(start_dim=1)
            ).reshape(*shape, self.n_steps)

            all_dones = samples["next", "done"] # [*shape, n_steps, 1]
            all_rewards = samples["next", "reward"].sum(dim=-1, keepdim=True) # [*shape, n_steps, 1]
            done_masks = torch.cumprod(1.0 - all_dones.float(), dim=2) # [*shape, n_steps, 1]
            discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device)) # [n_steps]
            
            masked_rewards = all_rewards * done_masks # [*shape, n_steps, 1]
            discounted_rewards = masked_rewards * discounts.unsqueeze(-1) # [*shape, n_steps, 1]
            n_step_rewards = discounted_rewards.sum(dim=2) # [*shape, 1]
            no_dones = all_dones.sum(dim=2) == 0
            first_done = torch.argmax((all_dones > 0).float(), dim=2)
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            final_indices = torch.gather(all_indices, 2, first_done).squeeze(-1)
            final_next = self.buffer["next"].gather(1, final_indices).reshape(*shape)
            
            result = samples[:, :, 0]
            assert torch.all(~no_dones == final_next["done"])
            result["next", "policy"] = final_next["policy"]
            result["next", "reward"] = n_step_rewards
            result["next", "done"] = final_next["done"]
            result["next", "terminated"] = final_next["terminated"]
            result["next", "truncated"] = final_next["truncated"]
            return result, None
    
    def update_critic(self, tensordict: TensorDictBase):
        next_tensordict = tensordict["next"]
        
        reward = next_tensordict["reward"]
        gamma = (self.gamma**self.n_steps) * (~next_tensordict["terminated"]).float()
        
        with torch.no_grad():
            next_action = self.actor_target(next_tensordict)["action"]
            noise = torch.randn_like(next_action).clamp(-3., 3.)
            next_action = torch.clamp(next_action + (noise * 0.05), -2., 2.)
            next_tensordict["action"] = next_action
            next_qs = reward + gamma * self.critic_target(next_tensordict)["qs"]
            q_target = torch.min(next_qs, dim=-1, keepdim=True).values - next_qs.var(dim=-1, keepdim=True)

        qs = self.critic(tensordict)["qs"]
        q_loss = 2.0 * F.mse_loss(qs, q_target.expand_as(qs))

        self.opt_critic.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=self.max_grad_norm,
        )
        self.opt_critic.step()
        return {
            "critic/q_std": qs.detach().std(dim=-1).mean(),
            "critic/q_loss": q_loss.detach(),
            "critic/q_value": qs.detach().mean(),
            "critic/grad_norm": critic_grad_norm,
        }
    
    def update_actor(self, tensordict: TensorDictBase):
        action_buffer = tensordict["action"].clone()
        self.actor(tensordict)
        action_old = tensordict["action"].clone()
        tensordict["action"].retain_grad()
        qs = self.critic(tensordict)["qs"]
        q_value = torch.mean(qs, dim=-1)
        actor_loss = -q_value.mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=self.max_grad_norm,
        )
        self.opt_actor.step()
        with torch.no_grad():
            action = self.actor(tensordict)["action"]
            action_dev_buffer = (action - action_buffer).norm(dim=-1)
            action_dev_update = (action - action_old).norm(dim=-1)
        return {
            "actor/action_dev_buffer": action_dev_buffer.mean(),
            "actor/action_dev_update": action_dev_update.mean(),
            "actor/loss": actor_loss.detach(),
            "actor/grad_norm": actor_grad_norm,
        }
    
    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            state_dict[name] = module.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys

