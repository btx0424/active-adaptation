from .ppo import *
from .bc import BCPolicy
from .sac import SAC
from .hbc import HBC

ALGOS = {
    "ppo": PPOPolicy,
    "ppo_asy": PPOAsyPolicy,
    "ppo_dual": PPODualPolicy,
    "ppo_rnn": PPORNNPolicy,
    "ppo_roa": PPOROAPolicy,
    "ppo_guided": PPOGuidedPolicy,
    "ppo_adapt": PPOAdaptPolicy,
    "ppo_ji": PPOJi,
    "ppo_stoch": PPOStochPolicy,
    # "ppo_rma": PPORMAPolicy,
    "ppg": PPGPolicy,
    "sac": SAC,
    "hbc": HBC,
    "bc": BCPolicy
}