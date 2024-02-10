# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from .ppo import PPOPolicy
from .ppo_adapt import PPORMAPolicy
from .ppo_rnn import PPORNNPolicy
from .ppo_tconv import PPOTConvPolicy
from .ppo_contrastive import PPOTConvPolicy as PPOContraPolicy
from .ppo_dual import PPODualPolicy
from .ppo_roa import PPOROAPolicy
from .ppo_model import PPOModelPolicy
from .ppo_guided import PPOGuidedPolicy
from .ppo_static import PPOStaticPolicy
from .final import Policy as PPOFinalPolicy
from .ppg import PPGPolicy

ALGOS = {
    "ppo": PPOPolicy,
    "ppo_dual": PPODualPolicy,
    "ppo_rnn": PPORNNPolicy,
    "ppo_tconv": PPOTConvPolicy,
    "ppo_contra": PPOContraPolicy,
    "ppo_rma": PPORMAPolicy,
    "ppo_roa": PPOROAPolicy,
    "ppo_model": PPOModelPolicy,
    "ppo_guided": PPOGuidedPolicy,
    "ppo_static": PPOStaticPolicy,
    "final": PPOFinalPolicy,
    "ppg": PPGPolicy
}