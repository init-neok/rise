"""
RISE: Routing Isolation-guided Subnetwork Enhancement
=====================================================
Adapt a Mixture-of-Experts language model to a low-resource language by
fine-tuning only the experts that language actually routes to.

High- and low-resource languages activate largely disjoint expert sets.
RISE measures that separation, turns it into a small expert subnetwork, and
trains only those experts -- so the target language improves while the rest
of the model, and its other languages, stay untouched.

Three stages
------------
1. :mod:`rise.routing_stats`   -- replay text, record expert activation frequencies.
2. :mod:`rise.expert_selector` -- score and select the expert subnetwork.
3. :mod:`rise.trainer`         -- freeze everything else and fine-tune.
"""

from .expert_selector import RISEConfig, RISESelector, SelectedExpert
from .model_freezer import (
    freeze_model_except_experts,
    load_expert_weights,
    load_selected_experts,
    save_expert_weights,
)
from .routing_stats import RoutingStatCollector, collect_routing_stats
from .trainer import RISETrainingArguments, build_causal_lm_dataset, train_rise

__version__ = "1.0.0"

__all__ = [
    # Stage 1
    "RoutingStatCollector",
    "collect_routing_stats",
    # Stage 2
    "RISEConfig",
    "RISESelector",
    "SelectedExpert",
    # Freezing / checkpoints
    "freeze_model_except_experts",
    "save_expert_weights",
    "load_expert_weights",
    "load_selected_experts",
    # Stage 3
    "RISETrainingArguments",
    "build_causal_lm_dataset",
    "train_rise",
]
