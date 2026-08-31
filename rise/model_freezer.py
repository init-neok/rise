"""
Parameter freezing and compact expert checkpoints
=================================================
RISE trains only the selected K experts (roughly 2-3% of all parameters)
and leaves every other weight frozen.  Two steps:

1. ``requires_grad_(False)`` on all parameters.
2. ``requires_grad_(True)`` on each selected expert module.

After training, only the selected expert tensors are written out.  For
Qwen3-30B-A3B that is a few hundred MB instead of a ~60 GB full
checkpoint, and the base model on disk is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .moe import get_expert_module

__all__ = [
    "get_selected_expert_modules",
    "freeze_model_except_experts",
    "save_expert_weights",
    "load_expert_weights",
    "load_selected_experts",
]


def _expert_key(entry: Dict) -> str:
    """Checkpoint key prefix for one selected expert."""
    return f"layer{entry['layer']}_expert{entry['expert']}"


def get_selected_expert_modules(
    model: nn.Module,
    selected_experts: List[Dict],
    model_type: str,
) -> List[nn.Module]:
    """
    Resolve ``[{"layer": l, "expert": i}, ...]`` to the matching modules.

    Parameters
    ----------
    model:
        The loaded language model.
    selected_experts:
        The ``"experts"`` list from a Stage 2 ``selected_experts.json``.
    model_type:
        Model family identifier, e.g. ``"qwen3_moe"``.
    """
    return [
        get_expert_module(model, e["layer"], e["expert"], model_type)
        for e in selected_experts
    ]


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def freeze_model_except_experts(
    model: nn.Module,
    selected_experts: List[Dict],
    model_type: str,
) -> Tuple[int, int]:
    """
    Freeze everything, then unfreeze the selected expert subnetwork.

    Returns
    -------
    (trainable_params, total_params)
    """
    if not selected_experts:
        raise ValueError("No experts selected; there would be nothing to train.")

    for param in model.parameters():
        param.requires_grad_(False)

    for module in get_selected_expert_modules(model, selected_experts, model_type):
        for param in module.parameters():
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    if trainable == 0:
        raise RuntimeError(
            "Freezing left zero trainable parameters. The selected experts do "
            "not resolve to parameterised modules for model_type "
            f"{model_type!r} -- check that the selection JSON matches the model."
        )
    return trainable, total


# ---------------------------------------------------------------------------
# Compact expert checkpoints
# ---------------------------------------------------------------------------

def save_expert_weights(
    model: nn.Module,
    selected_experts: List[Dict],
    model_type: str,
    output_path: str,
) -> None:
    """
    Write only the selected experts' tensors to a ``.pt`` file.

    The file is a flat ``{"layer{l}_expert{i}.{param}": tensor}`` dict, so it
    can be inspected or reloaded without instantiating the architecture.
    """
    modules = get_selected_expert_modules(model, selected_experts, model_type)
    weights: Dict[str, torch.Tensor] = {}

    for entry, module in zip(selected_experts, modules):
        prefix = _expert_key(entry)
        for name, param in module.named_parameters():
            weights[f"{prefix}.{name}"] = param.detach().to("cpu", copy=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)

    size_mb = out.stat().st_size / 1024 ** 2
    print(f"Saved {len(weights)} expert tensors to {out} ({size_mb:.1f} MB)")


def load_expert_weights(
    model: nn.Module,
    selected_experts: List[Dict],
    model_type: str,
    weights_path: str,
) -> None:
    """
    Load fine-tuned expert tensors back into a base model, in place.

    Use this to rebuild the fine-tuned model for evaluation: load the original
    checkpoint, then apply the compact expert file produced by
    :func:`save_expert_weights`.

    Raises
    ------
    KeyError
        If the checkpoint is missing tensors for the requested experts, which
        means the selection JSON and the weights file do not belong together.
    """
    weights: Dict[str, torch.Tensor] = torch.load(weights_path, map_location="cpu")
    modules = get_selected_expert_modules(model, selected_experts, model_type)

    missing: List[str] = []
    loaded = 0
    for entry, module in zip(selected_experts, modules):
        prefix = _expert_key(entry)
        for name, param in module.named_parameters():
            key = f"{prefix}.{name}"
            if key not in weights:
                missing.append(key)
                continue
            param.data.copy_(weights[key].to(device=param.device, dtype=param.dtype))
            loaded += 1

    if missing:
        raise KeyError(
            f"{len(missing)} expert tensors are absent from {weights_path} "
            f"(first missing: {missing[0]}). The selection JSON and the weights "
            "file do not match."
        )

    print(f"Loaded {loaded} expert tensors from {weights_path}")


def load_selected_experts(selection_path: str) -> Tuple[List[Dict], str]:
    """
    Read a Stage 2 selection file.

    Returns
    -------
    (experts, model_type)
        ``experts`` is the list of ``{"layer", "expert", ...}`` dicts.
    """
    with open(selection_path) as fh:
        data = json.load(fh)
    if not data.get("experts"):
        raise ValueError(f"{selection_path} contains no selected experts.")
    return data["experts"], data.get("model_type", "qwen3_moe")
