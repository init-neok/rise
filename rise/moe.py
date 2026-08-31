"""
MoE model-family introspection
==============================
A single place that knows how each supported MoE family names its
router gates, its experts, and its expert count.  Every other module in
:mod:`rise` goes through these helpers, so adding a new model family is a
one-file change.

Supported families
------------------
=================  ==============================  =====================
``model_type``     gate module                     expert module
=================  ==============================  =====================
``qwen3_moe``      ``layers[l].mlp.gate``          ``layers[l].mlp.experts[i]``
``qwen2_moe``      ``layers[l].mlp.gate``          ``layers[l].mlp.experts[i]``
``phimoe``         ``layers[l].block_sparse_moe.gate``  ``layers[l].block_sparse_moe.experts[i]``
=================  ==============================  =====================
"""

from __future__ import annotations

from typing import Iterator, Tuple

import torch.nn as nn

__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "check_model_path",
    "detect_model_type",
    "get_num_experts",
    "get_num_experts_per_tok",
    "iter_moe_gates",
    "get_expert_module",
]

#: ``model_type`` → attribute on the decoder layer that holds the MoE block.
_MOE_BLOCK_ATTR = {
    "qwen3_moe": "mlp",
    "qwen2_moe": "mlp",
    "phimoe": "block_sparse_moe",
}

#: ``model_type`` → config field holding the per-layer expert count.
#: Qwen calls it ``num_experts``; Phi-3.5-MoE calls it ``num_local_experts``.
_NUM_EXPERTS_FIELD = {
    "qwen3_moe": "num_experts",
    "qwen2_moe": "num_experts",
    "phimoe": "num_local_experts",
}

SUPPORTED_MODEL_TYPES = tuple(_MOE_BLOCK_ATTR)


def check_model_path(model_path: str) -> str:
    """
    Validate a model location before handing it to ``from_pretrained``.

    A mistyped local path is the most common first failure, and transformers
    reports it as an opaque "Repo id must be in the form 'namespace/name'"
    error, because it falls back to treating the string as a Hub id.

    Only paths that cannot be Hub ids are rejected here — anything starting
    with ``/``, ``./`` or ``~``. A bare ``namespace/name`` is always passed
    through, so a valid Hub id is never wrongly refused.
    """
    import os

    expanded = os.path.expanduser(model_path)
    if os.path.isdir(expanded):
        if not os.path.isfile(os.path.join(expanded, "config.json")):
            raise FileNotFoundError(
                f"{model_path} has no config.json, so it is not a model directory."
            )
        return expanded

    if model_path.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            "Set MODEL_PATH (or --model_path) to a local checkpoint directory, "
            "or pass a HuggingFace model id such as 'Qwen/Qwen3-30B-A3B'."
        )

    return model_path  # treat as a Hub id and let transformers resolve it


def _check(model_type: str) -> str:
    if model_type not in _MOE_BLOCK_ATTR:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Expected one of {SUPPORTED_MODEL_TYPES}."
        )
    return model_type


def detect_model_type(config) -> str:
    """Read ``model_type`` off a HuggingFace config and validate it."""
    return _check(str(getattr(config, "model_type", "")).lower())


def get_num_experts(config) -> int:
    """
    Number of experts per MoE layer.

    The config field differs across families, which is why this helper
    exists: reading ``config.num_experts`` directly raises
    ``AttributeError`` on Phi-3.5-MoE.
    """
    model_type = detect_model_type(config)
    field = _NUM_EXPERTS_FIELD[model_type]
    if not hasattr(config, field):
        raise AttributeError(
            f"Config for {model_type!r} has no field {field!r}; "
            "cannot determine the expert count."
        )
    return int(getattr(config, field))


def get_num_experts_per_tok(config) -> int:
    """Router top-k, i.e. how many experts each token is dispatched to."""
    if not hasattr(config, "num_experts_per_tok"):
        raise AttributeError(
            "Config has no 'num_experts_per_tok'; cannot determine router top-k."
        )
    return int(config.num_experts_per_tok)


def _moe_block(layer: nn.Module, model_type: str):
    """Return the MoE block of a decoder layer, or ``None`` if it is dense."""
    block = getattr(layer, _MOE_BLOCK_ATTR[model_type], None)
    # Some architectures interleave dense FFN layers among MoE layers; those
    # have the attribute but no router, so check for the gate as well.
    if block is None or not hasattr(block, "gate"):
        return None
    return block


def iter_moe_gates(
    model: nn.Module, model_type: str
) -> Iterator[Tuple[int, nn.Module]]:
    """
    Yield ``(layer_index, gate_module)`` for every MoE layer of ``model``.

    The gate is the router projection.  It receives flattened hidden states
    ``[T, D]`` and returns routing logits ``[T, num_experts]`` *before* the
    top-k selection, which is exactly what Stage 1 needs to hook.
    """
    _check(model_type)
    for idx, layer in enumerate(model.model.layers):
        block = _moe_block(layer, model_type)
        if block is not None:
            yield idx, block.gate


def _require_addressable_experts(experts, layer: int) -> None:
    """
    Fail clearly when the expert container cannot be indexed per expert.

    transformers 5.x fuses each MoE layer's experts into a single batched
    module holding stacked weight tensors, so there is no per-expert submodule
    to select, freeze, or save. RISE is built on that addressability, hence the
    ``transformers < 5`` pin. Without this check the failure surfaces much
    later as an opaque TypeError.
    """
    if isinstance(experts, (nn.ModuleList, nn.Sequential, list, tuple)):
        return
    try:
        import transformers

        version = transformers.__version__
    except Exception:  # pragma: no cover - transformers is a hard dependency
        version = "unknown"
    raise TypeError(
        f"Layer {layer} exposes its experts as {type(experts).__name__}, which "
        "cannot be indexed per expert. transformers 5.x fuses MoE experts into "
        "one batched module, and RISE needs to address them individually.\n"
        f"Installed transformers: {version}. Install a 4.x release:\n"
        '    pip install "transformers>=4.42,<5"'
    )


def get_expert_module(
    model: nn.Module, layer: int, expert: int, model_type: str
) -> nn.Module:
    """Return the ``nn.Module`` of one ``(layer, expert)`` pair."""
    _check(model_type)
    block = _moe_block(model.model.layers[layer], model_type)
    if block is None:
        raise ValueError(f"Layer {layer} of this {model_type} model is not an MoE layer.")
    _require_addressable_experts(block.experts, layer)
    return block.experts[expert]
