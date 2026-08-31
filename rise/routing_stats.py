"""
Stage 1 — Routing replay / statistics collection
================================================
Replay text through a frozen MoE model and record, for every layer ``l``
and expert ``i``, how often the router dispatches a token to that expert:

.. math::

    a^{(l)}_{\\lambda, i}
        = \\frac{1}{T_\\lambda} \\sum_{t=1}^{T_\\lambda} g^{(l)}_{t,i}
        \\in [0, 1]

where :math:`g^{(l)}_{t,i} = 1` iff expert ``i`` is among the router's
top-k for token ``t`` at layer ``l``, and :math:`T_\\lambda` is the number
of *real* (non-padding) tokens seen for language :math:`\\lambda`.

The result is written to ``routing_stats.json`` and consumed by Stage 2
(:mod:`rise.expert_selector`).

Padding is excluded
-------------------
Counting padding tokens would bias every language's distribution toward
whatever the router does with the pad embedding, and would do so unevenly
because languages tokenize to different lengths.  The collector therefore
masks each batch down to its real tokens before accumulating.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .moe import (
    check_model_path,
    detect_model_type,
    get_num_experts,
    get_num_experts_per_tok,
    iter_moe_gates,
)

__all__ = ["RoutingStatCollector", "collect_routing_stats"]


class RoutingStatCollector:
    """
    Accumulate expert activation counts by hooking every router gate.

    Usage::

        collector = RoutingStatCollector(model)
        for batch in batches:
            collector.set_token_mask(batch["attention_mask"])
            with torch.no_grad():
                model(**batch)          # hooks fire during the forward pass
        freqs = collector.activation_frequencies()
        collector.reset()               # clear before the next language
        collector.remove_hooks()        # cleanup when finished

    Parameters
    ----------
    model:
        A loaded MoE causal LM.
    model_type:
        ``"auto"`` to read it off ``model.config``, or an explicit family name.
    """

    def __init__(self, model: nn.Module, model_type: str = "auto"):
        self.model = model
        self.model_type = (
            detect_model_type(model.config) if model_type == "auto" else model_type
        )
        self.num_experts = get_num_experts(model.config)
        self.top_k = get_num_experts_per_tok(model.config)

        # counts[layer] = int64 tensor [num_experts]; totals[layer] = int token count
        self._counts: Dict[int, torch.Tensor] = {}
        self._totals: Dict[int, int] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._token_mask: Optional[torch.Tensor] = None

        self._install_hooks()
        if not self._counts:
            raise RuntimeError(
                f"No MoE router gates found in this {self.model_type} model. "
                "Is it really a Mixture-of-Experts checkpoint?"
            )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        for layer_idx, gate in iter_moe_gates(self.model, self.model_type):
            self._counts[layer_idx] = torch.zeros(self.num_experts, dtype=torch.int64)
            self._totals[layer_idx] = 0
            self._hooks.append(
                gate.register_forward_hook(self._make_hook(layer_idx))
            )

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: tuple, outputs) -> None:
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 3:  # [B, S, Ne] -> [B*S, Ne]
                logits = logits.reshape(-1, logits.size(-1))

            # Drop padding positions. The MoE block flattens hidden states as
            # [B, S, D] -> [B*S, D], so a flattened attention mask lines up
            # one-to-one with the rows of `logits`.
            mask = self._token_mask
            if mask is not None:
                if mask.numel() != logits.size(0):
                    raise RuntimeError(
                        f"Token mask has {mask.numel()} entries but the router "
                        f"saw {logits.size(0)} tokens at layer {layer_idx}."
                    )
                logits = logits[mask.to(logits.device)]

            if logits.size(0) == 0:
                return

            # float() because top-k on bf16 can tie-break inconsistently
            selected = torch.topk(logits.float(), k=self.top_k, dim=-1).indices

            counts = torch.bincount(
                selected.reshape(-1), minlength=self.num_experts
            )
            self._counts[layer_idx] += counts.to(self._counts[layer_idx].device)
            self._totals[layer_idx] += selected.size(0)

        return hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_token_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        """
        Tell the collector which positions of the next forward pass are real.

        Call this immediately before each ``model(**batch)``.  Pass ``None``
        to count every position (only correct when nothing is padded).
        """
        self._token_mask = (
            None if attention_mask is None else attention_mask.reshape(-1).bool()
        )

    def reset(self) -> None:
        """Zero all counters. Call between languages."""
        for layer_idx in self._counts:
            self._counts[layer_idx].zero_()
            self._totals[layer_idx] = 0
        self._token_mask = None

    def remove_hooks(self) -> None:
        """Unregister every hook. Call when collection is finished."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def total_tokens(self) -> int:
        """Number of real tokens accumulated so far (as seen by layer 0)."""
        return next(iter(self._totals.values())) if self._totals else 0

    def activation_frequencies(self) -> Dict[int, Dict[int, float]]:
        """
        Return ``{layer: {expert: frequency}}`` with frequency in ``[0, 1]``.

        Each layer's frequencies sum to ``top_k`` rather than 1, because every
        token activates ``top_k`` experts.
        """
        result: Dict[int, Dict[int, float]] = {}
        for layer_idx, counts in self._counts.items():
            total = max(self._totals[layer_idx], 1)
            result[layer_idx] = {
                i: float(c) / total for i, c in enumerate(counts.tolist())
            }
        return result


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_routing_stats(
    model_path: str,
    texts_by_language: Dict[str, Sequence[str]],
    output_dir: str,
    max_samples: int = 500,
    batch_size: int = 4,
    max_length: int = 512,
    model_type: str = "auto",
    device: str = "cuda",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> str:
    """
    Replay each language's texts through the model and save routing statistics.

    Parameters
    ----------
    model_path:
        Path to (or HuggingFace id of) the MoE checkpoint.
    texts_by_language:
        ``{"bn": [...], "en": [...]}`` — plain strings, one per example.
        At least two languages are required: specificity is defined relative
        to the cross-language mean, so a single language carries no signal.
    output_dir:
        Directory to write ``routing_stats.json`` into.
    max_samples:
        Cap on the number of texts used per language.
    batch_size, max_length:
        Inference batching and truncation.
    model_type:
        ``"auto"`` or an explicit family name.
    device:
        ``"cuda"``, ``"cuda:0"``, ``"cpu"``, or ``"auto"`` for multi-GPU sharding.
    dtype:
        Torch dtype for the weights (``"bfloat16"``, ``"float16"``, ``"float32"``).
    trust_remote_code:
        Forwarded to ``from_pretrained``; needed for checkpoints that ship
        custom modeling code (e.g. some Phi-3.5-MoE releases).

    Returns
    -------
    str
        Absolute path of the written ``routing_stats.json``.
    """
    if len(texts_by_language) < 2:
        raise ValueError(
            "Routing statistics need at least two languages "
            f"(got {list(texts_by_language)}). RISE scores every expert "
            "relative to the cross-language mean, which is undefined for one."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = getattr(torch, dtype)

    model_path = check_model_path(model_path)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    collector = RoutingStatCollector(model, model_type=model_type)
    print(
        f"Hooked {len(collector._counts)} MoE layers "
        f"({collector.num_experts} experts each, top-{collector.top_k} routing)."
    )

    stats: Dict[str, Dict] = {
        "model_config": {
            "model_path": model_path,
            "model_type": collector.model_type,
            "num_layers": int(model.config.num_hidden_layers),
            "num_experts_per_layer": collector.num_experts,
            "num_experts_per_tok": collector.top_k,
        },
        "languages": {},
    }

    for lang, all_texts in texts_by_language.items():
        texts = [t for t in list(all_texts)[:max_samples] if t and t.strip()]
        if not texts:
            raise ValueError(f"No non-empty texts supplied for language {lang!r}.")

        print(f"\n[{lang}] replaying {len(texts)} texts ...")
        collector.reset()

        for start in tqdm(range(0, len(texts), batch_size), desc=lang, unit="batch"):
            batch = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            batch = {k: v.to(model.device) for k, v in batch.items()}
            collector.set_token_mask(batch.get("attention_mask"))
            model(**batch)

        freqs = collector.activation_frequencies()
        stats["languages"][lang] = {
            str(layer): {str(e): f for e, f in expert_freqs.items()}
            for layer, expert_freqs in freqs.items()
        }
        print(f"  {collector.total_tokens():,} tokens over {len(freqs)} MoE layers.")

    collector.remove_hooks()

    out_file = out_dir / "routing_stats.json"
    with open(out_file, "w") as fh:
        json.dump(stats, fh)
    print(f"\nRouting statistics written to: {out_file.resolve()}")
    return str(out_file.resolve())
