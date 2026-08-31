#!/usr/bin/env python3
"""
End-to-end smoke test on a tiny randomly-initialised MoE
========================================================
Runs the whole RISE pipeline on a 4-layer / 8-expert model that fits on a
CPU in a couple of seconds, so the code path can be checked without a GPU
or a 30B checkpoint.

Run either way::

    python tests/test_pipeline.py
    pytest tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

# A login node can expose 100+ cores; letting torch spawn a thread per core
# makes this tiny model *slower* by orders of magnitude through contention.
torch.set_num_threads(min(4, torch.get_num_threads()))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.expert_selector import RISEConfig, RISESelector
from rise.model_freezer import (
    freeze_model_except_experts,
    load_expert_weights,
    load_selected_experts,
    save_expert_weights,
)
from rise.moe import get_expert_module, get_num_experts
from rise.routing_stats import RoutingStatCollector

NUM_LAYERS = 4
NUM_EXPERTS = 8
TOP_K = 2
VOCAB = 64


def build_tiny_moe():
    """A small Qwen3-MoE with random weights."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        decoder_sparse_step=1,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(config)


def make_batch(seed: int, batch: int = 2, seq: int = 12, pad: int = 3):
    """Random token ids with the last `pad` positions of one row masked out."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (batch, seq), generator=generator)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    attention_mask[0, -pad:] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def test_routing_replay_records_frequencies():
    model = build_tiny_moe().eval()
    assert get_num_experts(model.config) == NUM_EXPERTS

    collector = RoutingStatCollector(model)
    assert collector.top_k == TOP_K

    batch = make_batch(seed=1)
    collector.set_token_mask(batch["attention_mask"])
    with torch.no_grad():
        model(**batch)

    # Padding must not be counted: 2x12 positions minus 3 masked = 21.
    expected_tokens = int(batch["attention_mask"].sum())
    assert collector.total_tokens() == expected_tokens, (
        f"counted {collector.total_tokens()} tokens, expected {expected_tokens} "
        "(padding is leaking into the statistics)"
    )

    freqs = collector.activation_frequencies()
    assert len(freqs) == NUM_LAYERS
    for layer, per_expert in freqs.items():
        assert len(per_expert) == NUM_EXPERTS
        assert all(0.0 <= f <= 1.0 for f in per_expert.values())
        # Every token activates exactly top_k experts, so the row sums to top_k.
        assert abs(sum(per_expert.values()) - TOP_K) < 1e-6, layer

    collector.reset()
    assert collector.total_tokens() == 0
    collector.remove_hooks()
    print("PASS  stage 1: routing replay records masked, normalised frequencies")
    return model


def write_routing_stats(model, path: Path) -> Path:
    """Replay two synthetic 'languages' and save a Stage 1 file."""
    collector = RoutingStatCollector(model)
    stats = {
        "model_config": {
            "model_path": "tiny-moe",
            "model_type": "qwen3_moe",
            "num_layers": NUM_LAYERS,
            "num_experts_per_layer": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
        },
        "languages": {},
    }
    for lang, seed in (("bn", 11), ("en", 22)):
        collector.reset()
        for step in range(3):
            batch = make_batch(seed=seed + step)
            collector.set_token_mask(batch["attention_mask"])
            with torch.no_grad():
                model(**batch)
        stats["languages"][lang] = {
            str(layer): {str(e): f for e, f in per_expert.items()}
            for layer, per_expert in collector.activation_frequencies().items()
        }
    collector.remove_hooks()
    path.write_text(json.dumps(stats))
    return path


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def test_selection_respects_budget_and_zones(stats_path: Path, out_path: Path):
    budget = 10
    selector = RISESelector(
        str(stats_path),
        RISEConfig(target_language="bn", num_experts=budget),
    )
    experts = selector.select()

    assert len(experts) == budget, f"asked for {budget} experts, got {len(experts)}"
    assert len({(e.layer, e.expert) for e in experts}) == budget, "duplicate experts"

    for expert in experts:
        assert expert.layer in selector.zones[expert.zone], (
            f"expert on layer {expert.layer} labelled '{expert.zone}', "
            f"but that zone covers {selector.zones[expert.zone]}"
        )

    selector.select_and_save(str(out_path))
    saved = json.loads(out_path.read_text())
    assert saved["total_experts_selected"] == budget
    assert saved["model_type"] == "qwen3_moe"

    # A budget too large for one zone must spill over, not silently shrink.
    wide = RISESelector(
        str(stats_path),
        RISEConfig(target_language="bn", num_experts=budget, middle_ratio=0.0),
    )
    assert len(wide.select()) == budget, "empty middle zone swallowed its budget"

    print("PASS  stage 2: selection honours the budget, the zones, and zone overflow")
    return out_path


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def test_only_selected_experts_are_trained(model, selection_path: Path):
    """The defining invariant of RISE: no gradient reaches anything else."""
    selected, model_type = load_selected_experts(str(selection_path))
    trainable, total = freeze_model_except_experts(model, selected, model_type)
    assert 0 < trainable < total, (trainable, total)

    chosen = {(e["layer"], e["expert"]) for e in selected}
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1.0
    )
    batch = make_batch(seed=99)
    model.train()
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    optimizer.step()

    changed = {
        name for name, p in model.named_parameters()
        if not torch.equal(p.detach(), before[name])
    }
    assert changed, "nothing was updated at all"

    for name in changed:
        # Names look like model.layers.2.mlp.experts.5.gate_proj.weight
        parts = name.split(".")
        layer = int(parts[parts.index("layers") + 1])
        expert = int(parts[parts.index("experts") + 1])
        assert (layer, expert) in chosen, (
            f"parameter {name} changed but expert (layer={layer}, expert={expert}) "
            "was never selected -- freezing is leaking"
        )

    print(f"PASS  stage 3: {trainable:,}/{total:,} params trainable "
          f"({100 * trainable / total:.2f}%), only selected experts moved")


def test_expert_checkpoint_roundtrip(model, selection_path: Path, ckpt: Path):
    selected, model_type = load_selected_experts(str(selection_path))
    save_expert_weights(model, selected, model_type, str(ckpt))

    trained = {
        (e["layer"], e["expert"]): {
            name: p.detach().clone()
            for name, p in get_expert_module(
                model, e["layer"], e["expert"], model_type
            ).named_parameters()
        }
        for e in selected
    }

    # Overwrite the experts with noise, then restore them from the checkpoint.
    with torch.no_grad():
        for e in selected:
            for p in get_expert_module(
                model, e["layer"], e["expert"], model_type
            ).parameters():
                p.zero_()

    load_expert_weights(model, selected, model_type, str(ckpt))

    for e in selected:
        module = get_expert_module(model, e["layer"], e["expert"], model_type)
        for name, p in module.named_parameters():
            assert torch.equal(p.detach(), trained[(e["layer"], e["expert"])][name]), (
                f"layer{e['layer']}_expert{e['expert']}.{name} did not survive "
                "the save/load roundtrip"
            )

    size_mb = ckpt.stat().st_size / 1024 ** 2
    print(f"PASS  checkpoint: {len(selected)} experts roundtripped ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------

def main() -> int:
    print("Building a tiny MoE "
          f"({NUM_LAYERS} layers, {NUM_EXPERTS} experts, top-{TOP_K}) ...\n")
    model = test_routing_replay_records_frequencies()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stats_path = write_routing_stats(model, tmp / "routing_stats.json")
        selection = test_selection_respects_budget_and_zones(
            stats_path, tmp / "selected_experts.json"
        )
        test_only_selected_experts_are_trained(model, selection)
        test_expert_checkpoint_roundtrip(model, selection, tmp / "expert_weights.pt")

    print("\nAll stages passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
