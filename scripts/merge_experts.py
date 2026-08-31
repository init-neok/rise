#!/usr/bin/env python3
"""
Merge trained experts into a standalone checkpoint
==================================================
Stage 3 saves only the experts it trained, which keeps runs cheap but means
the result is not a model any evaluation harness can load on its own. This
script writes the two halves back together: base checkpoint + expert weights
-> an ordinary HuggingFace model directory.

The output is a plain checkpoint with no RISE-specific pieces, so vLLM,
OpenCompass, lm-eval-harness or `AutoModelForCausalLM.from_pretrained` all
take it directly.

Usage
-----
::

    python scripts/merge_experts.py \\
        --base_model /path/to/Qwen3-30B-A3B \\
        --finetuned outputs/rise_bn_k128/finetuned \\
        --output_dir /path/to/Qwen3-30B-A3B-RISE-bn-k128

``--finetuned`` is the directory Stage 3 produced; it must contain both
``expert_weights.pt`` and ``selected_experts.json``.

Note on disk: the merged model is the full size of the base checkpoint
(~60 GB for Qwen3-30B-A3B). The compact expert file is what you keep; the
merged copy is a build artefact you can delete after evaluating.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.model_freezer import load_expert_weights, load_selected_experts
from rise.moe import check_model_path

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge RISE expert weights into a standalone checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model", required=True,
                   help="The original MoE checkpoint the run started from.")
    p.add_argument("--finetuned", required=True,
                   help="Stage 3 output directory (expert_weights.pt + selected_experts.json).")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the merged checkpoint.")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES),
                   help="Weight dtype of the merged checkpoint.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_tokenizer", action="store_true",
                   help="Skip copying the tokenizer into the output directory.")
    return p.parse_args()


def main():
    args = parse_args()

    finetuned = Path(args.finetuned)
    weights_path = finetuned / "expert_weights.pt"
    selection_path = finetuned / "selected_experts.json"
    for path in (weights_path, selection_path):
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. --finetuned must point at a Stage 3 output "
                "directory. If you trained with --save_full_model there is "
                "nothing to merge: that run already wrote a full checkpoint."
            )

    selected, model_type = load_selected_experts(str(selection_path))
    print(f"Merging {len(selected)} experts [model_type={model_type}]")

    # Merge on CPU: the merged model never needs to run here, and keeping it
    # off the GPU means this works on a machine that cannot hold the model.
    args.base_model = check_model_path(args.base_model)
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=DTYPES[args.dtype],
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )

    load_expert_weights(model, selected, model_type, str(weights_path))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing merged checkpoint to {out_dir} ...")
    model.save_pretrained(out_dir, safe_serialization=True)

    if not args.no_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(out_dir)

    # Record what went into this checkpoint; a merged model is otherwise
    # indistinguishable from the base one.
    shutil.copyfile(selection_path, out_dir / "rise_selected_experts.json")
    with open(out_dir / "rise_merge_info.json", "w") as fh:
        json.dump(
            {
                "base_model": args.base_model,
                "finetuned_dir": str(finetuned.resolve()),
                "model_type": model_type,
                "num_experts_merged": len(selected),
                "dtype": args.dtype,
            },
            fh,
            indent=2,
        )

    total_gb = sum(f.stat().st_size for f in out_dir.rglob("*")) / 1024 ** 3
    print(f"\nMerged checkpoint ready: {out_dir.resolve()} ({total_gb:.1f} GB)")
    print("Load it like any other model:")
    print(f"    AutoModelForCausalLM.from_pretrained('{out_dir}')")


if __name__ == "__main__":
    main()
