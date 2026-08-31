#!/usr/bin/env python3
"""
Stage 3 — Selective expert fine-tuning
======================================
Fine-tune only the experts chosen in Stage 2, on target-language text.
Everything else in the model stays frozen, and the run writes out just the
trained expert tensors (``expert_weights.pt``) rather than a full checkpoint.

Examples
--------
Offline, from your own text file::

    python scripts/train_rise.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --selected_experts outputs/selected_experts.json \\
        --texts_file data/texts/bn.txt \\
        --output_dir outputs/rise_bn_k128

Downloading TyDiQA-GoldP Bengali::

    python scripts/train_rise.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --selected_experts outputs/selected_experts.json \\
        --dataset tydiqa --language bn \\
        --output_dir outputs/rise_bn_k128
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.data import load_texts_from_dataset, load_texts_from_predictions
from rise.trainer import RISETrainingArguments, train_rise


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune the RISE-selected expert subnetwork.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", required=True,
                   help="Path to (or HuggingFace id of) the MoE base model.")
    p.add_argument("--selected_experts", required=True,
                   help="selected_experts.json produced by Stage 2.")
    p.add_argument("--output_dir", required=True,
                   help="Where to write expert_weights.pt and training logs.")

    src = p.add_argument_group("training data (pick one)")
    src.add_argument("--texts_file", default=None,
                     help="Plain text file, one training example per line.")
    src.add_argument("--predictions", default=None,
                     help="OpenCompass predictions JSON to train on.")
    src.add_argument("--dataset", default=None, choices=["tydiqa", "mgsm"],
                     help="Download a built-in dataset instead (needs network).")
    src.add_argument("--dataset_split", default=None,
                     help="Split to read. Default: tydiqa=train, mgsm=test.")
    src.add_argument("--language", default="bn",
                     help="Target language code; used with --dataset.")
    src.add_argument("--max_samples", type=int, default=1000,
                     help="Maximum number of training examples.")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--num_epochs", type=float, default=3.0)
    opt.add_argument("--batch_size", type=int, default=2,
                     help="Per-device micro-batch size.")
    opt.add_argument("--gradient_accumulation_steps", type=int, default=8)
    opt.add_argument("--learning_rate", type=float, default=2e-5)
    opt.add_argument("--warmup_ratio", type=float, default=0.05)
    opt.add_argument("--weight_decay", type=float, default=0.0)
    opt.add_argument("--max_length", type=int, default=512)
    opt.add_argument("--seed", type=int, default=42)

    run = p.add_argument_group("runtime")
    run.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    run.add_argument("--device_map", default="auto",
                     help="'auto' shards the model across visible GPUs; 'cpu' for a dry run.")
    run.add_argument("--gradient_checkpointing", action="store_true",
                     help="Trade compute for memory when the base model barely fits.")
    run.add_argument("--trust_remote_code", action="store_true")
    run.add_argument("--save_full_model", action="store_true",
                     help="Save the entire model instead of just the trained experts.")
    return p.parse_args()


def resolve_texts(args):
    chosen = [n for n, v in (
        ("--texts_file", args.texts_file),
        ("--predictions", args.predictions),
        ("--dataset", args.dataset),
    ) if v]
    if len(chosen) != 1:
        raise SystemExit(
            "Pick exactly one training source: --texts_file, --predictions, or "
            f"--dataset (got {chosen or 'none'})."
        )

    if args.texts_file:
        lines = Path(args.texts_file).read_text(encoding="utf-8").splitlines()
        texts = [ln.strip() for ln in lines if ln.strip()][: args.max_samples]
        if not texts:
            raise SystemExit(f"{args.texts_file} contains no non-empty lines.")
        print(f"Loaded {len(texts)} training texts from {args.texts_file}")
        return texts

    if args.predictions:
        return load_texts_from_predictions(args.predictions, args.max_samples)

    by_lang = load_texts_from_dataset(
        args.dataset, [args.language], args.max_samples, split=args.dataset_split
    )
    return by_lang[args.language]


def main():
    args = parse_args()
    texts = resolve_texts(args)

    train_rise(
        RISETrainingArguments(
            model_path=args.model_path,
            selected_experts_path=args.selected_experts,
            output_dir=args.output_dir,
            texts=texts,
            max_length=args.max_length,
            num_epochs=args.num_epochs,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            seed=args.seed,
            precision=args.precision,
            device_map=args.device_map,
            gradient_checkpointing=args.gradient_checkpointing,
            trust_remote_code=args.trust_remote_code,
            save_expert_weights_only=not args.save_full_model,
        )
    )


if __name__ == "__main__":
    main()
