#!/usr/bin/env python3
"""
Stage 1 — Routing replay
========================
Replay text through a frozen MoE model and record how often each expert is
activated, per layer and per language.  Writes ``routing_stats.json``, which
Stage 2 consumes.

At least two languages are required: RISE scores every expert relative to the
cross-language mean, so a single language carries no signal.  The usual setup
is the target language plus English as the high-resource reference.

Examples
--------
Offline, from your own text files (``bn.txt``, ``en.txt``)::

    python scripts/collect_routing_stats.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --texts_dir data/texts \\
        --languages bn en \\
        --output_dir outputs/routing_stats

Replaying OpenCompass predictions (one file per language)::

    python scripts/collect_routing_stats.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --predictions bn=preds/mgsm_bn.json en=preds/mgsm_en.json \\
        --output_dir outputs/routing_stats

Downloading TyDiQA-GoldP::

    python scripts/collect_routing_stats.py \\
        --model_path /path/to/Qwen3-30B-A3B \\
        --dataset tydiqa --languages bn en \\
        --output_dir outputs/routing_stats
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.data import (
    load_texts_from_dataset,
    load_texts_from_dir,
    load_texts_from_predictions,
)
from rise.moe import SUPPORTED_MODEL_TYPES
from rise.routing_stats import collect_routing_stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Replay text through a MoE model and record expert routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path", required=True,
                   help="Path to (or HuggingFace id of) the MoE checkpoint.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write routing_stats.json into.")

    src = p.add_argument_group("text source (pick one)")
    src.add_argument("--texts_dir", default=None,
                     help="Directory of '<lang>.txt' files, one example per line.")
    src.add_argument("--predictions", nargs="+", default=None, metavar="LANG=FILE",
                     help="OpenCompass predictions JSON per language, e.g. 'bn=preds/bn.json'.")
    src.add_argument("--dataset", default=None, choices=["tydiqa", "mgsm"],
                     help="Download a built-in dataset instead (needs network).")
    src.add_argument("--dataset_split", default=None,
                     help="Split to read. Default: tydiqa=train, mgsm=test.")
    src.add_argument("--languages", nargs="+", default=["bn", "en"],
                     help="Languages to replay. Ignored when --predictions is used.")

    run = p.add_argument_group("replay settings")
    run.add_argument("--max_samples", type=int, default=500,
                     help="Maximum texts per language.")
    run.add_argument("--batch_size", type=int, default=4)
    run.add_argument("--max_length", type=int, default=512)
    run.add_argument("--model_type", default="auto",
                     choices=["auto", *SUPPORTED_MODEL_TYPES],
                     help="MoE family; 'auto' reads it off the model config.")
    run.add_argument("--device", default="cuda",
                     help="'cuda', 'cuda:0', 'cpu', or 'auto' to shard across GPUs.")
    run.add_argument("--dtype", default="bfloat16",
                     choices=["bfloat16", "float16", "float32"])
    run.add_argument("--trust_remote_code", action="store_true",
                     help="Allow checkpoints that ship custom modeling code.")
    return p.parse_args()


def resolve_texts(args):
    """Turn whichever source was requested into {language: [texts]}."""
    chosen = [n for n, v in (
        ("--texts_dir", args.texts_dir),
        ("--predictions", args.predictions),
        ("--dataset", args.dataset),
    ) if v]
    if len(chosen) != 1:
        raise SystemExit(
            "Pick exactly one text source: --texts_dir, --predictions, or --dataset "
            f"(got {chosen or 'none'})."
        )

    if args.texts_dir:
        return load_texts_from_dir(args.texts_dir, args.languages, args.max_samples)

    if args.predictions:
        texts = {}
        for item in args.predictions:
            if "=" not in item:
                raise SystemExit(
                    f"--predictions expects 'LANG=FILE' pairs, got {item!r}."
                )
            lang, path = item.split("=", 1)
            texts[lang] = load_texts_from_predictions(path, args.max_samples)
        return texts

    return load_texts_from_dataset(
        args.dataset, args.languages, args.max_samples, split=args.dataset_split
    )


def main():
    args = parse_args()
    texts_by_language = resolve_texts(args)

    collect_routing_stats(
        model_path=args.model_path,
        texts_by_language=texts_by_language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_type=args.model_type,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
