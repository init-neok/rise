#!/usr/bin/env python3
"""
Stage 2 — Layer-Aware Expert Selection (RISE)
==============================================
Select the language-specific expert subnetwork from routing statistics
collected in Stage 1.

The selection algorithm (Algorithm 1) partitions MoE layers into three
functional zones and applies distinct scoring criteria per zone:

  Shallow / Deep → Specificity score (language-preference)
  Middle         → Overlap score (cross-lingual sharing)

See ``rise/expert_selector.py`` for the full mathematical details.

Usage
-----
Basic (Qwen3-30B-A3B, Bengali, budget K=128)::

    python scripts/select_experts.py \\
        --routing_stats outputs/routing_stats/routing_stats.json \\
        --target_language bn \\
        --num_experts 128 \\
        --output outputs/selected_experts.json

Phi-3.5-MoE-instruct, K=16::

    python scripts/select_experts.py \\
        --routing_stats outputs/routing_stats/routing_stats.json \\
        --target_language bn \\
        --num_experts 16 \\
        --shallow_budget 0.125 \\
        --middle_budget 0.6875 \\
        --output outputs/selected_experts_phi.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.expert_selector import RISEConfig, RISESelector


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the RISE layer-aware expert selection algorithm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    p.add_argument(
        "--routing_stats", required=True,
        help="Path to routing_stats.json from Stage 1.",
    )
    p.add_argument(
        "--target_language", required=True,
        help="ISO code of the target low-resource language (e.g. 'bn').",
    )
    p.add_argument(
        "--num_experts", type=int, required=True,
        help=(
            "Total expert budget K. "
            "Typical values: 128 for Qwen3-30B-A3B (2.1%%), 16 for Phi-3.5-MoE (3.1%%)."
        ),
    )
    p.add_argument(
        "--output", required=True,
        help="Output path for the selected_experts.json file.",
    )

    # ---- Layer zone boundaries ----
    p.add_argument(
        "--shallow_ratio", type=float, default=0.375,
        help="Fraction of total layers in the shallow zone.",
    )
    p.add_argument(
        "--middle_ratio", type=float, default=0.25,
        help="Fraction of total layers in the middle zone.",
    )

    # ---- Budget allocation ----
    p.add_argument(
        "--shallow_budget", type=float, default=0.35,
        help="Fraction of K allocated to the shallow zone.",
    )
    p.add_argument(
        "--middle_budget", type=float, default=0.25,
        help="Fraction of K allocated to the middle zone.",
    )

    # ---- Composite score ----
    p.add_argument(
        "--alpha", type=float, default=10.0,
        help="Activation-magnitude weighting factor α in composite scores (Eq. 8).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    config = RISEConfig(
        target_language=args.target_language,
        num_experts=args.num_experts,
        shallow_ratio=args.shallow_ratio,
        middle_ratio=args.middle_ratio,
        shallow_budget=args.shallow_budget,
        middle_budget=args.middle_budget,
        alpha=args.alpha,
    )

    selector = RISESelector(args.routing_stats, config)
    selector.select_and_save(args.output)


if __name__ == "__main__":
    main()
