"""
Stage 2 — Layer-Aware Expert Selection (Algorithm 1)
======================================================
Implements the RISE expert subnetwork selection strategy described in
Section 4.2 of the paper.

Three-zone strategy
-------------------
Given L MoE layers partitioned into three functional groups:

    L_shallow = [0,  L1]       (language-specific encoding)
    L_middle  = (L1, L2]       (cross-lingual semantic processing)
    L_deep    = (L2, L-1]      (language-specific generation)

we select:

  * Shallow / Deep  →  experts with high **specificity** for the target language λ*:

        S^(l)_{λ*,i}  =  a^(l)_{λ*,i}  /  ā^(l)_i

        where ā^(l)_i = mean over all languages of a^(l)_{λ,i}

  * Middle           →  experts with high **cross-lingual overlap**:

        O^(l)_i  =  1 / (1 + CV^(l)_i)

        where CV^(l)_i = σ(a^(l)_{:,i}) / μ(a^(l)_{:,i})

Composite scores (Eq. 8) weight relative preference by absolute magnitude:

    Spec(l, i, λ*) = S^(l)_{λ*,i} · (1 + α · a^(l)_{λ*,i})
    Ovlp(l, i)     = O^(l)_i       · (1 + α · ā^(l)_i)

Budget
------
Given a total budget of K experts, we allocate K_s, K_m, K_d to the three
zones according to predefined ratios (ρ_s, ρ_m, ρ_d) with ρ_s+ρ_m+ρ_d=1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RISEConfig:
    """Configuration for the RISE layer-aware expert selection algorithm."""

    # ---- Required ----
    target_language: str
    """ISO 639-1 code of the target low-resource language (e.g. ``"bn"``)."""

    num_experts: int
    """Total expert budget K (number of experts to select for training)."""

    # ---- Layer zone boundaries (fraction of total layers) ----
    shallow_ratio: float = 0.375
    """Fraction of layers assigned to the shallow zone (default: 37.5%)."""

    middle_ratio: float = 0.25
    """Fraction of layers assigned to the middle zone (default: 25%)."""

    # deep_ratio is implied: 1 - shallow_ratio - middle_ratio = 37.5%

    # ---- Budget allocation across zones (must sum to 1.0) ----
    shallow_budget: float = 0.35
    """Fraction of K allocated to shallow-zone experts."""

    middle_budget: float = 0.25
    """Fraction of K allocated to middle-zone experts."""

    # deep_budget is implied: 1 - shallow_budget - middle_budget = 40%

    # ---- Composite score hyperparameter ----
    alpha: float = 10.0
    """Activation-magnitude weighting factor α in Eq. (8)."""

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}.")
        for name in ("shallow_ratio", "middle_ratio", "shallow_budget", "middle_budget"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")
        if self.shallow_ratio + self.middle_ratio > 1.0:
            raise ValueError(
                "shallow_ratio + middle_ratio must not exceed 1.0 "
                f"(got {self.shallow_ratio} + {self.middle_ratio}); "
                "the remainder is the deep zone."
            )
        if self.shallow_budget + self.middle_budget > 1.0:
            raise ValueError(
                "shallow_budget + middle_budget must not exceed 1.0 "
                f"(got {self.shallow_budget} + {self.middle_budget}); "
                "the remainder goes to the deep zone."
            )

    @property
    def deep_ratio(self) -> float:
        """Fraction of layers in the deep zone (the leftover)."""
        return 1.0 - self.shallow_ratio - self.middle_ratio

    @property
    def deep_budget(self) -> float:
        """Fraction of K allocated to the deep zone (the leftover)."""
        return 1.0 - self.shallow_budget - self.middle_budget


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SelectedExpert:
    """A single expert chosen by RISE for fine-tuning."""

    layer: int
    """Layer index (0-indexed)."""

    expert: int
    """Expert index within the layer."""

    zone: str
    """Which zone this expert belongs to: ``"shallow"`` | ``"middle"`` | ``"deep"``."""

    score: float
    """Composite score used for ranking (Spec or Ovlp)."""

    specificity: Optional[float] = None
    """Raw specificity score S (non-None for shallow/deep experts)."""

    overlap: Optional[float] = None
    """Raw overlap score O (non-None for middle experts)."""


# ---------------------------------------------------------------------------
# RISE selector
# ---------------------------------------------------------------------------

class RISESelector:
    """
    Implements Algorithm 1: Language-specific Expert Subnetwork Selection.

    Parameters
    ----------
    routing_stats_path:
        Path to ``routing_stats.json`` produced by Stage 1.
    config:
        :class:`RISEConfig` instance specifying target language and budget.

    Example
    -------
    ::

        selector = RISESelector("outputs/routing_stats.json", config)
        selector.select_and_save("outputs/selected_experts.json")
    """

    def __init__(self, routing_stats_path: str, config: RISEConfig):
        with open(routing_stats_path) as f:
            raw = json.load(f)

        self.config = config
        self.model_cfg: Dict = raw["model_config"]
        self.num_layers: int = self.model_cfg["num_layers"]
        self.num_experts_per_layer: int = self.model_cfg["num_experts_per_layer"]
        self.model_type: str = self.model_cfg.get("model_type", "qwen3_moe")

        # freqs[lang]: np.ndarray of shape [num_layers, num_experts]
        self.freqs: Dict[str, np.ndarray] = self._load_frequencies(raw["languages"])

        if config.target_language not in self.freqs:
            raise ValueError(
                f"Target language '{config.target_language}' not found in "
                f"routing stats. Available: {list(self.freqs.keys())}"
            )

        # Per-layer score memos (populated lazily by the scoring helpers).
        self._mean_cache: Dict[int, np.ndarray] = {}
        self._spec_cache: Dict[int, np.ndarray] = {}
        self._ovlp_cache: Dict[int, np.ndarray] = {}

        # Partition layers into the three functional zones.
        L1 = round(self.num_layers * config.shallow_ratio)
        L2 = round(self.num_layers * (config.shallow_ratio + config.middle_ratio))
        self.shallow_layers: List[int] = list(range(0, L1))
        self.middle_layers:  List[int] = list(range(L1, L2))
        self.deep_layers:    List[int] = list(range(L2, self.num_layers))
        self.zones: Dict[str, List[int]] = {
            "shallow": self.shallow_layers,
            "middle":  self.middle_layers,
            "deep":    self.deep_layers,
        }

        if config.num_experts > self.num_layers * self.num_experts_per_layer:
            raise ValueError(
                f"Budget K={config.num_experts} exceeds the total expert pool "
                f"({self.num_layers} layers x {self.num_experts_per_layer} experts "
                f"= {self.num_layers * self.num_experts_per_layer})."
            )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_frequencies(
        self, languages_raw: Dict
    ) -> Dict[str, np.ndarray]:
        """Parse raw JSON into per-language frequency arrays."""
        result: Dict[str, np.ndarray] = {}
        for lang, layer_data in languages_raw.items():
            arr = np.zeros((self.num_layers, self.num_experts_per_layer), dtype=np.float32)
            for layer_str, expert_freqs in layer_data.items():
                l = int(layer_str)
                for expert_str, freq in expert_freqs.items():
                    arr[l, int(expert_str)] = float(freq)
            result[lang] = arr
        return result

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _mean_freq(self, layer: int) -> np.ndarray:
        """ā^(l)_i: mean activation frequency of expert i across all languages."""
        if layer not in self._mean_cache:
            self._mean_cache[layer] = np.mean(
                [f[layer] for f in self.freqs.values()], axis=0
            )  # [Ne]
        return self._mean_cache[layer]

    def _specificity(self, layer: int) -> np.ndarray:
        """
        S^(l)_{λ*,i} = a^(l)_{λ*,i} / ā^(l)_i

        Values > 1 indicate expert i is preferentially activated by the
        target language; values < 1 indicate the opposite.
        """
        if layer not in self._spec_cache:
            target = self.freqs[self.config.target_language][layer]  # [Ne]
            mean   = self._mean_freq(layer)
            self._spec_cache[layer] = target / np.where(mean > 0, mean, 1e-8)
        return self._spec_cache[layer]

    def _overlap(self, layer: int) -> np.ndarray:
        """
        O^(l)_i = 1 / (1 + CV^(l)_i),  CV = σ / μ  (coefficient of variation)

        High O → expert activated uniformly across languages → shared expert.
        Low  O → expert activation varies across languages → language-specific.
        """
        if layer not in self._ovlp_cache:
            layer_freqs = np.stack(
                [f[layer] for f in self.freqs.values()]
            )  # [num_langs, Ne]
            mu  = np.mean(layer_freqs, axis=0)
            std = np.std(layer_freqs, axis=0)
            cv  = std / np.where(mu > 0, mu, 1e-8)
            self._ovlp_cache[layer] = 1.0 / (1.0 + cv)
        return self._ovlp_cache[layer]

    def _composite_spec(self, layer: int) -> np.ndarray:
        """
        Spec(l, i, λ*) = S^(l)_{λ*,i} · (1 + α · a^(l)_{λ*,i})   [Eq. 8]
        """
        S    = self._specificity(layer)
        freq = self.freqs[self.config.target_language][layer]
        return S * (1.0 + self.config.alpha * freq)

    def _composite_ovlp(self, layer: int) -> np.ndarray:
        """
        Ovlp(l, i) = O^(l)_i · (1 + α · ā^(l)_i)   [Eq. 8]
        """
        O      = self._overlap(layer)
        mean_a = self._mean_freq(layer)
        return O * (1.0 + self.config.alpha * mean_a)

    # ------------------------------------------------------------------
    # Main selection
    # ------------------------------------------------------------------

    def select(self) -> List[SelectedExpert]:
        """
        Run Algorithm 1 and return the selected expert subnetwork.

        Returns
        -------
        List[SelectedExpert]
            All selected experts, sorted by (layer, expert).
        """
        cfg = self.config
        K   = cfg.num_experts
        budgets = {
            "shallow": int(K * cfg.shallow_budget),   # floor; deep absorbs the remainder
            "middle":  int(K * cfg.middle_budget),
        }
        budgets["deep"] = K - budgets["shallow"] - budgets["middle"]

        # A zone can be too small to absorb its share -- e.g. middle_ratio=0
        # leaves it with no layers at all. Spill the shortfall into the zones
        # that still have room, so the caller always gets K experts back.
        budgets = self._redistribute_budgets(budgets)

        selected: List[SelectedExpert] = []
        chosen:   set[Tuple[int, int]] = set()

        def _pick_top(
            layers: List[int],
            score_fn,
            budget: int,
            zone: str,
        ):
            """Collect candidates from all layers in this zone, rank, pick top."""
            candidates: List[Tuple[float, int, int]] = []  # (score, layer, expert)
            for l in layers:
                scores = score_fn(l)
                for i, s in enumerate(scores.tolist()):
                    if (l, i) not in chosen:
                        candidates.append((s, l, i))
            candidates.sort(key=lambda x: x[0], reverse=True)

            for s, l, i in candidates[:budget]:
                chosen.add((l, i))
                spec_val = float(self._specificity(l)[i]) if zone != "middle" else None
                ovlp_val = float(self._overlap(l)[i])     if zone == "middle" else None
                selected.append(
                    SelectedExpert(
                        layer=l,
                        expert=i,
                        zone=zone,
                        score=float(s),
                        specificity=spec_val,
                        overlap=ovlp_val,
                    )
                )

        # Phase 1: shallow — language-specific experts
        _pick_top(self.shallow_layers, self._composite_spec, budgets["shallow"], "shallow")
        # Phase 2: middle — cross-lingual shared experts
        _pick_top(self.middle_layers,  self._composite_ovlp, budgets["middle"], "middle")
        # Phase 3: deep — language-specific experts
        _pick_top(self.deep_layers,    self._composite_spec, budgets["deep"], "deep")

        return sorted(selected, key=lambda e: (e.layer, e.expert))

    def _redistribute_budgets(self, budgets: Dict[str, int]) -> Dict[str, int]:
        """
        Cap each zone's budget at its expert pool and spill the excess elsewhere.

        Without this, a zone smaller than its allocated share silently returns
        fewer experts and the run trains less than the requested K.
        """
        capacity = {
            zone: len(layers) * self.num_experts_per_layer
            for zone, layers in self.zones.items()
        }
        budgets = dict(budgets)

        overflow = 0
        for zone in budgets:
            excess = budgets[zone] - capacity[zone]
            if excess > 0:
                budgets[zone] = capacity[zone]
                overflow += excess

        while overflow > 0:
            # Zones that can still take more, largest headroom first.
            room = {z: capacity[z] - budgets[z] for z in budgets}
            open_zones = [z for z, r in room.items() if r > 0]
            if not open_zones:
                raise ValueError(
                    f"Budget K={self.config.num_experts} exceeds the capacity of "
                    "all three zones combined."
                )
            zone = max(open_zones, key=lambda z: room[z])
            take = min(room[zone], overflow)
            budgets[zone] += take
            overflow -= take

        return budgets

    def select_and_save(self, output_path: str) -> str:
        """
        Run selection and write results to a JSON file.

        Parameters
        ----------
        output_path:
            Destination file path (e.g. ``"outputs/selected_experts.json"``).

        Returns
        -------
        str
            The absolute path of the saved file.
        """
        experts = self.select()

        zone_counts: Dict[str, int] = {}
        for e in experts:
            zone_counts[e.zone] = zone_counts.get(e.zone, 0) + 1

        total_pool = self.num_layers * self.num_experts_per_layer
        output = {
            "method": "RISE",
            "model_type": self.model_type,
            "target_language": self.config.target_language,
            "total_experts_selected": len(experts),
            "total_expert_pool": total_pool,
            "selection_ratio": round(len(experts) / total_pool, 4),
            "zone_counts": zone_counts,
            "zone_boundaries": {
                zone: ([layers[0], layers[-1]] if layers else None)
                for zone, layers in self.zones.items()
            },
            "config": {
                "num_experts": self.config.num_experts,
                "alpha": self.config.alpha,
                "shallow_ratio": self.config.shallow_ratio,
                "middle_ratio": self.config.middle_ratio,
                "shallow_budget": self.config.shallow_budget,
                "middle_budget": self.config.middle_budget,
            },
            "experts": [
                {
                    "layer":  e.layer,
                    "expert": e.expert,
                    "zone":   e.zone,
                    "score":  round(e.score, 6),
                }
                for e in experts
            ],
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(output, f, indent=2)

        # Summary
        print(
            f"\nSelected {len(experts)} / {total_pool} experts "
            f"({100 * len(experts) / total_pool:.1f}%) "
            f"for language [{self.config.target_language}]:"
        )
        for zone in ("shallow", "middle", "deep"):
            print(f"  {zone:8s}: {zone_counts.get(zone, 0):4d} experts")
        print(f"Saved to: {out.resolve()}")
        return str(out.resolve())
