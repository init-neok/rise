# RISE: Routing Isolation-guided Subnetwork Enhancement

> **Unveiling Language Routing Isolation in Multilingual MoE Models for
> Interpretable Subnetwork Adaptation**
>
> Accepted to **Findings of EMNLP 2026** · [arXiv:2604.03592](https://arxiv.org/abs/2604.03592)

Adapt a Mixture-of-Experts language model to a low-resource language by
fine-tuning **only the experts that language actually routes to** — roughly
2–3% of the parameters.

High- and low-resource languages activate largely disjoint expert sets. RISE
measures that separation from routing statistics, turns it into a small expert
subnetwork, and trains just those experts. The target language improves; the
rest of the model, and the other languages it serves, are left untouched.

---

## Install

```bash
git clone https://github.com/init-neok/rise.git
cd rise
bash setup_env.sh          # add --conda to use a conda env instead of a venv
```

The script creates the environment, installs the package, prints the versions
it resolved, and runs a CPU smoke test that exercises all three stages on a
tiny randomly-initialised MoE. If that test passes, the pipeline works.

If your cluster needs a specific CUDA build of PyTorch, install torch first —
`setup_env.sh` will not overwrite an existing one.

**transformers must be 4.x.** Version 5 fuses each MoE layer's experts into a
single batched module, so individual experts can no longer be addressed,
frozen, or saved — which is the whole mechanism RISE rests on. The dependency
is pinned to `<5`, and the code fails with an explicit message rather than an
obscure error if a 5.x install slips through. Verified against
transformers 4.57 with torch 2.9.

---

## Quick start

```bash
bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

Edit `MODEL_PATH` in that config; everything else has a working default.

To run without network access, point it at the bundled parallel Bengali/English
corpus in [`examples/data/`](examples/data/):

```bash
TEXTS_DIR=examples/data bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

The three stages run in order and each writes its output into `OUTPUT_DIR`:

| Stage | Script | Output |
|---|---|---|
| 1. Routing replay | `scripts/collect_routing_stats.py` | `routing_stats.json` |
| 2. Expert selection | `scripts/select_experts.py` | `selected_experts.json` |
| 3. Selective fine-tuning | `scripts/train_rise.py` | `expert_weights.pt` |

A fourth step turns that into something a benchmark can load:

| | Script | Output |
|---|---|---|
| Merge for evaluation | `scripts/merge_experts.py` | a standard HF checkpoint |

See [`evaluation/`](evaluation/) for the full recipe, including ready
OpenCompass configs for MGSM and TyDiQA.

A stage is skipped when its output already exists, so an interrupted run
resumes where it stopped. Pass `--force` to redo everything, or
`--stages 23` to run a subset.

Any setting can be overridden per run — exported variables win over the config
file:

```bash
NUM_EXPERTS=64 bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

---

## Method

### Stage 1 — Routing replay

Replay text through the frozen model and record how often the router dispatches
a token to each expert, per layer and per language:

$$a^{(l)}_{\lambda, i} = \frac{1}{T_\lambda} \sum_{t=1}^{T_\lambda} g^{(l)}_{t,i} \in [0, 1]$$

where $g^{(l)}_{t,i} = 1$ iff expert $i$ is among the router's top-k for token
$t$ at layer $l$. Padding positions are excluded — counting them would bias
each language by however long its sequences happen to tokenize.

**At least two languages are required.** Specificity is defined relative to the
cross-language mean, so a single language carries no signal. The usual setup is
the target language plus English as the high-resource reference.

### Stage 2 — Layer-aware expert selection

MoE layers split into three functional zones, each scored differently:

| Zone | Default depth | Routing behaviour | Criterion |
|---|---|---|---|
| Shallow | 0 – 37.5% | Language-specific encoding | **Specificity** |
| Middle | 37.5 – 62.5% | Cross-lingual semantics | **Overlap** |
| Deep | 62.5 – 100% | Language-specific generation | **Specificity** |

**Specificity** — how preferentially expert $i$ is activated by the target
language $\lambda^*$, against the mean $\bar{a}^{(l)}_i$ over all languages:

$$S^{(l)}_{\lambda^*, i} = \frac{a^{(l)}_{\lambda^*, i}}{\bar{a}^{(l)}_i}$$

**Overlap** — how uniformly expert $i$ is activated across languages, via the
coefficient of variation:

$$O^{(l)}_i = \frac{1}{1 + \widehat{CV}^{(l)}_i}, \qquad \widehat{CV}^{(l)}_i = \frac{\sigma(a^{(l)}_{\cdot, i})}{\mu(a^{(l)}_{\cdot, i})}$$

Both are ratios, so a rarely-used expert can score highly on noise alone. The
composite scores damp that by weighting each ratio with the absolute activation
magnitude:

$$\text{Spec}(l, i, \lambda^*) = S^{(l)}_{\lambda^*,i} \cdot \left(1 + \alpha \cdot a^{(l)}_{\lambda^*,i}\right), \qquad \text{Ovlp}(l, i) = O^{(l)}_i \cdot \left(1 + \alpha \cdot \bar{a}^{(l)}_i\right)$$

A budget of $K$ experts is split across the zones by the ratios
$(\rho_s, \rho_m, \rho_d)$. If a zone is too small to absorb its share, the
remainder spills into the zones that still have room, so a run always trains
exactly $K$ experts.

### Stage 3 — Selective fine-tuning

Freeze everything, unfreeze the $K$ selected experts, and train on
target-language text with the causal-LM objective:

$$\mathcal{L}(\Theta_{\text{train}}) = -\mathbb{E}_{x \sim \mathcal{D}_{\lambda^*}}\left[\sum_{t} \log P_\Theta(x_t \mid x_{<t})\right]$$

Only the selected expert tensors are written out — a few hundred MB rather than
a ~60 GB checkpoint. The base model on disk is never modified.

---

## Supported models

| Model | `model_type` | Layers | Experts/layer | Suggested K |
|---|---|---|---|---|
| Qwen3-30B-A3B | `qwen3_moe` | 48 | 128 | 128 (2.1%) |
| Qwen2-MoE | `qwen2_moe` | — | — | ~2% of pool |
| Phi-3.5-MoE-instruct | `phimoe` | 32 | 16 | 16 (3.1%) |

Adding a family is a one-file change: `rise/moe.py` holds every
architecture-specific detail (where the router gate lives, where the experts
live, which config field names the expert count).

---

## Text sources

The same three sources feed both the routing replay and the fine-tuning, so you
can reproduce the paper's setup or plug in your own corpus.

**Plain text files** (no network — use this on an offline cluster). One
example per line, one `<lang>.txt` per language. A 250-line parallel
Bengali/English corpus ships in [`examples/data/`](examples/data/) so the
pipeline runs out of the box:

```bash
TEXTS_DIR=examples/data bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

Those two files are aligned line for line, which matters: specificity measures
the target language against the cross-language mean, so any *content*
difference between the corpora would be read as a language difference. Holding
the content fixed isolates the language. 250 lines is enough to exercise the
pipeline; the paper's statistics come from several hundred longer examples per
language, so point `TEXTS_DIR` at your own corpus for full runs.

**OpenCompass predictions** — replays exactly the prompts the model was
evaluated on, which is how the paper's statistics were produced:

```bash
python scripts/collect_routing_stats.py \
    --model_path /path/to/Qwen3-30B-A3B \
    --predictions bn=preds/mgsm_bn.json en=preds/mgsm_en.json \
    --output_dir outputs/routing_stats
```

**A built-in dataset** (`tydiqa` = TyDiQA-GoldP, or `mgsm`), downloaded through
`datasets`:

```bash
DATASET=tydiqa bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

The split is chosen per dataset — `tydiqa` uses `train`, `mgsm` uses `test`.
Override with `DATASET_SPLIT` only deliberately: MGSM's `train` split is the
eight few-shot exemplars, not data.

---

## Key hyperparameters

| Setting | Default | Meaning |
|---|---|---|
| `NUM_EXPERTS` (K) | — | Total expert budget |
| `SHALLOW_RATIO` | 0.375 | Fraction of layers in the shallow zone |
| `MIDDLE_RATIO` | 0.25 | Fraction of layers in the middle zone |
| `SHALLOW_BUDGET` (ρ_s) | 0.35 | Share of K for shallow experts |
| `MIDDLE_BUDGET` (ρ_m) | 0.25 | Share of K for middle experts |
| `ALPHA` (α) | 10.0 | Activation-magnitude weighting |
| `LEARNING_RATE` | 2e-5 | — |
| `NUM_EPOCHS` | 3 | — |
| `TRAIN_BATCH_SIZE` × `GRAD_ACCUM` | 2 × 8 | Effective batch size 16 |

Deep-zone values are implied: `1 - shallow - middle`.

Phi-3.5-MoE has only 16 experts per layer, so it wants a different split —
see `configs/phi35_moe_bn.env`.

---

## Output

```
OUTPUT_DIR/
├── routing_stats/
│   └── routing_stats.json     # per-language activation frequencies
├── selected_experts.json      # the chosen subnetwork, with scores and zones
└── finetuned/
    ├── selected_experts.json  # copy — needed to load the weights back
    └── expert_weights.pt      # only the trained expert tensors
```

`expert_weights.pt` is a flat dict keyed by
`"layer{l}_expert{i}.{param_name}"`, so it can be inspected without
instantiating the architecture.

To evaluate, merge it back into the base model — the result is an ordinary
checkpoint that vLLM, OpenCompass or lm-eval-harness will load:

```bash
python scripts/merge_experts.py \
    --base_model /path/to/Qwen3-30B-A3B \
    --finetuned outputs/rise_bn_k128/finetuned \
    --output_dir /path/to/merged
```

Or apply the experts in-process, without writing a second copy to disk:

```python
from transformers import AutoModelForCausalLM
from rise import load_expert_weights, load_selected_experts

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="bfloat16", device_map="auto")
experts, model_type = load_selected_experts("outputs/.../finetuned/selected_experts.json")
load_expert_weights(model, experts, model_type, "outputs/.../finetuned/expert_weights.pt")
```

---

## Programmatic API

```python
from rise import (
    RISEConfig, RISESelector,
    RISETrainingArguments, train_rise,
    collect_routing_stats,
)

# Stage 1
collect_routing_stats(
    model_path=BASE_MODEL,
    texts_by_language={"bn": bn_texts, "en": en_texts},
    output_dir="outputs/routing_stats",
)

# Stage 2
RISESelector(
    "outputs/routing_stats/routing_stats.json",
    RISEConfig(target_language="bn", num_experts=128),
).select_and_save("outputs/selected_experts.json")

# Stage 3
train_rise(RISETrainingArguments(
    model_path=BASE_MODEL,
    selected_experts_path="outputs/selected_experts.json",
    output_dir="outputs/rise_bn_k128",
    texts=bn_texts,
))
```

---

## Project structure

```
rise/
├── run_rise.sh                # end-to-end pipeline
├── setup_env.sh               # environment + install + smoke test
│
├── rise/                      # core library
│   ├── moe.py                 # architecture introspection (gates, experts, counts)
│   ├── routing_stats.py       # Stage 1
│   ├── expert_selector.py     # Stage 2
│   ├── model_freezer.py       # freezing + compact expert checkpoints
│   ├── trainer.py             # Stage 3
│   └── data.py                # text sources
│
├── scripts/                   # CLI entry points, one per stage, plus merge_experts.py
├── configs/                   # example configs, one per model
├── examples/data/             # 250 parallel bn/en lines, for offline runs
├── evaluation/                # reproduction recipe + OpenCompass configs
└── tests/test_pipeline.py     # CPU smoke test, no GPU required
```

---

## Tests

```bash
python tests/test_pipeline.py     # or: pytest tests/test_pipeline.py
```

Builds a 4-layer, 8-expert model on CPU and checks the properties that matter:
padding stays out of the routing statistics, selection honours the budget and
the zone boundaries, a training step moves the selected experts **and nothing
else**, and the expert checkpoint survives a save/load roundtrip.

---

## Citation

```bibtex
@misc{zheng2026unveilinglanguageroutingisolation,
      title={Unveiling Language Routing Isolation in Multilingual MoE Models for Interpretable Subnetwork Adaptation},
      author={Kening Zheng and Wei-Chieh Huang and Jiahao Huo and Zhonghao Li and Henry Peng Zou and Yibo Yan and Xin Zou and Jungang Li and Junzhuo Li and Hanrong Zhang and Xuming Hu and Philip S. Yu},
      year={2026},
      eprint={2604.03592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.03592},
}
```

## License

MIT — see [LICENSE](LICENSE).
