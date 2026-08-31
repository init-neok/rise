# Reproducing a result

This directory carries everything needed to turn a RISE run into numbers.
The training pipeline stops at a compact expert checkpoint, which no
evaluation harness can load on its own, so reproduction has three steps:

```
run_rise.sh  ->  expert_weights.pt  ->  merge_experts.py  ->  a normal HF checkpoint  ->  any harness
```

The reference target is the basic experiment from the paper: **adapt
Qwen3-30B-A3B to Bengali with K=128 experts, and check that Bengali improves
while the other nine languages hold.**

---

## 1. Train

```bash
bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

Produces `outputs/rise_qwen3_bn_k128/finetuned/expert_weights.pt` — only the
128 trained experts, a few hundred MB.

## 2. Merge

```bash
python scripts/merge_experts.py \
    --base_model /path/to/Qwen3-30B-A3B \
    --finetuned outputs/rise_qwen3_bn_k128/finetuned \
    --output_dir /path/to/Qwen3-30B-A3B-RISE-bn-k128
```

The result is an ordinary HuggingFace model directory. Nothing in it is
RISE-specific, so vLLM, OpenCompass, lm-eval-harness and
`AutoModelForCausalLM.from_pretrained` all take it directly.

It is also the full size of the base model (~60 GB). Keep
`expert_weights.pt`; treat the merged copy as a build artefact and delete it
once you have the numbers.

## 3. Evaluate

The paper's numbers come from [OpenCompass](https://github.com/open-compass/opencompass).
Two ready configs are in [`opencompass/`](opencompass/):

| File | Benchmark | Languages |
|---|---|---|
| `eval_rise_mgsm.py` | MGSM, first 200 problems per language | 10 |
| `eval_rise_tydiqa.py` | TyDiQA-GoldP | 9 |

```bash
git clone https://github.com/open-compass/opencompass
cd opencompass && pip install -e .

cp <rise>/evaluation/opencompass/eval_rise_mgsm.py opencompass/configs/
# edit MODEL_PATH and ABBR at the top of the file
python run.py opencompass/configs/eval_rise_mgsm.py -w outputs/rise_mgsm
```

The configs must sit inside `opencompass/configs/` — they pull the stock
dataset definitions through a relative `read_base()` import, which only
resolves from within that tree.

**Run each config twice**: once with `MODEL_PATH` pointing at the merged RISE
checkpoint, once at the untouched base model. A single row of numbers says
nothing; the claim is the difference between the two.

### What to expect

Bengali rises, the other languages do not fall. That second half is the
substance of the method — fine-tuning a MoE model on one language usually
costs you the others, and selecting experts by routing isolation is what
avoids it. If Bengali improves but English drops several points, something is
wrong with the selection, not with the training.

---

## Notes on matching the paper exactly

The configs here use stock OpenCompass classes so they run against an
unmodified install. Two differences from the authors' own setup are worth
knowing about:

**Thinking mode.** Qwen3 enables it by default and it changes MGSM scores
completely. Both configs set `chat_template_kwargs=dict(enable_thinking=False)`
to match the paper. Phi-3.5-MoE has no such switch — delete that block when
evaluating it.

**Over-long prompts.** Some TyDiQA passages exceed the model's context. The
authors used a local OpenCompass subclass that skips those prompts and scores
them as empty. Stock OpenCompass does not do this, so a few examples may be
truncated instead, which moves TyDiQA numbers slightly. `MAX_SEQ_LEN` is set
to 8192 in `eval_rise_tydiqa.py` to keep it small.

Expect the trend to reproduce; expect the third decimal place not to.

---

## Using a different harness

Nothing here is tied to OpenCompass. After step 2 you have a standard
checkpoint, so any of these work equally well:

```bash
# lm-evaluation-harness
lm_eval --model hf --model_args pretrained=/path/to/merged --tasks ...

# vLLM directly
vllm serve /path/to/merged
```

If you skip the merge and want to evaluate in-process instead, load the base
model and apply the experts to it:

```python
from transformers import AutoModelForCausalLM
from rise import load_expert_weights, load_selected_experts

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16", device_map="auto")
experts, model_type = load_selected_experts("outputs/.../finetuned/selected_experts.json")
load_expert_weights(model, experts, model_type, "outputs/.../finetuned/expert_weights.pt")
```
