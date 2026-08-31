# Sample corpus

250 parallel Bengali/English lines, enough to run all three RISE stages
offline — no network, no `datasets` download.

```
bn.txt   250 lines   Bengali
en.txt   250 lines   English
```

The two files are **aligned line for line**: line *n* of `bn.txt` is a
translation of line *n* of `en.txt`. That alignment is the point. Specificity
scores an expert by how much the target language prefers it relative to the
cross-language mean, so any content difference between the two corpora would
show up as a routing difference and be mistaken for a language effect. With a
parallel corpus the content is held fixed and what remains is the language.

## Usage

```bash
TEXTS_DIR=examples/data bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
```

or directly:

```bash
python scripts/collect_routing_stats.py \
    --model_path /path/to/your-moe-model \
    --texts_dir examples/data --languages bn en \
    --output_dir outputs/routing_stats
```

## Scale

250 short questions exercise every code path and yield a usable selection, but
they are not the scale the paper reports. Those statistics come from several
hundred longer examples per language. For full runs, point `--texts_dir` at
your own corpus, replay OpenCompass predictions with `--predictions`, or use
`--dataset`.

## Provenance

The questions are the MGSM test split (`mgsm_bn.tsv`, `mgsm_en.tsv`), question
column only — answers are dropped, since routing replay and causal-LM
fine-tuning both need plain text.

MGSM comes from *Language Models are Multilingual Chain-of-Thought Reasoners*
(Shi et al., ICLR 2023), which human-translated 250 problems from GSM8K
(Cobbe et al., 2021) into ten languages.

- MGSM — CC BY-SA 4.0
- GSM8K — MIT

Redistributed here under those terms.
