"""
Text sources
============
RISE needs plain strings twice: once per language for the Stage 1 routing
replay, and once for the target language in Stage 3.  This module is the
single place that turns a *source* into those strings.

Three sources are supported, in the order the CLIs try them:

``--texts_dir DIR``
    ``DIR/<lang>.txt``, one example per line.  No network, no dependencies --
    use this on an offline cluster, or to replay your own corpus.

``--predictions FILE``
    An OpenCompass predictions JSON.  This replays exactly the prompts the
    model was evaluated on, which is how the statistics in the paper were
    produced.

``--dataset {tydiqa,mgsm}``
    Downloads through :mod:`datasets`. Convenient, but needs network access
    and a HuggingFace cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = [
    "TYDIQA_LANGUAGES",
    "MGSM_LANGUAGES",
    "DEFAULT_SPLITS",
    "load_texts_from_dir",
    "load_texts_from_predictions",
    "load_texts_from_dataset",
]

#: TyDiQA-GoldP ("secondary_task"): ISO 639-1 code -> the name used in the
#: dataset. GoldP carries no ``language`` column; the language is the prefix
#: of each example id, e.g. ``"bengali--3540...-0"``.
TYDIQA_LANGUAGES = {
    "ar": "arabic",
    "bn": "bengali",
    "en": "english",
    "fi": "finnish",
    "id": "indonesian",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu",
}

#: MGSM ships one config per language code.
MGSM_LANGUAGES = ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")

#: Default split per dataset. These differ, and getting it wrong fails quietly:
#: MGSM's "train" split holds only the 8 few-shot exemplars, so replaying it
#: would build routing statistics out of eight sentences.
DEFAULT_SPLITS = {"tydiqa": "train", "mgsm": "test"}


def load_texts_from_dir(
    texts_dir: str,
    languages: Sequence[str],
    max_samples: int,
) -> Dict[str, List[str]]:
    """Read ``<texts_dir>/<lang>.txt`` for each language, one example per line."""
    root = Path(texts_dir)
    result: Dict[str, List[str]] = {}
    for lang in languages:
        path = root / f"{lang}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {path} for language {lang!r}. "
                "Each language needs a <lang>.txt file with one example per line."
            )
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        texts = [ln for ln in lines if ln][:max_samples]
        if not texts:
            raise ValueError(f"{path} contains no non-empty lines.")
        result[lang] = texts
        print(f"  [{lang}] {len(texts)} texts from {path}")
    return result


def _flatten_prompt(origin_prompt) -> str:
    """
    Normalise an OpenCompass ``origin_prompt`` into one string.

    It is either a plain string, or a chat-style list of ``{"role", "prompt"}``
    turns, depending on whether the eval ran in completion or chat mode.
    """
    if isinstance(origin_prompt, str):
        return origin_prompt
    if isinstance(origin_prompt, list):
        parts = [
            turn.get("prompt", "") if isinstance(turn, dict) else str(turn)
            for turn in origin_prompt
        ]
        return "\n".join(p for p in parts if p)
    return str(origin_prompt)


def load_texts_from_predictions(
    predictions_path: str,
    max_samples: int,
    include_prediction: bool = True,
) -> List[str]:
    """
    Extract replayable text from an OpenCompass predictions JSON.

    The file maps a sample index to ``{"origin_prompt": ..., "prediction": ...}``.
    By default prompt and prediction are concatenated, so the replay sees the
    generated tokens too -- that is where the decoder-side routing behaviour
    actually shows up.

    Parameters
    ----------
    include_prediction:
        Set ``False`` to replay prompts only.
    """
    with open(predictions_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"{predictions_path} is not an OpenCompass predictions file "
            "(expected a JSON object keyed by sample index)."
        )

    texts: List[str] = []
    for _, record in sorted(data.items(), key=lambda kv: str(kv[0])):
        if not isinstance(record, dict):
            continue
        text = _flatten_prompt(record.get("origin_prompt", ""))
        if include_prediction:
            text = (text + "\n" + str(record.get("prediction", ""))).strip()
        if text.strip():
            texts.append(text)
        if len(texts) >= max_samples:
            break

    if not texts:
        raise ValueError(f"No usable text found in {predictions_path}.")
    print(f"  {len(texts)} texts from {predictions_path}")
    return texts


def load_texts_from_dataset(
    dataset: str,
    languages: Sequence[str],
    max_samples: int,
    split: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Download a built-in dataset and return its texts per language.

    Parameters
    ----------
    split:
        ``None`` picks the right split for the dataset (see
        :data:`DEFAULT_SPLITS`). Override it only if you know what you want:
        MGSM's ``train`` split is the 8-example few-shot prompt, not data.
    """
    if dataset not in DEFAULT_SPLITS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Expected one of {sorted(DEFAULT_SPLITS)}."
        )
    if split is None:
        split = DEFAULT_SPLITS[dataset]
    print(f"  {dataset} split={split}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("Install the 'datasets' package: pip install datasets") from exc

    result: Dict[str, List[str]] = {}
    for lang in languages:
        print(f"  Loading {dataset} [{lang}] ...")
        if dataset == "tydiqa":
            if lang not in TYDIQA_LANGUAGES:
                raise ValueError(
                    f"TyDiQA has no language {lang!r}. "
                    f"Available: {sorted(TYDIQA_LANGUAGES)}"
                )
            name = TYDIQA_LANGUAGES[lang]
            ds = load_dataset(
                "google-research-datasets/tydiqa", "secondary_task", split=split
            )
            ds = ds.filter(lambda x: x["id"].split("-")[0] == name)
            texts = [f"{ex['question']}\n{ex['context']}" for ex in ds]
        elif dataset == "mgsm":
            if lang not in MGSM_LANGUAGES:
                raise ValueError(
                    f"MGSM has no language {lang!r}. Available: {sorted(MGSM_LANGUAGES)}"
                )
            ds = load_dataset("juletxara/mgsm", lang, split=split)
            texts = [ex["question"] for ex in ds]
        else:
            raise ValueError(f"Unknown dataset {dataset!r}. Expected 'tydiqa' or 'mgsm'.")

        texts = [t for t in texts if t and t.strip()][:max_samples]
        if not texts:
            raise ValueError(f"{dataset} [{lang}] yielded no usable text.")
        result[lang] = texts
        print(f"  [{lang}] {len(texts)} texts")
    return result
