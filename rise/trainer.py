"""
Stage 3 — Selective expert fine-tuning
======================================
Fine-tune only the RISE-selected experts on target-language text with the
standard causal-LM objective:

.. math::

    \\mathcal{L}(\\Theta_{\\text{train}})
        = -\\mathbb{E}_{x \\sim \\mathcal{D}_{\\lambda^*}}
          \\Big[ \\sum_t \\log P_\\Theta(x_t \\mid x_{<t}) \\Big]

Every parameter outside :math:`\\Theta_{\\text{train}}` stays frozen for the
whole run.  Training itself is an ordinary HuggingFace
:class:`~transformers.Trainer` loop; the selectivity lives entirely in the
freezing step.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .moe import check_model_path
from .model_freezer import (
    freeze_model_except_experts,
    load_selected_experts,
    save_expert_weights,
)

__all__ = ["RISETrainingArguments", "build_causal_lm_dataset", "train_rise"]


@dataclass
class RISETrainingArguments:
    """Everything needed for one RISE fine-tuning run."""

    # ---- Paths ----
    model_path: str
    """Path to (or HuggingFace id of) the MoE base model."""

    selected_experts_path: str
    """Path to the ``selected_experts.json`` produced by Stage 2."""

    output_dir: str
    """Directory for training artefacts and the expert checkpoint."""

    # ---- Data ----
    texts: Optional[List[str]] = None
    """Target-language training strings. Ignored if a Dataset is passed to
    :func:`train_rise` directly."""

    max_length: int = 512
    """Truncation length for tokenization."""

    # ---- Optimisation ----
    num_epochs: float = 3.0
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    """Effective batch size = ``per_device_batch_size × gradient_accumulation_steps``."""

    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    logging_steps: int = 10
    seed: int = 42

    # ---- Runtime ----
    precision: str = "bf16"
    """``"bf16"`` (recommended on A100/H100/H200), ``"fp16"``, or ``"fp32"``."""

    device_map: str = "auto"
    """Passed to ``from_pretrained``. ``"auto"`` shards a large model across GPUs."""

    gradient_checkpointing: bool = False
    """Trades compute for memory. Useful when the base model barely fits."""

    trust_remote_code: bool = False
    """Needed for checkpoints that ship custom modeling code."""

    # ---- Checkpointing ----
    save_expert_weights_only: bool = True
    """Save just the trained expert tensors (recommended) instead of the full model."""

    def __post_init__(self) -> None:
        if self.precision not in ("bf16", "fp16", "fp32"):
            raise ValueError(
                f"precision must be 'bf16', 'fp16' or 'fp32', got {self.precision!r}."
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.precision]


def build_causal_lm_dataset(
    texts: List[str],
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """
    Tokenize plain strings into a causal-LM dataset.

    Sequences are left unpadded here; the collator pads each batch and masks
    the padding out of the loss, so no compute is wasted on a global max length.
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("No non-empty training texts were provided.")

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
    )


def train_rise(
    args: RISETrainingArguments,
    train_dataset: Optional[Dataset] = None,
) -> str:
    """
    Run the full Stage 3 pipeline and return the output directory.

    Steps: load the Stage 2 selection, load the base model, freeze everything
    but the selected experts, tokenize, train, and write the expert checkpoint.
    """
    # 1. Which experts are we training?
    selected_experts, model_type = load_selected_experts(args.selected_experts_path)
    print(f"RISE: fine-tuning {len(selected_experts)} experts [model_type={model_type}]")

    # 2. Base model
    args.model_path = check_model_path(args.model_path)
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # 3. Freeze everything except the selected subnetwork
    trainable, total = freeze_model_except_experts(model, selected_experts, model_type)
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.3f}%)"
    )

    # 4. Data
    if train_dataset is None:
        if not args.texts:
            raise ValueError("Pass either `args.texts` or an explicit `train_dataset`.")
        print(f"Tokenizing {len(args.texts)} training examples ...")
        train_dataset = build_causal_lm_dataset(
            args.texts, tokenizer, max_length=args.max_length
        )
    print(f"Training dataset: {len(train_dataset)} examples")

    # 5. Trainer
    hf_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        seed=args.seed,
        save_strategy="no",   # the expert checkpoint is written by hand below
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting RISE fine-tuning ...")
    trainer.train()

    # 6. Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the selection next to the weights: loading them back requires both.
    shutil.copyfile(args.selected_experts_path, out_dir / "selected_experts.json")

    if args.save_expert_weights_only:
        save_expert_weights(
            model, selected_experts, model_type, str(out_dir / "expert_weights.pt")
        )
    else:
        model.save_pretrained(out_dir / "full_model")
        tokenizer.save_pretrained(out_dir / "full_model")
        print(f"Full model saved to: {out_dir / 'full_model'}")

    print(f"\nRISE training complete. Outputs in: {out_dir.resolve()}")
    return str(out_dir.resolve())
