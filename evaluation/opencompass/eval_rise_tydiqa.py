# TyDiQA-GoldP multilingual evaluation of a merged RISE checkpoint.
#
#   Copy this file into <opencompass>/opencompass/configs/ and run:
#       python run.py opencompass/configs/eval_rise_tydiqa.py -w outputs/rise_tydiqa
#
# Set MODEL_PATH to the directory produced by scripts/merge_experts.py, and
# run the same file with the base checkpoint to get the baseline row.

from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

# --- edit this -------------------------------------------------------------
MODEL_PATH = '/path/to/merged-rise-checkpoint'
ABBR = 'rise-bn-k128'
# ---------------------------------------------------------------------------

TARGET_LANGS = [
    'arabic', 'bengali', 'english', 'finnish', 'indonesian',
    'korean', 'russian', 'swahili', 'telugu',
]

NUM_GPUS = 1
MAX_OUT_LEN = 50          # GoldP answers are short spans
MAX_SEQ_LEN = 8192        # TyDiQA passages are long; raise this if prompts are dropped
BATCH_SIZE = 16

with read_base():
    from .datasets.tydiqa.tydiqa_gen_978d2a import tydiqa_datasets

# tydiqa_gen names each split 'tydiqa-goldp_<language>'.
datasets = [
    d for d in tydiqa_datasets
    if d.get('abbr', '').replace('tydiqa-goldp_', '') in TARGET_LANGS
]
assert datasets, 'No TyDiQA languages matched TARGET_LANGS.'

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr=ABBR,
        path=MODEL_PATH,
        model_kwargs=dict(
            tensor_parallel_size=NUM_GPUS,
            trust_remote_code=True,
        ),
        max_out_len=MAX_OUT_LEN,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        # Qwen3 enables "thinking" by default, which changes MGSM scores
        # completely. The paper evaluates with it off. Phi-3.5-MoE has no
        # such switch -- delete this block for that model.
        chat_template_kwargs=dict(enable_thinking=False),
        generation_kwargs=dict(temperature=0, top_p=1.0),
        run_cfg=dict(num_gpus=NUM_GPUS),
    )
]
