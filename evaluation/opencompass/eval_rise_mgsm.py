# MGSM multilingual evaluation of a merged RISE checkpoint.
#
#   Copy this file into <opencompass>/opencompass/configs/ and run:
#       python run.py opencompass/configs/eval_rise_mgsm.py -w outputs/rise_mgsm
#
# Set MODEL_PATH to the directory produced by scripts/merge_experts.py, and
# run the same file with the base checkpoint to get the baseline row.

from mmengine.config import read_base
from opencompass.models import VLLMwithChatTemplate

# --- edit this -------------------------------------------------------------
MODEL_PATH = '/path/to/merged-rise-checkpoint'
ABBR = 'rise-bn-k128'
# ---------------------------------------------------------------------------

# The paper reports the first 200 problems of each language. MGSM has 250.
N_SAMPLES = 200
TARGET_LANGS = ['en', 'zh', 'es', 'fr', 'de', 'ru', 'ja', 'th', 'sw', 'bn']

NUM_GPUS = 1
MAX_OUT_LEN = 1024
MAX_SEQ_LEN = 2048
BATCH_SIZE = 16

with read_base():
    from .datasets.mgsm.mgsm_gen_d967bc import mgsm_datasets

# mgsm_gen ships one dataset per language, named 'mgsm_<lang>'.
datasets = [
    dict(d, reader_cfg=dict(d.get('reader_cfg', {}), test_range=f'[0:{N_SAMPLES}]'))
    for d in mgsm_datasets
    if d.get('abbr', '').replace('mgsm_', '') in TARGET_LANGS
]
assert datasets, 'No MGSM languages matched TARGET_LANGS.'

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
