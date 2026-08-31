#!/usr/bin/env bash
# =============================================================================
# run_rise.sh -- the whole RISE pipeline, end to end
# =============================================================================
#
#   Stage 1  routing replay      -> routing_stats.json
#   Stage 2  expert selection    -> selected_experts.json
#   Stage 3  selective fine-tune -> expert_weights.pt
#
# Usage
#   bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
#   MODEL_PATH=/path/to/model bash run_rise.sh
#
# Every setting is an environment variable, so a config file is just a list of
# them. Anything already exported wins over the config file, which makes
# one-off overrides easy:
#
#   NUM_EXPERTS=64 bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
#
# Stages are skipped when their output already exists, so an interrupted run
# resumes where it stopped. Pass --force to redo everything.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG=""
FORCE=0
STAGES="123"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIG="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        --stages)  STAGES="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Load the config without clobbering variables the caller already exported.
if [[ -n "$CONFIG" ]]; then
    [[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        key="${line%%=*}"; value="${line#*=}"
        key="$(echo "$key" | tr -d '[:space:]')"
        [[ -n "${!key-}" ]] || export "$key=$value"
    done < "$CONFIG"
    echo "Config: $CONFIG"
fi

# ---- Settings and defaults --------------------------------------------------
: "${MODEL_PATH:?Set MODEL_PATH (path to the MoE checkpoint) or pass --config}"

TARGET_LANG="${TARGET_LANG:-bn}"
REFERENCE_LANGS="${REFERENCE_LANGS:-en}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"

SHALLOW_RATIO="${SHALLOW_RATIO:-0.375}"
MIDDLE_RATIO="${MIDDLE_RATIO:-0.25}"
SHALLOW_BUDGET="${SHALLOW_BUDGET:-0.35}"
MIDDLE_BUDGET="${MIDDLE_BUDGET:-0.25}"
ALPHA="${ALPHA:-10.0}"

DATASET="${DATASET:-tydiqa}"
# Empty means "let the loader pick": tydiqa=train, mgsm=test. MGSM's train
# split is only the 8 few-shot exemplars, so it must not be the default.
DATASET_SPLIT="${DATASET_SPLIT:-}"
TEXTS_DIR="${TEXTS_DIR:-}"
PREDICTIONS="${PREDICTIONS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1000}"

DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-bf16}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/rise_${TARGET_LANG}_k${NUM_EXPERTS}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

STATS_DIR="$OUTPUT_DIR/routing_stats"
STATS_FILE="$STATS_DIR/routing_stats.json"
SELECTION_FILE="$OUTPUT_DIR/selected_experts.json"
TRAIN_DIR="$OUTPUT_DIR/finetuned"

TRC_FLAG=()
[[ "$TRUST_REMOTE_CODE" == "1" ]] && TRC_FLAG=(--trust_remote_code)

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

banner() { printf '\n=== %s ===\n' "$1"; }

banner "RISE"
cat <<SUMMARY
  model      $MODEL_PATH
  target     $TARGET_LANG   (reference: $REFERENCE_LANGS)
  budget     K=$NUM_EXPERTS
  output     $OUTPUT_DIR
SUMMARY

# ---- Stage 1: routing replay ------------------------------------------------
if [[ "$STAGES" == *1* ]]; then
    if [[ -f "$STATS_FILE" && "$FORCE" -eq 0 ]]; then
        banner "Stage 1 skipped (found $STATS_FILE)"
    else
        banner "Stage 1: routing replay"
        SOURCE_ARGS=()
        if [[ -n "$PREDICTIONS" ]]; then
            # Space-separated LANG=FILE pairs.
            read -r -a PRED_ARR <<< "$PREDICTIONS"
            SOURCE_ARGS=(--predictions "${PRED_ARR[@]}")
        elif [[ -n "$TEXTS_DIR" ]]; then
            SOURCE_ARGS=(--texts_dir "$TEXTS_DIR"
                         --languages "$TARGET_LANG" $REFERENCE_LANGS)
        else
            SOURCE_ARGS=(--dataset "$DATASET"
                         --languages "$TARGET_LANG" $REFERENCE_LANGS)
            [[ -n "$DATASET_SPLIT" ]] && SOURCE_ARGS+=(--dataset_split "$DATASET_SPLIT")
        fi

        "$PYTHON" scripts/collect_routing_stats.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$STATS_DIR" \
            --max_samples "$MAX_SAMPLES" \
            --batch_size "$REPLAY_BATCH_SIZE" \
            --max_length "$MAX_LENGTH" \
            --device "$DEVICE" \
            "${SOURCE_ARGS[@]}" "${TRC_FLAG[@]}"
    fi
fi

# ---- Stage 2: expert selection ----------------------------------------------
if [[ "$STAGES" == *2* ]]; then
    if [[ -f "$SELECTION_FILE" && "$FORCE" -eq 0 ]]; then
        banner "Stage 2 skipped (found $SELECTION_FILE)"
    else
        banner "Stage 2: expert selection"
        "$PYTHON" scripts/select_experts.py \
            --routing_stats "$STATS_FILE" \
            --target_language "$TARGET_LANG" \
            --num_experts "$NUM_EXPERTS" \
            --shallow_ratio "$SHALLOW_RATIO" \
            --middle_ratio "$MIDDLE_RATIO" \
            --shallow_budget "$SHALLOW_BUDGET" \
            --middle_budget "$MIDDLE_BUDGET" \
            --alpha "$ALPHA" \
            --output "$SELECTION_FILE"
    fi
fi

# ---- Stage 3: selective fine-tuning -----------------------------------------
if [[ "$STAGES" == *3* ]]; then
    if [[ -f "$TRAIN_DIR/expert_weights.pt" && "$FORCE" -eq 0 ]]; then
        banner "Stage 3 skipped (found $TRAIN_DIR/expert_weights.pt)"
    else
        banner "Stage 3: selective fine-tuning"
        TRAIN_SOURCE=()
        if [[ -n "$TEXTS_DIR" ]]; then
            TRAIN_SOURCE=(--texts_file "$TEXTS_DIR/$TARGET_LANG.txt")
        else
            TRAIN_SOURCE=(--dataset "$DATASET" --language "$TARGET_LANG")
            [[ -n "$DATASET_SPLIT" ]] && TRAIN_SOURCE+=(--dataset_split "$DATASET_SPLIT")
        fi

        "$PYTHON" scripts/train_rise.py \
            --model_path "$MODEL_PATH" \
            --selected_experts "$SELECTION_FILE" \
            --output_dir "$TRAIN_DIR" \
            --max_samples "$TRAIN_MAX_SAMPLES" \
            --num_epochs "$NUM_EPOCHS" \
            --batch_size "$TRAIN_BATCH_SIZE" \
            --gradient_accumulation_steps "$GRAD_ACCUM" \
            --learning_rate "$LEARNING_RATE" \
            --max_length "$MAX_LENGTH" \
            --precision "$PRECISION" \
            --device_map "$DEVICE" \
            "${TRAIN_SOURCE[@]}" "${TRC_FLAG[@]}"
    fi
fi

banner "Done"
cat <<DONE
  routing stats     $STATS_FILE
  selected experts  $SELECTION_FILE
  expert weights    $TRAIN_DIR/expert_weights.pt

To evaluate, load the base model and apply the expert weights:

    from rise import load_selected_experts, load_expert_weights
    experts, model_type = load_selected_experts("$TRAIN_DIR/selected_experts.json")
    load_expert_weights(model, experts, model_type, "$TRAIN_DIR/expert_weights.pt")
DONE
