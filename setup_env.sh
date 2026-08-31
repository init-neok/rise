#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- create an environment and install RISE into it
# =============================================================================
#
#   bash setup_env.sh                  # venv at ./.venv (no conda needed)
#   bash setup_env.sh --conda          # conda env named "rise"
#   bash setup_env.sh --conda --name X # conda env named X
#   bash setup_env.sh --python 3.11    # pick the interpreter version
#
# Afterwards the script prints the one line you need to activate it.
#
# Torch is installed from PyPI, which ships a CUDA build by default. If your
# cluster needs a specific CUDA version, install torch yourself first and
# re-run this script -- it will not overwrite an existing torch.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="venv"
ENV_NAME="rise"
PY_VERSION="3.10"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda)  BACKEND="conda"; shift ;;
        --venv)   BACKEND="venv";  shift ;;
        --name)   ENV_NAME="$2";   shift 2 ;;
        --python) PY_VERSION="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$REPO_DIR"

if [[ "$BACKEND" == "conda" ]]; then
    command -v conda >/dev/null 2>&1 || {
        echo "conda not found on PATH. Re-run without --conda to use a venv." >&2
        exit 1
    }
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "Reusing existing conda env: $ENV_NAME"
    else
        echo "Creating conda env '$ENV_NAME' (python $PY_VERSION) ..."
        conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
    fi
    # Address the interpreter directly: `conda activate` does not work reliably
    # inside a non-interactive script.
    ENV_PREFIX="$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
    PYTHON="$ENV_PREFIX/bin/python"
    ACTIVATE_HINT="conda activate $ENV_NAME"
else
    if [[ ! -d .venv ]]; then
        echo "Creating venv at $REPO_DIR/.venv ..."
        "python${PY_VERSION}" -m venv .venv 2>/dev/null || python3 -m venv .venv
    else
        echo "Reusing existing venv at $REPO_DIR/.venv"
    fi
    PYTHON="$REPO_DIR/.venv/bin/python"
    ACTIVATE_HINT="source $REPO_DIR/.venv/bin/activate"
fi

echo "Using interpreter: $PYTHON"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

if "$PYTHON" -c "import torch" 2>/dev/null; then
    echo "torch already present ($("$PYTHON" -c 'import torch; print(torch.__version__)')) -- keeping it."
fi

echo "Installing RISE and its dependencies ..."
"$PYTHON" -m pip install -e .

echo
echo "Verifying the install ..."
"$PYTHON" - <<'PYCHECK'
import torch, transformers, rise
print(f"  rise         {rise.__version__}")
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  CUDA         {'available, ' + str(torch.cuda.device_count()) + ' GPU(s)' if torch.cuda.is_available() else 'not available (CPU only)'}")
PYCHECK

echo
echo "Running the CPU smoke test (a few seconds) ..."
OMP_NUM_THREADS=4 "$PYTHON" tests/test_pipeline.py

cat <<MSG

-----------------------------------------------------------------------------
Environment ready. Activate it with:

    $ACTIVATE_HINT

Then run the pipeline:

    bash run_rise.sh --config configs/qwen3_30b_a3b_bn.env
-----------------------------------------------------------------------------
MSG
