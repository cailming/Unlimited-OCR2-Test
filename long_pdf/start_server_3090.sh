#!/usr/bin/env bash
set -euo pipefail

# RTX 3090 24GB 单卡启动脚本。
# 目标：优先跑通 Unlimited OCR 2 长文档，并规避当前 SGLang eager-mode bbox 循环问题。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-sglang"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: ${VENV_DIR} does not exist."
  echo "Run: bash long_pdf/setup_sglang.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# -----------------------------------------------------------------------------
# CUDA 12.8 JIT host compiler
# -----------------------------------------------------------------------------
# CUDA 12.8 需要 GCC <= 14。优先使用 gcc-14/g++-14。
HOST_GCC="${UOCR_HOST_GCC:-}"
HOST_GXX="${UOCR_HOST_GXX:-}"

if [[ -z "${HOST_GCC}" ]]; then
  if command -v gcc-14 >/dev/null 2>&1; then
    HOST_GCC="$(command -v gcc-14)"
  elif command -v gcc-13 >/dev/null 2>&1; then
    HOST_GCC="$(command -v gcc-13)"
  elif command -v gcc >/dev/null 2>&1; then
    DEFAULT_GCC="$(command -v gcc)"
    DEFAULT_GCC_MAJOR="$(${DEFAULT_GCC} -dumpfullversion -dumpversion | cut -d. -f1)"
    if [[ "${DEFAULT_GCC_MAJOR}" -le 14 ]]; then
      HOST_GCC="${DEFAULT_GCC}"
    fi
  fi
fi

if [[ -z "${HOST_GXX}" ]]; then
  if command -v g++-14 >/dev/null 2>&1; then
    HOST_GXX="$(command -v g++-14)"
  elif command -v g++-13 >/dev/null 2>&1; then
    HOST_GXX="$(command -v g++-13)"
  elif command -v g++ >/dev/null 2>&1; then
    DEFAULT_GXX="$(command -v g++)"
    DEFAULT_GXX_MAJOR="$(${DEFAULT_GXX} -dumpfullversion -dumpversion | cut -d. -f1)"
    if [[ "${DEFAULT_GXX_MAJOR}" -le 14 ]]; then
      HOST_GXX="${DEFAULT_GXX}"
    fi
  fi
fi

if [[ -z "${HOST_GCC}" || -z "${HOST_GXX}" ]]; then
  cat <<'EOF'
ERROR: CUDA 12.8 needs a supported host compiler, but no GCC <= 14 was found.
Install GCC 14 first:

  sudo apt-get update
  sudo apt-get install -y gcc-14 g++-14
EOF
  exit 2
fi

export CC="${HOST_GCC}"
export CXX="${HOST_GXX}"
export CUDAHOSTCXX="${HOST_GXX}"
export NVCC_CCBIN="${HOST_GCC}"

# 当前机器存在混合 cuDNN 库，默认通过 sitecustomize 禁用 cuDNN，CUDA 仍保持启用。
export UOCR_DISABLE_CUDNN="${UOCR_DISABLE_CUDNN:-1}"
export PYTHONPATH="${ROOT_DIR}/long_pdf${PYTHONPATH:+:${PYTHONPATH}}"

# -----------------------------------------------------------------------------
# Unlimited OCR 2 long-document settings
# -----------------------------------------------------------------------------
PAGE_WINDOW="${UOCR_PAGE_WINDOW:-4}"
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW="${PAGE_WINDOW}"
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW="${PAGE_WINDOW}"
export SGLANG_UOCR_MAX_PAGE_TOKENS="${UOCR_MAX_PAGE_TOKENS:-12000}"
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB="${UOCR_SAM_REL_POS_BIAS_MB:-64}"
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1
export SGLANG_UOCR_SHM_WRAP_MIN_MB=1
export SGLANG_VLM_CACHE_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MEM_FRACTION_STATIC="${UOCR_MEM_FRACTION_STATIC:-0.65}"
MAX_PREFILL_TOKENS="${UOCR_MAX_PREFILL_TOKENS:-2048}"
TOKENIZER_WORKERS="${UOCR_TOKENIZER_WORKERS:-2}"
PORT="${UOCR_PORT:-10000}"
CUDA_GRAPH_CONFIG='{"prefill":{"backend":"full","full_prefill_max_req":4}}'

NVCC_PATH="$(command -v nvcc || true)"
NVCC_VERSION="unknown"
if [[ -n "${NVCC_PATH}" ]]; then
  NVCC_VERSION="$(${NVCC_PATH} --version | tail -n 1)"
fi
GCC_VERSION="$(${HOST_GCC} --version | head -n 1)"
GXX_VERSION="$(${HOST_GXX} --version | head -n 1)"

cat <<EOF
========================================================================
Unlimited OCR 2 / SGLang / RTX 3090 long-document server
HF endpoint          : ${HF_ENDPOINT}
Page window          : ${PAGE_WINDOW}
Max page tokens      : ${SGLANG_UOCR_MAX_PAGE_TOKENS}
Mem fraction static  : ${MEM_FRACTION_STATIC}
Max prefill tokens   : ${MAX_PREFILL_TOKENS}
Tokenizer workers    : ${TOKENIZER_WORKERS}
Disable cuDNN        : ${UOCR_DISABLE_CUDNN}
CUDA Graph           : full (workaround for current UOCR2 eager-mode bug)
NVCC                 : ${NVCC_PATH:-not found}
NVCC version         : ${NVCC_VERSION}
CC                   : ${CC}
CC version           : ${GCC_VERSION}
CXX                  : ${CXX}
CXX version          : ${GXX_VERSION}
NVCC_CCBIN           : ${NVCC_CCBIN}
Port                 : ${PORT}
========================================================================
EOF

# 注意：当前安装的 SGLang CLI 没有独立的 --cuda-graph 开关。
# CUDA Graph 默认未禁用；下面用 per-phase 参数显式指定 full backend。
python -m sglang.launch_server \
  --model-path baidu/Unlimited-OCR-2 \
  --served-model-name Unlimited-OCR-2 \
  --chat-template unlimited-ocr \
  --tokenizer-backend fastokens \
  --tokenizer-worker-num "${TOKENIZER_WORKERS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests 1 \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
  --enable-custom-logit-processor \
  --enable-streaming-session \
  --allow-auto-truncate \
  --disable-radix-cache \
  --mm-preprocess-cache-size-mb 0 \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 4 \
  --cuda-graph-backend-prefill full \
  --cuda-graph-max-bs-prefill 16384 \
  --cuda-graph-bs-prefill 1024 2048 \
  --cuda-graph-config "${CUDA_GRAPH_CONFIG}" \
  --host 0.0.0.0 \
  --port "${PORT}"
