#!/usr/bin/env bash
set -euo pipefail

# RTX 3090 24GB 单卡保守启动参数。
# 目标不是高并发，而是优先把 100 页级 PDF 跑通。

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
# SGLang 会在首次视觉推理时调用 nvcc JIT 编译 CUDA kernel。
# CUDA 12.8 官方只支持 GCC <= 14；新 Ubuntu 的默认 gcc 可能已经是 15+，
# 会导致：unsupported GNU version -> Failed to build JIT module -> scheduler 崩溃。
#
# 优先级：
#   1. 用户显式指定 UOCR_HOST_GCC / UOCR_HOST_GXX
#   2. gcc-14 / g++-14
#   3. gcc-13 / g++-13
#   4. 当前 gcc / g++（仅当主版本 <= 14）
#
# NVCC_CCBIN 是 nvcc 官方支持的默认 host compiler 覆盖方式。
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

Then rerun:

  bash long_pdf/start_server_3090.sh
EOF
  exit 2
fi

export CC="${HOST_GCC}"
export CXX="${HOST_GXX}"
export CUDAHOSTCXX="${HOST_GXX}"
export NVCC_CCBIN="${HOST_GCC}"

# 这台 3090 主机已经确认同时存在 pip cuDNN 9.20 与系统 cuDNN 9.22，
# Transformers 测试会触发 CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH。
# SGLang 有多个 worker，因此通过 sitecustomize 让每个 Python worker 都禁用 cuDNN。
# CUDA/GPU 仍然启用。以后清理好系统 CUDA/cuDNN 后可设 UOCR_DISABLE_CUDNN=0。
export UOCR_DISABLE_CUDNN="${UOCR_DISABLE_CUDNN:-1}"
export PYTHONPATH="${ROOT_DIR}/long_pdf${PYTHONPATH:+:${PYTHONPATH}}"

# Unlimited OCR 2 的长文档 page-window scheduling。
# 3090 只有 24GB，所以默认从 4 页窗口开始；如果 OOM，可改成 2。
PAGE_WINDOW="${UOCR_PAGE_WINDOW:-4}"
export SGLANG_UOCR_EANABLE_DOC_LIMIT=1
export SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW="${PAGE_WINDOW}"
export SGLANG_UOCR_DOC_BASE_PAGE_WINDOW="${PAGE_WINDOW}"
export SGLANG_UOCR_MAX_PAGE_TOKENS="${UOCR_MAX_PAGE_TOKENS:-12000}"

# 官方推荐的长文档内存保护开关。
export SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB="${UOCR_SAM_REL_POS_BIAS_MB:-64}"
export SGLANG_UOCR_SHM_WRAP_MODEL_DATA=1
export SGLANG_UOCR_SHM_WRAP_MIN_MB=1
export SGLANG_VLM_CACHE_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 24GB 显卡：把并发压到 1，并为激活值保留较多空间。
MEM_FRACTION_STATIC="${UOCR_MEM_FRACTION_STATIC:-0.65}"
MAX_PREFILL_TOKENS="${UOCR_MAX_PREFILL_TOKENS:-1024}"
TOKENIZER_WORKERS="${UOCR_TOKENIZER_WORKERS:-2}"
PORT="${UOCR_PORT:-10000}"

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

# 对 24GB 卡先禁用 CUDA Graph，牺牲吞吐换更低显存峰值。
# 如果后续确认显存充足，再按官方 README 打开 CUDA Graph 参数。
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
  --disable-cuda-graph \
  --host 0.0.0.0 \
  --port "${PORT}"
