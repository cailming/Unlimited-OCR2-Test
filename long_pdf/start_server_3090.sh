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
