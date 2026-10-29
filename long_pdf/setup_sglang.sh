#!/usr/bin/env bash
set -euo pipefail

# Unlimited OCR 2 长文档测试环境。
# 官方对超长/百页文档推荐使用自定义 SGLang，而不是 Transformers infer_multi。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/.third_party/Unlimited-OCR"
VENV_DIR="${ROOT_DIR}/.venv-sglang"
WHEEL_NAME="sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, installing it into the current user environment ..."
  python3 -m pip install --user -U uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p "${ROOT_DIR}/.third_party"

if [[ ! -d "${THIRD_PARTY_DIR}/.git" ]]; then
  echo "[setup] cloning official baidu/Unlimited-OCR ..."
  git clone --depth 1 https://github.com/baidu/Unlimited-OCR.git "${THIRD_PARTY_DIR}"
else
  echo "[setup] updating official baidu/Unlimited-OCR ..."
  git -C "${THIRD_PARTY_DIR}" pull --ff-only
fi

WHEEL_PATH="${THIRD_PARTY_DIR}/wheel/${WHEEL_NAME}"
if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "ERROR: official customized SGLang wheel not found:"
  echo "  ${WHEEL_PATH}"
  echo "The upstream repository may have changed. Check its README/wheel directory."
  exit 2
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[setup] creating isolated Python 3.10 environment: ${VENV_DIR}"
  uv venv --python 3.10 "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Follow the upstream Unlimited OCR 2 SGLang setup.
echo "[setup] installing customized SGLang ..."
uv pip install "${WHEEL_PATH}"
uv pip install "pymupdf==1.27.2.2" "fastokens==0.3.2" requests

echo
python - <<'PY'
import sys
import torch
import fitz
import sglang

print("Python :", sys.version.split()[0])
print("Torch  :", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("CUDA OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
print("PyMuPDF:", fitz.VersionBind)
print("SGLang :", getattr(sglang, "__version__", "unknown"))
PY

echo
echo "Environment ready."
echo "Next: bash long_pdf/start_server_3090.sh"
