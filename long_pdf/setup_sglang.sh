#!/usr/bin/env bash
set -euo pipefail

# Unlimited OCR 2 长文档测试环境。
# 官方对超长/百页文档推荐使用自定义 SGLang，而不是 Transformers infer_multi。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/.third_party/Unlimited-OCR"
VENV_DIR="${ROOT_DIR}/.venv-sglang"
VENV_PYTHON="${VENV_DIR}/bin/python"
WHEEL_NAME="sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found"
  exit 1
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  cat <<'EOF'
ERROR: git-lfs is required.
The official Unlimited-OCR repository stores its customized SGLang .whl in Git LFS.

Ubuntu install:
  sudo apt-get update
  sudo apt-get install -y git-lfs
  git lfs install

Then rerun:
  bash long_pdf/setup_sglang.sh
EOF
  exit 2
fi

git lfs install >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, installing it into the current user environment ..."
  python3 -m pip install --user -U uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p "${ROOT_DIR}/.third_party"

if [[ ! -d "${THIRD_PARTY_DIR}/.git" ]]; then
  echo "[setup] cloning official baidu/Unlimited-OCR ..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/baidu/Unlimited-OCR.git "${THIRD_PARTY_DIR}"
else
  echo "[setup] updating official baidu/Unlimited-OCR ..."
  git -C "${THIRD_PARTY_DIR}" pull --ff-only
fi

# Only pull the wheel we actually need; do not download all LFS images/PDF assets.
echo "[setup] downloading official customized SGLang wheel from Git LFS ..."
git -C "${THIRD_PARTY_DIR}" lfs pull --include="wheel/${WHEEL_NAME}" --exclude=""

WHEEL_PATH="${THIRD_PARTY_DIR}/wheel/${WHEEL_NAME}"
if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "ERROR: official customized SGLang wheel not found:"
  echo "  ${WHEEL_PATH}"
  exit 3
fi

# A Git-LFS pointer is only ~100-200 bytes. Refuse to feed that to pip.
WHEEL_SIZE="$(stat -c%s "${WHEEL_PATH}")"
if [[ "${WHEEL_SIZE}" -lt 100000 ]]; then
  echo "ERROR: ${WHEEL_PATH} is only ${WHEEL_SIZE} bytes."
  echo "It is still a Git-LFS pointer, not the real wheel."
  echo "Try: git -C ${THIRD_PARTY_DIR} lfs pull --include='wheel/${WHEEL_NAME}'"
  exit 4
fi

echo "[setup] wheel size: ${WHEEL_SIZE} bytes"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[setup] creating isolated Python 3.10 environment: ${VENV_DIR}"
  uv venv --python 3.10 "${VENV_DIR}"
fi

# IMPORTANT: always target the venv by absolute Python path. This avoids accidentally
# installing packages into Conda base when the shell prompt shows both (.venv-sglang) and (base).
echo "[setup] target python: ${VENV_PYTHON}"
echo "[setup] installing customized SGLang ..."
uv pip install --python "${VENV_PYTHON}" "${WHEEL_PATH}"
uv pip install --python "${VENV_PYTHON}" \
  "pymupdf==1.27.2.2" \
  "fastokens==0.3.2" \
  requests

echo
echo "[setup] validating imports in the exact target interpreter ..."
"${VENV_PYTHON}" - <<'PY'
import sys
import torch
import fitz
import requests
import sglang

print("Python executable:", sys.executable)
print("Python :", sys.version.split()[0])
print("Torch  :", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("CUDA OK:", torch.cuda.is_available())
print("cuDNN  :", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
print("PyMuPDF:", fitz.VersionBind)
print("requests:", requests.__version__)
print("SGLang :", getattr(sglang, "__version__", "unknown"))
PY

echo
echo "Environment ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Next: bash long_pdf/start_server_3090.sh"
