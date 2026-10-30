#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================================================"
echo "Unlimited OCR 2 / Transformers long-PDF setup"
echo "Python: $(${PYTHON_BIN} --version 2>&1)"
echo "========================================================================"

${PYTHON_BIN} -m pip install "pymupdf==1.27.2.2"

if [[ ! -f ".third_party/Unlimited-OCR/utils/process_unlimitedOCR2.py" ]]; then
  mkdir -p .third_party
  rm -rf .third_party/Unlimited-OCR
  git clone --depth 1 https://github.com/baidu/Unlimited-OCR.git \
    .third_party/Unlimited-OCR
fi

${PYTHON_BIN} - <<'PY'
import pymupdf
import torch
import transformers

print("PyMuPDF     :", pymupdf.VersionBind)
print("PyTorch     :", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA ready  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
PY

echo
echo "Setup complete."
echo "Example:"
echo "  python long_pdf/pdf_ocr_transformers.py test.pdf --max-pages 100"
