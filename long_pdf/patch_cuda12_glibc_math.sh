#!/usr/bin/env bash
set -euo pipefail

# Patch CUDA 12.8 crt/math_functions.h for modern glibc (2.41+)
# that declares sinpi/cospi/rsqrt families with noexcept(true).
#
# This script:
#   1. verifies the target CUDA header exists;
#   2. creates a timestamped backup next to the original;
#   3. patches ONLY these six declarations:
#        sinpi, sinpif, cospi, cospif, rsqrt, rsqrtf
#   4. is idempotent: already-patched declarations are left unchanged.
#
# The patch is a workaround for CUDA 12.8 + modern glibc header mismatch.
# CUDA 13.2+ is the cleaner long-term fix.

CUDA_HOME="${UOCR_CUDA_HOME:-/usr/local/cuda-12.8}"
TARGET="${CUDA_HOME}/targets/x86_64-linux/include/crt/math_functions.h"

if [[ ! -f "${TARGET}" ]]; then
  echo "ERROR: CUDA math header not found: ${TARGET}"
  echo "Set UOCR_CUDA_HOME if CUDA 12.8 is installed elsewhere."
  exit 1
fi

echo "CUDA_HOME : ${CUDA_HOME}"
echo "Target    : ${TARGET}"
echo "glibc     : $(ldd --version | head -n 1)"
echo

echo "Current declarations:"
grep -nE '(^|[[:space:]])(sinpi|sinpif|cospi|cospif|rsqrt|rsqrtf)\(' "${TARGET}" | head -n 30 || true

echo
read -r -p "Patch the six CUDA declarations and create a backup? [y/N] " answer
case "${answer}" in
  y|Y|yes|YES) ;;
  *) echo "Cancelled."; exit 0 ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${TARGET}.bak-${STAMP}"

echo "Creating backup: ${BACKUP}"
sudo cp -a "${TARGET}" "${BACKUP}"

sudo python3 - "${TARGET}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

names = ["sinpi", "sinpif", "cospi", "cospif", "rsqrt", "rsqrtf"]
changed = []
already = []
missing = []

for name in names:
    # CUDA 12.8 device declaration, for example:
    # extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double sinpi(double x);
    pattern = re.compile(
        rf"(extern\s+__DEVICE_FUNCTIONS_DECL__\s+__device_builtin__\s+"
        rf"(?:double|float)\s+{name}\s*\([^;\n]*\))"
        rf"(\s+noexcept\s*\(true\))?\s*;"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        missing.append(name)
        continue

    # Patch all matching declarations for this exact function name, but only
    # when noexcept(true) is absent.
    def repl(m):
        if m.group(2):
            return m.group(0)
        changed.append(name)
        return m.group(1) + " noexcept (true);"

    before = text
    text = pattern.sub(repl, text)
    if before == text and name not in changed:
        already.append(name)

if missing:
    print("ERROR: expected declarations were not found:", ", ".join(missing), file=sys.stderr)
    print("The CUDA header layout differs from the expected CUDA 12.8 form.", file=sys.stderr)
    sys.exit(2)

path.write_text(text)
print("Patched:", ", ".join(dict.fromkeys(changed)) if changed else "none")
print("Already patched:", ", ".join(already) if already else "none")
PY

echo
echo "Patched declarations:"
grep -nE '(^|[[:space:]])(sinpi|sinpif|cospi|cospif|rsqrt|rsqrtf)\(' "${TARGET}" | head -n 30 || true

echo
echo "Done."
echo "Backup: ${BACKUP}"
echo
echo "Next steps:"
echo "  fuser -k 10000/tcp || true"
echo "  rm -rf ~/.cache/sglang/jit/sm86/sgl_kernel_jit_window_unpartition_add_768_false"
echo "  bash long_pdf/start_server_3090.sh"
