"""Process-wide startup hook for this test project.

The user's RTX 3090 host currently has:
- torch 2.13 / CUDA 13 pip cuDNN 9.20
- system cuDNN 9.22 for CUDA 12.x

That combination can produce CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH when a
missing pip cuDNN sublibrary is resolved from /lib.  SGLang spawns worker
processes, so setting torch.backends.cudnn.enabled=False only in one launcher
process is not sufficient.  Python automatically imports sitecustomize in each
worker when this directory is on PYTHONPATH.

Set UOCR_DISABLE_CUDNN=0 to disable this workaround after the CUDA/cuDNN
installation has been cleaned up.
"""

import os

if os.environ.get("UOCR_DISABLE_CUDNN", "0") == "1":
    try:
        import torch

        torch.backends.cudnn.enabled = False
        print(
            "[uocr2 sitecustomize] cuDNN disabled; CUDA remains enabled "
            "(workaround for mixed cuDNN libraries).",
            flush=True,
        )
    except Exception as exc:  # startup diagnostics only; never hide the real app error
        print(f"[uocr2 sitecustomize] warning: could not disable cuDNN: {exc}", flush=True)
