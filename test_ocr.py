#!/usr/bin/env python3
"""Minimal Unlimited OCR 2 test for Ubuntu + NVIDIA RTX 3090.

Examples:
  python test_ocr.py --image samples/sample_ocr.png
  python test_ocr.py --images page1.jpg page2.jpg --multi

The code follows Baidu's official Transformers inference path for
`baidu/Unlimited-OCR-2` and adds RTX-3090 friendly diagnostics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

print("cuDNN enabled:", torch.backends.cudnn.enabled)

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "baidu/Unlimited-OCR-2"


def gib(n: int) -> float:
    return n / 1024**3


def print_cuda_info() -> None:
    print("=" * 72)
    print("Unlimited OCR 2 / CUDA environment")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"CUDA ready  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is not available. Check NVIDIA driver and CUDA-enabled PyTorch.")

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    free_b, total_b = torch.cuda.mem_get_info(idx)
    print(f"GPU         : {props.name}")
    print(f"VRAM total  : {gib(total_b):.2f} GiB")
    print(f"VRAM free   : {gib(free_b):.2f} GiB")
    if "3090" not in props.name:
        print("NOTE        : script targets RTX 3090, but another CUDA GPU was detected.")
    print("=" * 72)


def choose_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("WARNING: BF16 is not reported as supported; falling back to FP16.")
            return torch.float16
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def load_model(dtype: torch.dtype):
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Loading model: {MODEL_NAME}, dtype={dtype}")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval().cuda()
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"CUDA allocated: {gib(allocated):.2f} GiB")
    print(f"CUDA reserved : {gib(reserved):.2f} GiB")
    return tokenizer, model


def ensure_files(paths: list[str]) -> list[str]:
    checked = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"ERROR: image not found: {path}")
        checked.append(str(path))
    return checked


def run_single(model, tokenizer, image: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[single] image: {image}")
    t0 = time.time()
    with torch.inference_mode():
        text = model.infer(
            tokenizer,
            image_file=image,
            output_path=str(output_dir),
            save_results=True,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[single] elapsed: {elapsed:.2f}s")
    print("\n===== OCR RESULT =====")
    print(text)
    print("===== END RESULT =====\n")


def run_multi(model, tokenizer, images: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[multi] pages: {len(images)}")
    for i, p in enumerate(images, 1):
        print(f"  {i:02d}: {p}")
    t0 = time.time()
    with torch.inference_mode():
        text, num_tokens = model.infer_multi(
            tokenizer,
            image_files=images,
            output_path=str(output_dir),
            save_results=True,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[multi] elapsed: {elapsed:.2f}s")
    print(f"[multi] output tokens: {num_tokens}")
    print("\n===== OCR RESULT =====")
    print(text)
    print("===== END RESULT =====\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="single image path")
    parser.add_argument("--images", nargs="+", help="multiple page image paths")
    parser.add_argument("--multi", action="store_true", help="force infer_multi")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--output", default="output", help="result directory")
    args = parser.parse_args()

    # Helps reduce allocator fragmentation on a 24 GiB RTX 3090.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.images:
        images = ensure_files(args.images)
    elif args.image:
        images = ensure_files([args.image])
    else:
        raise SystemExit("Give --image FILE or --images PAGE1 PAGE2 ...")

    print_cuda_info()
    dtype = choose_dtype(args.dtype)

    try:
        tokenizer, model = load_model(dtype)
        out = Path(args.output)
        if args.multi or len(images) > 1:
            run_multi(model, tokenizer, images, out / "multi")
        else:
            run_single(model, tokenizer, images[0], out / "single")
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print("\nCUDA OUT OF MEMORY", file=sys.stderr)
        print("RTX 3090 has 24 GiB VRAM. Try these in order:", file=sys.stderr)
        print("  1) close other GPU processes (check: nvidia-smi)", file=sys.stderr)
        print("  2) test one page first instead of --multi", file=sys.stderr)
        print("  3) try --dtype fp16", file=sys.stderr)
        print("  4) reduce image resolution before inference", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
