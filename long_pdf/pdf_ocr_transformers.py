#!/usr/bin/env python3
"""Stable long-PDF OCR with Unlimited OCR 2 + Transformers.

Design goal: finish 100-page PDFs reliably on a 24 GiB RTX 3090.
The default path processes one page at a time with model.infer(), because
single-page Transformers inference has already proven stable on this machine.
Optional --batch-pages > 1 uses model.infer_multi() for small batches.

Outputs are checkpointed page-by-page, so rerunning with --resume only processes
missing pages.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pymupdf
import torch

# This host has mixed cuDNN libraries; CUDA remains enabled.
torch.backends.cudnn.enabled = False

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "baidu/Unlimited-OCR-2"
ROOT_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_POSTPROCESSOR = (
    ROOT_DIR / ".third_party" / "Unlimited-OCR" / "utils" / "process_unlimitedOCR2.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read long PDFs with Unlimited OCR 2 through native Transformers."
    )
    p.add_argument("pdf", type=Path, help="input PDF")
    p.add_argument("--first-page", type=int, default=1, help="1-based first page")
    p.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="number of pages to process; 0 = all remaining pages",
    )
    p.add_argument("--dpi", type=int, default=220, help="initial render DPI")
    p.add_argument(
        "--retry-dpis",
        default="180,150",
        help="lower DPIs used only if a page/batch hits CUDA OOM",
    )
    p.add_argument(
        "--batch-pages",
        type=int,
        default=1,
        help="1 = safest per-page infer(); >1 = infer_multi() in small batches",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/transformers_pdf"),
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip pages whose page_NNNN.md already exists",
    )
    p.add_argument(
        "--keep-images",
        action="store_true",
        help="keep rendered JPEGs after OCR",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop on first failed page/batch instead of recording the error and continuing",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if args.first_page < 1:
        raise SystemExit("--first-page must be >= 1")
    if args.max_pages < 0:
        raise SystemExit("--max-pages must be >= 0")
    if args.batch_pages < 1:
        raise SystemExit("--batch-pages must be >= 1")
    if not 72 <= args.dpi <= 600:
        raise SystemExit("--dpi should be between 72 and 600")


def parse_retry_dpis(text: str, first: int) -> list[int]:
    values = [first]
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 72 or value > 600:
            raise SystemExit("--retry-dpis values should be between 72 and 600")
        values.append(value)
    return list(dict.fromkeys(values))


def gib(n: int) -> float:
    return n / 1024**3


def print_cuda_info() -> None:
    print("=" * 80)
    print("Unlimited OCR 2 / Transformers long-PDF mode")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA build   : {torch.version.cuda}")
    print(f"CUDA ready   : {torch.cuda.is_available()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU          : {props.name}")
    print(f"VRAM total   : {gib(total_b):.2f} GiB")
    print(f"VRAM free    : {gib(free_b):.2f} GiB")
    print("=" * 80)


def choose_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[model] BF16 not reported as supported; falling back to FP16")
        return torch.float16
    return torch.float16


def load_model(dtype: torch.dtype):
    print(f"[model] loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"[model] loading model: {MODEL_NAME}, dtype={dtype}")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval().cuda()
    torch.cuda.synchronize()
    print(f"[model] loaded in {time.time() - t0:.1f}s")
    print(f"[model] CUDA allocated={gib(torch.cuda.memory_allocated()):.2f} GiB")
    print(f"[model] CUDA reserved ={gib(torch.cuda.memory_reserved()):.2f} GiB")
    return tokenizer, model


def load_converter():
    if not OFFICIAL_POSTPROCESSOR.is_file():
        raise RuntimeError(
            "Baidu official postprocessor not found:\n"
            f"  {OFFICIAL_POSTPROCESSOR}\n"
            "Run: bash long_pdf/setup_transformers.sh"
        )
    spec = importlib.util.spec_from_file_location(
        "uocr2_official_postprocess", OFFICIAL_POSTPROCESSOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {OFFICIAL_POSTPROCESSOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def selected_pages(pdf: Path, first_page: int, max_pages: int) -> list[int]:
    doc = pymupdf.open(pdf)
    try:
        total = doc.page_count
    finally:
        doc.close()
    start = first_page
    if start > total:
        raise SystemExit(f"PDF has {total} pages; --first-page={first_page} is out of range")
    end = total if max_pages == 0 else min(total, first_page + max_pages - 1)
    pages = list(range(start, end + 1))
    print(f"[pdf] total={total}, selected={len(pages)}, range={start}-{end}")
    return pages


def render_pages(pdf: Path, page_numbers: list[int], dpi: int, image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    paths: list[Path] = []
    try:
        matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
        for page_no in page_numbers:
            target = image_dir / f"page_{page_no:04d}_{dpi}dpi.jpg"
            page = doc.load_page(page_no - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(target, jpg_quality=90)
            paths.append(target)
    finally:
        doc.close()
    return paths


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text)


def infer_one(model, tokenizer, image: Path, model_output_dir: Path) -> tuple[str, int | None]:
    model_output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        text = model.infer(
            tokenizer,
            image_file=str(image),
            output_path=str(model_output_dir),
            save_results=True,
        )
    torch.cuda.synchronize()
    return text, None


def infer_multi(
    model,
    tokenizer,
    images: list[Path],
    model_output_dir: Path,
) -> tuple[str, int | None]:
    model_output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        text, num_tokens = model.infer_multi(
            tokenizer,
            image_files=[str(p) for p in images],
            output_path=str(model_output_dir),
            save_results=True,
        )
    torch.cuda.synchronize()
    return text, int(num_tokens)


def write_batch_results(
    convert,
    raw_text: str,
    page_numbers: list[int],
    raw_dir: Path,
    pages_dir: Path,
) -> None:
    """Split on <PAGE> when possible and save one raw/md checkpoint per page."""
    chunks = raw_text.split("<PAGE>")
    if chunks and not chunks[0].strip():
        chunks = chunks[1:]

    if len(chunks) != len(page_numbers):
        start, end = page_numbers[0], page_numbers[-1]
        (raw_dir / f"batch_{start:04d}_{end:04d}.txt").write_text(
            raw_text, encoding="utf-8"
        )
        markdown = convert(raw_text)
        (pages_dir / f"batch_{start:04d}_{end:04d}.md").write_text(
            markdown, encoding="utf-8"
        )
        raise RuntimeError(
            f"Expected {len(page_numbers)} <PAGE> sections, got {len(chunks)} "
            f"for pages {start}-{end}"
        )

    for page_no, chunk in zip(page_numbers, chunks):
        raw = "<PAGE>" + chunk.lstrip("\n")
        markdown = convert(raw)
        (raw_dir / f"page_{page_no:04d}.txt").write_text(raw, encoding="utf-8")
        (pages_dir / f"page_{page_no:04d}.md").write_text(markdown, encoding="utf-8")


def build_document(page_numbers: list[int], pages_dir: Path, document_path: Path) -> list[int]:
    missing: list[int] = []
    with document_path.open("w", encoding="utf-8") as out:
        for page_no in page_numbers:
            page_path = pages_dir / f"page_{page_no:04d}.md"
            if not page_path.is_file():
                missing.append(page_no)
                continue
            out.write(f"<!-- source-page: {page_no} -->\n\n")
            out.write(page_path.read_text(encoding="utf-8").rstrip())
            out.write("\n\n")
    return missing


def main() -> None:
    args = parse_args()
    validate_args(args)
    retry_dpis = parse_retry_dpis(args.retry_dpis, args.dpi)

    run_dir = args.output_dir / args.pdf.stem
    pages_dir = run_dir / "pages"
    raw_dir = run_dir / "raw"
    image_dir = run_dir / "rendered"
    model_dir = run_dir / "model_artifacts"
    errors_dir = run_dir / "errors"
    for path in (pages_dir, raw_dir, image_dir, model_dir, errors_dir):
        path.mkdir(parents=True, exist_ok=True)

    page_numbers = selected_pages(args.pdf, args.first_page, args.max_pages)

    print_cuda_info()
    dtype = choose_dtype(args.dtype)
    convert = load_converter()
    tokenizer, model = load_model(dtype)

    started = time.time()
    failures: dict[int, str] = {}
    completed = 0

    cursor = 0
    while cursor < len(page_numbers):
        batch = page_numbers[cursor : cursor + args.batch_pages]
        if args.resume and all(
            (pages_dir / f"page_{p:04d}.md").is_file() for p in batch
        ):
            print(f"[skip] pages {batch[0]}-{batch[-1]} already complete")
            completed += len(batch)
            cursor += len(batch)
            continue

        batch_ok = False
        last_exc: BaseException | None = None

        for dpi in retry_dpis:
            images: list[Path] = []
            try:
                print(
                    f"[ocr] pages={batch[0]}-{batch[-1]} "
                    f"count={len(batch)} dpi={dpi}"
                )
                t0 = time.time()
                images = render_pages(args.pdf, batch, dpi, image_dir)
                out_dir = model_dir / f"pages_{batch[0]:04d}_{batch[-1]:04d}_{dpi}dpi"

                if len(batch) == 1:
                    raw_text, num_tokens = infer_one(
                        model, tokenizer, images[0], out_dir
                    )
                else:
                    raw_text, num_tokens = infer_multi(
                        model, tokenizer, images, out_dir
                    )

                write_batch_results(convert, raw_text, batch, raw_dir, pages_dir)
                elapsed = time.time() - t0
                token_text = "" if num_tokens is None else f", tokens={num_tokens}"
                print(
                    f"[ok] pages={batch[0]}-{batch[-1]} "
                    f"elapsed={elapsed:.1f}s{token_text}"
                )
                completed += len(batch)
                batch_ok = True
                break
            except BaseException as exc:
                last_exc = exc
                cleanup_cuda()
                if is_oom(exc):
                    print(
                        f"[oom] pages={batch[0]}-{batch[-1]} at {dpi} DPI; "
                        "retrying with lower DPI ..."
                    )
                    continue
                print(
                    f"[error] pages={batch[0]}-{batch[-1]} at {dpi} DPI: {exc}",
                    file=sys.stderr,
                )
                break
            finally:
                if not args.keep_images:
                    for image in images:
                        image.unlink(missing_ok=True)
                cleanup_cuda()

        if not batch_ok:
            message = repr(last_exc) if last_exc is not None else "unknown error"
            for p in batch:
                failures[p] = message
                (errors_dir / f"page_{p:04d}.txt").write_text(
                    message + "\n", encoding="utf-8"
                )
            if args.stop_on_error:
                raise RuntimeError(
                    f"OCR failed for pages {batch[0]}-{batch[-1]}: {message}"
                )

        cursor += len(batch)

    document_path = run_dir / "document.md"
    missing = build_document(page_numbers, pages_dir, document_path)
    metadata = {
        "source_pdf": str(args.pdf.resolve()),
        "model": MODEL_NAME,
        "mode": "transformers",
        "first_page": args.first_page,
        "requested_pages": page_numbers,
        "batch_pages": args.batch_pages,
        "dpi_attempts": retry_dpis,
        "dtype": args.dtype,
        "resume": args.resume,
        "completed_count": completed,
        "failed_pages": sorted(failures),
        "missing_pages_in_document": missing,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.keep_images:
        shutil.rmtree(image_dir, ignore_errors=True)

    print("=" * 80)
    print("DONE")
    print("document :", document_path)
    print("pages    :", pages_dir)
    print("raw      :", raw_dir)
    print("metadata :", run_dir / "metadata.json")
    print(
        f"summary  : requested={len(page_numbers)}, completed={completed}, "
        f"missing={len(missing)}"
    )
    if missing:
        print("missing pages:", ",".join(str(p) for p in missing))
        print("Rerun the same command; --resume will only retry missing pages.")
        raise SystemExit(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed page files are preserved for --resume.", file=sys.stderr)
        raise SystemExit(130)
