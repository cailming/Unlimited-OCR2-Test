#!/usr/bin/env python3
"""Unlimited OCR 2 long-PDF client.

This follows the upstream SGLang path for ultra-long documents:
1. Render a PDF to ordered page images.
2. Send all selected pages in ONE OpenAI-compatible request.
3. Let the server-side Unlimited OCR 2 page-window scheduler reuse context
   between windows.
4. Save the raw <PAGE>/<BLOCK> stream to disk.
5. Run Baidu's official process_unlimitedOCR2.convert() to produce readable Markdown.

Example:
    python long_pdf/pdf_ocr.py document.pdf --max-pages 100

Use the same .venv-sglang environment as the server because the custom
no-repeat logit processor comes from SGLang.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import pymupdf
import requests
from sglang.srt.sampling.custom_logit_processor import (
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

MODEL = "Unlimited-OCR-2"
PAGE_TOKEN_ID = 128827
EOS_TOKEN_ID = 1
NGRAM_SIZE = 35
NGRAM_PROCESSOR = DeepseekOCRNoRepeatNGramLogitProcessor.to_str()
ROOT_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_POSTPROCESSOR = (
    ROOT_DIR / ".third_party" / "Unlimited-OCR" / "utils" / "process_unlimitedOCR2.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse a long PDF with Unlimited OCR 2 through the SGLang server."
    )
    p.add_argument("pdf", type=Path, help="input PDF")
    p.add_argument("--server", default="http://127.0.0.1:10000")
    p.add_argument("--output-dir", type=Path, default=Path("output/long_pdf"))
    p.add_argument("--first-page", type=int, default=1, help="1-based page number")
    p.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="0 = all remaining pages; use 100 for a 100-page test",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="render DPI; 220 is a 3090-friendly default, use 300 to match upstream example",
    )
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--mode",
        choices=("gundam", "base"),
        default="gundam",
        help="gundam = dynamic tiling/crop_mode=True; base = one global page view",
    )
    p.add_argument(
        "--tokens-per-page",
        type=int,
        default=6500,
        help="global max_tokens = selected pages * this value",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=700000,
        help="cap global output budget; 0 disables this cap",
    )
    p.add_argument("--max-page-tokens", type=int, default=12000)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument(
        "--keep-images",
        action="store_true",
        help="keep rendered page JPEGs after the request",
    )
    p.add_argument(
        "--show-raw",
        action="store_true",
        help="also print the raw <PAGE>/<BLOCK> stream to the terminal (very verbose)",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if args.first_page < 1:
        raise SystemExit("--first-page must be >= 1")
    if args.max_pages < 0:
        raise SystemExit("--max-pages must be >= 0")
    if not 72 <= args.dpi <= 600:
        raise SystemExit("--dpi should be between 72 and 600")
    if not 20 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality should be between 20 and 100")
    if args.tokens_per_page <= 0:
        raise SystemExit("--tokens-per-page must be > 0")


def render_pdf(args: argparse.Namespace, image_dir: Path) -> tuple[list[Path], list[int]]:
    doc = pymupdf.open(args.pdf)
    try:
        total = doc.page_count
        start = args.first_page - 1
        if start >= total:
            raise SystemExit(f"PDF has {total} pages; --first-page={args.first_page} is out of range")

        stop = total if args.max_pages == 0 else min(total, start + args.max_pages)
        indexes = list(range(start, stop))
        image_dir.mkdir(parents=True, exist_ok=True)
        matrix = pymupdf.Matrix(args.dpi / 72.0, args.dpi / 72.0)
        paths: list[Path] = []

        print(
            f"[render] PDF pages={total}, selected={len(indexes)}, "
            f"range={start + 1}-{stop}, dpi={args.dpi}"
        )

        for n, page_index in enumerate(indexes, start=1):
            page_no = page_index + 1
            output = image_dir / f"page_{page_no:04d}.jpg"
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(output, jpg_quality=args.jpeg_quality)
            paths.append(output)
            if n == 1 or n % 10 == 0 or n == len(indexes):
                print(f"[render] {n}/{len(indexes)} -> {output.name}")
        return paths, [i + 1 for i in indexes]
    finally:
        doc.close()


def encode_image(path: Path) -> dict:
    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{data}"},
    }


def build_payload(args: argparse.Namespace, images: list[Path]) -> tuple[dict, int]:
    n_pages = len(images)
    requested_tokens = args.tokens_per_page * n_pages
    max_tokens = requested_tokens
    if args.max_output_tokens > 0:
        max_tokens = min(max_tokens, args.max_output_tokens)

    print(f"[request] encoding {n_pages} page images as base64 ...")
    content = [encode_image(p) for p in images]
    prompt = "<PAGE><image>" * n_pages + "Multi page parsing."
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"crop_mode": args.mode == "gundam"},
        "custom_logit_processor": NGRAM_PROCESSOR,
        "custom_params": {
            "ngram_size": NGRAM_SIZE,
            "window_size": 1024 if n_pages > 1 else 128,
            "page_token_id": PAGE_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "max_page_tokens": args.max_page_tokens,
        },
        "stream": True,
    }
    return payload, max_tokens


def stream_request(args: argparse.Namespace, payload: dict, raw_path: Path) -> dict:
    url = args.server.rstrip("/") + "/v1/chat/completions"
    session = requests.Session()
    # Avoid accidentally sending a localhost request through HTTP(S)_PROXY.
    session.trust_env = False

    print(f"[request] POST {url}")
    print("[request] raw layout stream ->", raw_path)
    started = time.time()
    response = session.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=args.timeout,
        stream=True,
    )

    if response.status_code != 200:
        body = response.text[:5000]
        raise RuntimeError(f"SGLang returned HTTP {response.status_code}:\n{body}")

    usage = None
    chars = 0
    next_progress = 10_000
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as out:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                out.write(delta)
                out.flush()
                chars += len(delta)
                if args.show_raw:
                    print(delta, end="", flush=True)
                elif chars >= next_progress:
                    print(f"[request] received {chars:,} raw chars ...")
                    while next_progress <= chars:
                        next_progress += 10_000

    elapsed = time.time() - started
    if args.show_raw:
        print()
    print(f"[request] completed in {elapsed:.1f}s, raw chars={chars:,}")
    return {"elapsed_seconds": round(elapsed, 2), "output_chars": chars, "usage": usage}


def split_raw_pages(raw_path: Path, page_numbers: list[int], raw_pages_dir: Path) -> tuple[int, int]:
    """Split raw <PAGE> sections for diagnostics only; these are NOT Markdown."""
    text = raw_path.read_text(encoding="utf-8")
    marker_count = text.count("<PAGE>")
    chunks = text.split("<PAGE>")
    if chunks and not chunks[0].strip():
        chunks = chunks[1:]

    raw_pages_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, chunk in enumerate(chunks[: len(page_numbers)]):
        page_no = page_numbers[idx]
        target = raw_pages_dir / f"page_{page_no:04d}.txt"
        target.write_text("<PAGE>" + chunk.lstrip("\n"), encoding="utf-8")
        written += 1

    return marker_count, written


def load_official_converter():
    """Load Baidu's official utils/process_unlimitedOCR2.py from the cloned upstream repo."""
    if not OFFICIAL_POSTPROCESSOR.is_file():
        raise RuntimeError(
            "Official postprocessor not found:\n"
            f"  {OFFICIAL_POSTPROCESSOR}\n"
            "Run: bash long_pdf/setup_sglang.sh"
        )

    spec = importlib.util.spec_from_file_location(
        "uocr2_official_postprocess", OFFICIAL_POSTPROCESSOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official postprocessor: {OFFICIAL_POSTPROCESSOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "convert"):
        raise RuntimeError("Official process_unlimitedOCR2.py has no convert() function")
    return module.convert


def postprocess_to_markdown(raw_path: Path, markdown_path: Path) -> int:
    """Convert the raw document stream to readable Markdown with Baidu's official converter."""
    convert = load_official_converter()
    raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    markdown = convert(raw)
    markdown_path.write_text(markdown, encoding="utf-8")
    return len(markdown)


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_name = args.pdf.stem
    run_dir = args.output_dir / run_name
    image_dir = run_dir / "rendered_pages"
    raw_path = run_dir / "raw_output.txt"
    raw_pages_dir = run_dir / "raw_pages"
    markdown_path = run_dir / "document.md"
    metadata_path = run_dir / "metadata.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Unlimited OCR 2 / long PDF test")
    print("PDF       :", args.pdf.resolve())
    print("Server    :", args.server)
    print("Mode      :", args.mode)
    print("Output dir:", run_dir.resolve())
    print("=" * 80)

    images, page_numbers = render_pdf(args, image_dir)
    payload, max_tokens = build_payload(args, images)

    try:
        stats = stream_request(args, payload, raw_path)
        marker_count, raw_page_files = split_raw_pages(raw_path, page_numbers, raw_pages_dir)

        print("[postprocess] converting raw layout blocks with Baidu official converter ...")
        markdown_chars = postprocess_to_markdown(raw_path, markdown_path)
        print(f"[postprocess] Markdown chars={markdown_chars:,}")

        metadata = {
            "source_pdf": str(args.pdf.resolve()),
            "selected_page_count": len(images),
            "source_pages": page_numbers,
            "dpi": args.dpi,
            "jpeg_quality": args.jpeg_quality,
            "mode": args.mode,
            "tokens_per_page": args.tokens_per_page,
            "max_tokens": max_tokens,
            "max_page_tokens": args.max_page_tokens,
            "page_markers_in_output": marker_count,
            "raw_page_files_written": raw_page_files,
            "markdown_chars": markdown_chars,
            "official_postprocessor": str(OFFICIAL_POSTPROCESSOR),
            **stats,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print("=" * 80)
        print("DONE")
        print("Markdown   :", markdown_path)
        print("raw output :", raw_path)
        print("raw pages  :", raw_pages_dir)
        print("metadata   :", metadata_path)
        print(
            f"page check : requested={len(images)}, <PAGE> markers={marker_count}, "
            f"raw pages={raw_page_files}"
        )
        if marker_count != len(images):
            print(
                "WARNING: output <PAGE> marker count differs from requested pages. "
                "Keep raw_output.txt for diagnosis."
            )
    finally:
        if not args.keep_images:
            shutil.rmtree(image_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
