#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("samples/sample_ocr.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

img = Image.new("RGB", (1600, 1000), "white")
d = ImageDraw.Draw(img)

# Try a few common Ubuntu fonts; fall back to PIL default.
font_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]
font_path = next((p for p in font_candidates if Path(p).exists()), None)
font_big = ImageFont.truetype(font_path, 56) if font_path else ImageFont.load_default()
font_mid = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
font_small = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()

lines = [
    (80, 70, "Unlimited OCR 2 - RTX 3090 Test", font_big),
    (80, 180, "Invoice No: UOCR2-2026-1029", font_mid),
    (80, 250, "Customer: OpenAI Lab Demo", font_mid),
    (80, 340, "Item                  Qty      Price", font_mid),
    (80, 410, "GPU OCR Test            1      $3090", font_mid),
    (80, 480, "Document Parsing        2      $128", font_mid),
    (80, 590, "Total: $3346", font_big),
    (80, 730, "中文测试：百度文心 Unlimited OCR 2", font_mid),
    (80, 800, "目标环境：Ubuntu + NVIDIA RTX 3090", font_mid),
    (80, 900, "The quick brown fox jumps over the lazy dog.", font_small),
]
for x, y, text, font in lines:
    d.text((x, y), text, fill="black", font=font)

img.save(OUT)
print(f"saved: {OUT}")
