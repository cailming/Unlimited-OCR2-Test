# Unlimited OCR 2：100 页 PDF 稳定读取

当前在 RTX 3090 24GB 上，**优先推荐 Transformers 路径**。目标先收敛为：稳定把 100 页 PDF 识别出来并生成 Markdown；SGLang 长文档 serving 暂时保留为实验路径。

百度官方 Unlimited OCR 2 同时支持：

- **Transformers**：原生 Python 推理，适合直接测试和离线处理；
- **SGLang**：更适合高性能 serving、长文档调度和并发。

本项目当前推荐先使用 Transformers，因为单页路径已经在 RTX 3090 上验证稳定，而当前定制 SGLang wheel 在长内容场景下仍遇到输出退化/停滞问题。

---

## 推荐方案：Transformers 逐页/小批量处理

### 目录

```text
long_pdf/
├── pdf_ocr_transformers.py  # 推荐：100 页 PDF 稳定读取
├── setup_transformers.sh    # Transformers 长 PDF 依赖检查/补充
├── pdf_ocr.py               # SGLang 长文档客户端（实验）
├── setup_sglang.sh          # SGLang 环境（实验）
└── start_server_3090.sh     # SGLang 服务端（实验）
```

## 1. 拉取最新代码

```bash
git pull
```

## 2. 使用原来的 uocr2 Transformers 环境

```bash
conda activate uocr2
```

安装/检查 PDF 与官方后处理依赖：

```bash
bash long_pdf/setup_transformers.sh
```

脚本只补充 `PyMuPDF`，并把百度官方仓库拉到：

```text
.third_party/Unlimited-OCR/
```

用于调用官方：

```text
utils/process_unlimitedOCR2.py
```

把模型原始 `<PAGE>/<BLOCK>` 输出转换成 Markdown。

## 3. 先关闭 SGLang，释放 3090 显存

Transformers 运行时不要同时保留 SGLang scheduler。

```bash
fuser -k 10000/tcp || true
nvidia-smi
```

如果 `nvidia-smi` 中还有旧的 Unlimited OCR Python 进程，确认用途后再结束，尽量给 Transformers 留出完整的 24GB 显存。

## 4. 先测试 3 页

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 3
```

默认配置：

```text
batch-pages = 1
DPI         = 220
retry DPI   = 180,150
resume      = true
```

也就是说默认采用最稳的方式：

```text
PDF
 ↓
第1页 -> model.infer() -> page_0001.md
 ↓
第2页 -> model.infer() -> page_0002.md
 ↓
第3页 -> model.infer() -> page_0003.md
 ↓
...
 ↓
合并 document.md
```

模型只加载一次，不会每页重新加载模型。

## 5. 正式读取 100 页

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 100
```

输出目录：

```text
output/transformers_pdf/test/
├── document.md
├── metadata.json
├── pages/
│   ├── page_0001.md
│   ├── page_0002.md
│   └── ...
├── raw/
│   ├── page_0001.txt
│   ├── page_0002.txt
│   └── ...
├── model_artifacts/
└── errors/
```

最终主要看：

```text
output/transformers_pdf/test/document.md
```

## 6. 断点续跑

默认启用 `--resume`。

例如已经完成 1～37 页，第 38 页意外失败或手工 `Ctrl+C`，直接重新执行同一命令：

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 100
```

已有的：

```text
pages/page_0001.md
...
pages/page_0037.md
```

会直接跳过，只处理缺失页。

如果希望全部重新识别：

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 100 \
  --no-resume
```

## 7. OOM 自动降 DPI

默认先使用：

```text
220 DPI
```

如果某页出现 CUDA OOM，会依次尝试：

```text
180 DPI
150 DPI
```

可以自行修改：

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 100 \
  --dpi 200 \
  --retry-dpis 170,140
```

## 8. 后续再尝试小批量 infer_multi

当前优先稳定，所以默认：

```bash
--batch-pages 1
```

以后确认逐页 100 页稳定后，可尝试两页一组：

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 100 \
  --batch-pages 2
```

或四页一组：

```bash
python long_pdf/pdf_ocr_transformers.py test.pdf \
  --max-pages 100 \
  --batch-pages 4
```

`batch-pages > 1` 会调用官方 `model.infer_multi()`，可以利用组内跨页上下文，但会提高显存和长序列压力。

对于当前 RTX 3090，建议先保持：

```text
batch-pages = 1
```

把 100 页完整读出来以后，再优化性能。

## 9. 从中间页开始

例如读取 101～200 页：

```bash
python long_pdf/pdf_ocr_transformers.py book.pdf \
  --first-page 101 \
  --max-pages 100
```

## 10. 为什么当前先不用 SGLang

SGLang 不是 Unlimited OCR 2 模型本身，而是高性能推理/服务框架，主要负责：

- HTTP serving；
- 多请求调度；
- KV cache；
- CUDA Graph；
- page-window 长文档调度；
- 更高吞吐。

这些能力对在线服务和高性能很有价值，但不是“读取 100 页 PDF”的必要条件。

目前项目需求优先级调整为：

```text
第一阶段：100 页稳定读取
       ↓
Transformers 单页 checkpoint
       ↓
第二阶段：2～4 页 infer_multi
       ↓
第三阶段：修复/升级 SGLang 后再测试整份长文档 serving
```

因此当前先把 `pdf_ocr_transformers.py` 作为主路径。

---

## SGLang 实验路径

原有文件仍保留：

```text
long_pdf/setup_sglang.sh
long_pdf/start_server_3090.sh
long_pdf/pdf_ocr.py
```

后续如果百度更新定制 SGLang wheel，或者需要测试真正的 page-window / KV-tail 跨页长文档能力，再继续使用这套路径。
