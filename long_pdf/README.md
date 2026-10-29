# Unlimited OCR 2：100 页级 PDF 测试

这个目录用于测试 Unlimited OCR 2 真正主打的 **ultra-long / unlimited-horizon document parsing**，而不是简单的单页 OCR。

官方 README 对超长文档推荐使用定制版 **SGLang**：PDF 转为按顺序排列的页面图片后，**整份文档作为一次请求**提交；服务端通过 page-window scheduling 分窗口处理，并复用上一窗口的 KV tail。

> 目标环境：Ubuntu + NVIDIA RTX 3090 24GB。
>
> 3090 显存远小于官方常见的 A800-80GB 环境，因此这里默认优先“跑通”而不是追求并发和吞吐。

## 目录

```text
long_pdf/
├── setup_sglang.sh       # 创建独立 SGLang 环境并安装百度官方定制 wheel
├── start_server_3090.sh  # 3090 24GB 的保守服务端参数
└── pdf_ocr.py            # PDF -> 页面图 -> 单次长文档请求 -> 输出结果
```

## 1. 拉取最新代码

```bash
git pull
```

## 2. 建立独立 SGLang 环境

不要直接复用之前的 `uocr2` Transformers 环境。长文档测试使用单独的 `.venv-sglang`：

```bash
bash long_pdf/setup_sglang.sh
```

脚本会：

1. 拉取 `baidu/Unlimited-OCR` 官方仓库到 `.third_party/Unlimited-OCR`；
2. 创建 Python 3.10 的 `.venv-sglang`；
3. 安装官方仓库中的定制 SGLang wheel；
4. 安装 `PyMuPDF`、`fastokens`、`requests`。

## 3. 启动 3090 长文档服务

终端 1：

```bash
bash long_pdf/start_server_3090.sh
```

默认关键参数：

```text
page window          = 4 pages
max running requests = 1
mem fraction static  = 0.65
max prefill tokens   = 1024
CUDA graph           = disabled
```

这不是官方吞吐最优配置，而是针对 RTX 3090 24GB 的第一版保守配置。

看到服务监听 `10000` 端口后，再开第二个终端运行客户端。

### 如果服务端 OOM

先把 page window 从 4 降到 2：

```bash
UOCR_PAGE_WINDOW=2 bash long_pdf/start_server_3090.sh
```

还不够时进一步降低静态显存比例：

```bash
UOCR_PAGE_WINDOW=2 \
UOCR_MEM_FRACTION_STATIC=0.60 \
bash long_pdf/start_server_3090.sh
```

不要一开始就提高并发。我们的目标是先验证 100 页 PDF 能否完整跑通。

## 4. 先测试 10 页

终端 2：

```bash
source .venv-sglang/bin/activate

python long_pdf/pdf_ocr.py /path/to/document.pdf \
  --max-pages 10
```

默认会：

- 从第 1 页开始；
- 220 DPI 渲染为 JPEG；
- 使用 `gundam` 动态裁剪模式；
- 把这 10 页作为 **一次请求**提交；
- 流式打印 OCR 结果；
- 把结果写入 `output/long_pdf/<pdf-name>/`。

## 5. 正式测试 100 页 PDF

```bash
source .venv-sglang/bin/activate

python long_pdf/pdf_ocr.py /path/to/your_100_page.pdf \
  --max-pages 100
```

如果 PDF 本身刚好 100 页，也可以直接：

```bash
python long_pdf/pdf_ocr.py /path/to/your_100_page.pdf
```

输出结构：

```text
output/long_pdf/your_100_page/
├── raw_output.txt
├── metadata.json
└── pages/
    ├── page_0001.md
    ├── page_0002.md
    ├── ...
    └── page_0100.md
```

`raw_output.txt` 最重要：它保留模型原始 `<PAGE>` 输出。如果逐页拆分数量不对，先看这个文件。

`metadata.json` 会记录：

- 实际请求页数；
- 原 PDF 页码；
- DPI；
- gundam/base 模式；
- 最大输出 token；
- `<PAGE>` marker 数；
- 总耗时；
- API usage（如果服务返回）。

### 完整性判断

100 页测试完成后，重点看最后一行：

```text
page check : requested=100, <PAGE> markers=100, page files=100
```

理想情况下三者都应该是 100。

## 6. 更接近官方示例：300 DPI

官方 README 的 PDF 示例使用 300 DPI。如果 220 DPI 已经跑通、显存和系统内存也够，可以测试：

```bash
python long_pdf/pdf_ocr.py /path/to/your_100_page.pdf \
  --max-pages 100 \
  --dpi 300
```

注意：300 DPI 会显著增大页面图片和 HTTP 请求体。100 页时 CPU 内存、网络/本机请求序列化开销也会明显增加。

## 7. gundam 与 base

默认：

```bash
--mode gundam
```

对应：

```python
"images_config": {"crop_mode": True}
```

如果某份 PDF 页面非常多、图像很复杂或出现显存问题，可以测试 base：

```bash
python long_pdf/pdf_ocr.py /path/to/document.pdf \
  --max-pages 100 \
  --mode base
```

`base` 每页使用全局视图，图像 token 压力通常更小，但细小文字的效果可能弱于动态裁剪。

## 8. 从中间页开始测试

例如测试第 101～200 页：

```bash
python long_pdf/pdf_ocr.py /path/to/book.pdf \
  --first-page 101 \
  --max-pages 100
```

## 9. 关于“Unlimited”的正确理解

这里的“Unlimited”不是指 GPU 一次把 100 页全部展开进普通 attention context。

Unlimited OCR 2 的关键是服务器端长文档调度：

```text
100-page PDF
    ↓
100 ordered page images
    ↓
ONE logical request
    ↓
page window 1 (e.g. pages 1-4)
    ↓ reuse KV tail
page window 2 (pages 5-8)
    ↓ reuse KV tail
...
    ↓
page window 25 (pages 97-100)
    ↓
continuous <PAGE> output
```

因此 **page window 可以很小，但整份 PDF 仍然是一个连续请求**。这才是我们要验证的能力。

## 10. 3090 与官方宣传效果的区别

代码路径与官方 SGLang 长文档方案一致，但 RTX 3090 只有 24GB 显存，因此不能预先保证能复现官方所有 benchmark 的速度、窗口大小或准确率。

建议测试顺序：

```text
10 pages
  ↓
30 pages
  ↓
50 pages
  ↓
100 pages
```

每一级都确认：

1. 服务端没有 OOM；
2. `<PAGE>` 数量与输入页数一致；
3. 中间页和最后几页没有明显截断；
4. 表格、段落、跨页内容的 Markdown 结构可接受。
