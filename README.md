# Unlimited-OCR2-Test

用于在 **Ubuntu + NVIDIA RTX 3090 24GB** 上测试百度 **Unlimited OCR 2**（`baidu/Unlimited-OCR-2`）。

本仓库现在分成两条测试路径：

1. **Transformers 快速验证**：确认模型能在 RTX 3090 上加载并完成单页/少量页面推理；
2. **SGLang 长文档验证**：测试 Unlimited OCR 2 真正主打的几十页、100 页级 PDF / unlimited-horizon parsing。

> 上游项目：`baidu/Unlimited-OCR`  
> 模型：`baidu/Unlimited-OCR-2`

---

# A. 100 页级 PDF / 长文档测试（推荐重点测试）

不要用 `test_ocr.py` 循环 100 次单页 OCR 来验证官方宣称的超长文档能力。

官方 Unlimited OCR 2 对超长文档推荐使用定制版 **SGLang**：

```text
PDF
 ↓
按页渲染为图片
 ↓
整份文档作为 ONE logical request
 ↓
SGLang page-window scheduling
 ↓
窗口间复用 KV tail
 ↓
连续输出 <PAGE> 结果
```

仓库已经增加：

```text
long_pdf/
├── README.md
├── setup_sglang.sh
├── start_server_3090.sh
├── sitecustomize.py
└── pdf_ocr.py
```

## A1. 安装长文档环境

```bash
git pull
bash long_pdf/setup_sglang.sh
```

这里使用独立的：

```text
.venv-sglang/
```

不会复用前面 Transformers 的 `uocr2` conda 环境。

官方定制 SGLang wheel 存在 Git LFS 中，因此如果机器没有 `git-lfs`，先执行：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
```

然后重新运行：

```bash
bash long_pdf/setup_sglang.sh
```

## A2. 启动 RTX 3090 长文档服务

终端 1：

```bash
bash long_pdf/start_server_3090.sh
```

针对 24GB RTX 3090，默认采用偏保守参数：

```text
page window          = 4
max running requests = 1
mem fraction static  = 0.65
max prefill tokens   = 1024
CUDA graph           = disabled
```

另外，你当前机器已经确认存在 pip cuDNN 9.20 与系统 cuDNN 9.22 混装问题，因此这个启动脚本默认：

```text
UOCR_DISABLE_CUDNN=1
```

让 SGLang 的各个 worker 都绕过 cuDNN；CUDA / RTX 3090 仍然启用。

以后系统 CUDA/cuDNN 清理完成后可以：

```bash
UOCR_DISABLE_CUDNN=0 bash long_pdf/start_server_3090.sh
```

## A3. 先跑 10 页

终端 2：

```bash
source .venv-sglang/bin/activate

python long_pdf/pdf_ocr.py /path/to/document.pdf \
  --max-pages 10
```

## A4. 再跑 100 页

```bash
python long_pdf/pdf_ocr.py /path/to/your_100_page.pdf \
  --max-pages 100
```

输出：

```text
output/long_pdf/<pdf-name>/
├── raw_output.txt
├── metadata.json
└── pages/
    ├── page_0001.md
    ├── page_0002.md
    ├── ...
    └── page_0100.md
```

完整性重点看：

```text
page check : requested=100, <PAGE> markers=100, page files=100
```

更详细的参数、OOM 调整、300 DPI、gundam/base 模式说明见：

```text
long_pdf/README.md
```

---

# B. Transformers 快速验证

这一部分适合确认：

- CUDA 是否工作；
- 模型是否能下载；
- 权重是否能加载进 RTX 3090；
- 单页 / 少量页面接口是否工作。

它**不是** 100 页 PDF 的正式验证路径。

## B1. 推荐环境

官方 README 当前给出的 Transformers 快速测试环境为：

- Python 3.10
- `torch==2.13.0`
- `transformers==4.52.1`
- `huggingface-hub==0.36.0`
- `tokenizers==0.21.4`

RTX 3090 支持 FP16/BF16，本项目默认使用 BF16。

## B2. 创建环境

```bash
conda create -n uocr2 python=3.10 -y
conda activate uocr2

pip install -U pip
pip install -r requirements.txt
```

检查：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## B3. 生成测试图片

```bash
python make_sample_image.py
```

生成：

```text
samples/sample_ocr.png
```

## B4. 单页 OCR

```bash
python test_ocr.py --image samples/sample_ocr.png
```

如果需要 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python test_ocr.py --image samples/sample_ocr.png
```

## B5. 少量多页 OCR

```bash
python test_ocr.py \
  --images page_1.jpg page_2.jpg page_3.jpg \
  --multi
```

几十页/100 页不要继续堆到 Transformers `infer_multi()`，改用上面的 `long_pdf/` SGLang 路径。

---

# 文件说明

```text
Unlimited-OCR2-Test/
├── README.md
├── requirements.txt
├── make_sample_image.py
├── test_ocr.py
├── long_pdf/
│   ├── README.md
│   ├── setup_sglang.sh
│   ├── start_server_3090.sh
│   ├── sitecustomize.py
│   └── pdf_ocr.py
└── .gitignore
```

## 参考

- GitHub: https://github.com/baidu/Unlimited-OCR
- Hugging Face: https://huggingface.co/baidu/Unlimited-OCR-2
