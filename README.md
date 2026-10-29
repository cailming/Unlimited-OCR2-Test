# Unlimited-OCR2-Test

用于在 **Ubuntu + NVIDIA RTX 3090 24GB** 上快速验证百度 **Unlimited OCR 2**（`baidu/Unlimited-OCR-2`）的 Transformers 推理。

> 上游项目：`baidu/Unlimited-OCR`  
> 模型：`baidu/Unlimited-OCR-2`

## 1. 推荐环境

官方 README 当前给出的 Transformers 快速测试环境为：

- Python 3.10
- `torch==2.13.0`
- `transformers==4.52.1`
- `huggingface-hub==0.36.0`
- `tokenizers==0.21.4`

RTX 3090 支持 FP16/BF16，本项目默认使用 BF16。

建议 Ubuntu 22.04 / 24.04，先确认 NVIDIA 驱动正常：

```bash
nvidia-smi
```

## 2. 创建 Python 环境

推荐 conda：

```bash
conda create -n uocr2 python=3.10 -y
conda activate uocr2
```

或者使用 venv：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

## 3. 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

> 注意：PyTorch 是否能直接安装成功取决于当前官方 wheel/你的 CUDA 环境。如果 `torch==2.13.0` 的默认 PyPI wheel 在你的机器上不可用，请按照 PyTorch 官方 CUDA wheel 安装方式先安装对应 CUDA 版本的 torch/torchvision，再安装其余依赖。

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

## 4. 生成测试图片

仓库自带一段生成测试图片的代码，不需要另外找图片：

```bash
python make_sample_image.py
```

生成：

```text
samples/sample_ocr.png
```

## 5. 单页 OCR（RTX 3090 首先测试这个）

```bash
python test_ocr.py --image samples/sample_ocr.png
```

默认：

- 模型：`baidu/Unlimited-OCR-2`
- dtype：BF16
- GPU：CUDA
- 输出目录：`output/single/`

程序会打印：

- Python / PyTorch / CUDA 版本
- GPU 名称
- 总显存 / 空闲显存
- 模型加载后 allocated/reserved 显存
- OCR 推理耗时
- OCR 文本结果

首次运行会自动从 Hugging Face 下载模型，因此需要能访问 `huggingface.co`。

如果需要指定 HF 镜像，可自行配置环境变量，例如：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 6. 测试自己的图片

```bash
python test_ocr.py --image /path/to/test.jpg
```

指定输出目录：

```bash
python test_ocr.py \
  --image /path/to/test.jpg \
  --output output/my_test
```

## 7. 多页 OCR

Unlimited OCR 2 官方支持 `infer_multi()`，多页按顺序一次送入模型，可利用跨页上下文：

```bash
python test_ocr.py \
  --images page_1.jpg page_2.jpg page_3.jpg \
  --multi
```

RTX 3090 只有 24GB 显存，因此建议先从 2 页开始测试，再逐步增加页数。

## 8. BF16 / FP16

默认：

```bash
python test_ocr.py --image test.jpg --dtype bf16
```

如果 BF16 出现兼容性问题：

```bash
python test_ocr.py --image test.jpg --dtype fp16
```

## 9. RTX 3090 出现 CUDA OOM 时

先看 GPU 是否被其他程序占用：

```bash
nvidia-smi
```

本项目会自动设置：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

仍然 OOM 时，按顺序尝试：

1. 关闭其他占用 RTX 3090 的进程；
2. 单页测试，不要一开始使用多页；
3. 使用 `--dtype fp16`；
4. 将超高分辨率图片缩小后再识别；
5. 如果目标是几十页/上百页文档，改用官方推荐的 SGLang 服务化方案，而不是直接在 24GB 显卡上用 Transformers 一次塞入大量页面。

## 10. 文件说明

```text
Unlimited-OCR2-Test/
├── README.md
├── requirements.txt
├── make_sample_image.py    # 生成测试图片
├── test_ocr.py             # 单页 / 多页 OCR 测试程序
├── samples/                # 运行 make_sample_image.py 后产生
└── output/                 # OCR 输出目录
```

## 11. 一次性跑通流程

```bash
git clone https://github.com/cailming/Unlimited-OCR2-Test.git
cd Unlimited-OCR2-Test

conda create -n uocr2 python=3.10 -y
conda activate uocr2

pip install -U pip
pip install -r requirements.txt

python make_sample_image.py
python test_ocr.py --image samples/sample_ocr.png
```

如果最后能看到类似：

```text
GPU         : NVIDIA GeForce RTX 3090
CUDA ready  : True
...
===== OCR RESULT =====
...
===== END RESULT =====
```

就说明 **Unlimited OCR 2 已经成功在 RTX 3090 上完成本地推理**。

## 参考

- GitHub: https://github.com/baidu/Unlimited-OCR
- Hugging Face: https://huggingface.co/baidu/Unlimited-OCR-2
