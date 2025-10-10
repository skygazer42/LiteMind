# BiRefNet 模型说明

BiRefNet 是一款针对前景抠图/主体分割场景的轻量级网络，能够生成单通道 Alpha Matte（0-255 灰度掩码）。本目录提供了基于 ONNX Runtime 与 MNN 的离线推理脚本，便于在桌面或移动端集成前完成验证。

## 目录结构

```
models/birefnet/
├── birefnet_infer.py          # 直接从 Hugging Face 拉取 ONNX 模型
├── birefnet_infer_local.py    # 使用本地 ONNX 文件的推理脚本
└── birefnet_infer_mnn.py      # MNN 推理脚本

resources/birefnet/
├── raw/
│   ├── model.onnx             # 默认浮点 ONNX
│   └── model_int8_qdq.onnx    # QDQ 量化版本
└── mnn/
    ├── birefnet_w4_nostat.mnn # 4-bit 量化权重
    └── birefnet_w8_nostat.mnn # 8-bit 量化权重
```

## 环境依赖

- Python 3.9+
- 公共依赖：`pip install numpy pillow requests`
- ONNX 推理：
  - `pip install onnxruntime`（或 `onnxruntime-gpu`）
  - 需从 Hugging Face 拉取资源时：`pip install huggingface_hub`
- MNN 推理：`pip install MNN`

> 如果使用国内网络，建议提前配置 Hugging Face 镜像或下载后改用 `birefnet_infer_local.py`。

## 快速体验

### 1. 远程下载 + 推理（默认脚本）

```bash
python models/birefnet/birefnet_infer.py \
  --repo onnx-community/BiRefNet-ONNX \
  --image https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg \
  --save-mask outputs/mask.png \
  --save-cutout outputs/cutout.png
```

- 首次运行会自动从指定仓库下载 `onnx/model.onnx` 与 `preprocessor_config.json`。
- `--providers` 支持自定义执行后端（例如 `'CUDAExecutionProvider,CPUExecutionProvider'`）。

### 2. 使用本地 ONNX

```bash
python models/birefnet/birefnet_infer_local.py \
  --model resources/birefnet/raw/model.onnx \
  --use-default-pp \
  --image assets/sample.png \
  --save-mask outputs/local_mask.png
```

- 若已下载官方 `preprocessor_config.json`，可通过 `--pp-json` 指定文件并移除 `--use-default-pp`。
- `--repo` 同样支持远程拉取模型与预处理配置，仅当本地文件缺失时使用。

### 3. MNN 推理

```bash
python models/birefnet/birefnet_infer_mnn.py \
  --mnn resources/birefnet/mnn/birefnet_w8_nostat.mnn \
  --image assets/sample.png \
  --save-mask outputs/mask_mnn.png \
  --save-cutout outputs/cutout_mnn.png
```

- 默认输入尺寸为 512×512，脚本会自动缩放并在后处理阶段恢复到原尺寸。
- `--threads` 可调整 CPU 执行线程数（默认 4）。

## 输入输出与预处理

- 输入：RGB 图像，按照配置缩放到 `size`（默认 512×512），使用 ImageNet 均值方差归一化。
- 输出：单通道掩码（`mask.png`），取值范围 0-255，越接近 255 表示越接近前景。
- 可选：设置 `--save-cutout` 获得带 Alpha 通道的 RGBA PNG。

## 常见问题

- **提示缺少 onnxruntime / huggingface_hub：** 参考上方依赖安装命令补齐包。
- **HF 下载失败或速度慢：** 先通过浏览器下载模型与 `preprocessor_config.json` 至 `resources/birefnet/raw/`，再使用 `birefnet_infer_local.py`。
- **MNN API 兼容性：** 脚本对旧版 API 做了兼容处理；若仍然报错，请确认 MNN Python 版本 ≥ 1.0.0。

## 后续工作

- 补充 `preprocessor_config.json` 与精度评估报告。
- 在 `docs/` 中新增移动端集成示例，说明如何在 JNI 层加载 MNN 权重。
- 视需要添加批量推理脚本或单元测试，保障模型更新后的可回归性。
