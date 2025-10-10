# 脚本说明

该目录用于存放模型转换、自动化测试与部署相关脚本。建议结构：

- `export/`：模型格式转换脚本（PyTorch → ONNX → MNN 等）。
- `validation/`：推理精度与性能验证脚本。
- `tools/`：通用辅助脚本（打包、资源清理等）。

首次初始化后，可在此创建 `Python` 虚拟环境并通过 `requirements.txt` 管理依赖。
