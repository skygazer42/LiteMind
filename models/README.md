# 模型资源

用于存放训练完成并经过转换的模型权重、标签文件与相关元数据。建议组织方式如下：

```
models/
├── raw/           # 训练框架导出的原始模型（ONNX / TorchScript 等）
├── mnn/           # 转换后的 MNN 模型
├── tflite/        # 可选：TensorFlow Lite 模型
└── metadata/      # JSON/YAML, 记录输入尺寸、预处理、准确率等
```

每次提交模型产物时，务必附带版本号、生成脚本和校验报告，确保可追溯性。
