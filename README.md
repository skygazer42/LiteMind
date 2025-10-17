# LiteMind

LiteMind 是一个面向移动端的深度学习推理应用，基于 [MNN](https://github.com/alibaba/MNN) 构建，通过 JNI 将高性能原生算法能力暴露给 Android 前端，并提供灵活的模型导出与部署流水线。

## 项目目标
- 打造一套轻量级、可扩展的移动端推理解决方案。
- 支持从多种训练框架导出模型，并转换为 MNN、ONNX、TensorFlow Lite 等常用推理格式。
- 使用 C++/JNI 实现高性能核心算法，并通过 Android UI 提供易用的交互体验。
- 构建自动化的构建、测试与发布流程，保证模型与应用同步迭代。

## 技术栈
- **推理引擎：** Alibaba MNN（C++）
- **原生桥接：** JNI / C++17、CMake、Android NDK
- **移动前端：** Android（Kotlin/Java）
- **模型导出：** Python 3、ONNX、TensorFlow、PyTorch（计划支持）
- **CI & 工具链：** Gradle、CMake、GitHub Actions（待补充）

## 目录结构
仓库初始化后将包含如下核心目录 ：

```
LiteMind/
├── android/           # Android 应用源码（Kotlin/Java）
│   └── app/           # Gradle 模块，UI & 网络交互
├── jni/               # C++/JNI 源码、CMake 构建脚本
├── models/            # 训练输出、转换后的权重与校验数据
├── scripts/           # 模型转换、CI 辅助脚本
├── docs/              # 设计文档与 API 说明
├── README.md          # 当前说明文档
└── LICENSE
```

> 当前仓库已创建基础目录与占位文件，后续开发可直接在对应位置迭代。

## 快速开始
1. **准备环境**
   - Android Studio Flamingo 及以上版本
   - Android NDK (r23+) 与 CMake 3.22+
   - Python 3.9 及以上，用于模型转换脚本
   - （可选）Conda / Virtualenv 管理训练环境
2. **克隆仓库**
   ```bash
   git clone https://github.com/<your-org>/LiteMind.git
   cd LiteMind
   ```
3. **同步依赖**
   - 后续将通过 `git submodule`/`fetch` 引入 MNN 源码或预编译产物
   - Android 端在添加 Gradle Wrapper 后可运行 `./gradlew tasks` 验证构建环境是否就绪
4. **本地构建**（占位）
   - 当 `android/` 模块就绪后，可在 Android Studio 中直接运行
   - JNI 层通过 `cmake --build` 生成 `.so` 并由 Gradle 自动包装

## 架构概览
```
┌──────────────┐    导出脚本     ┌─────────────┐
│  训练框架     │ ─────────────→ │  模型格式化   │
│ (PyTorch等) │                │ (ONNX/MNN) │
└──────────────┘                └────┬────────┘
                                      │
                                 ┌────▼────┐ JNI/C++
                                 │ LiteMind │ 原生层
                                 └────┬────┘
                                      │
                                 ┌────▼────┐
                                 │ Android │ UI & UX
                                 └─────────┘
```

## 模型导出流水线
1. 在训练框架中完成模型训练并保存原始权重。
2. 使用 `scripts/` 下的转换脚本生成 ONNX/TFLite/MNN 等格式。
3. 将产物放入 `models/`，并附上元数据（输入尺寸、量化信息等）。
4. 在 `jni/` 中实现对应的加载与推理逻辑，通过 JNI 暴露给 Android。
5. Android 层负责处理权限、输入输出、UI 显示与业务逻辑。

## 开发流程建议
- 采用 Feature Branch + Pull Request，确保代码评审与自动化测试。
- 优先实现模型转换和 JNI 推理核心，再搭建 Android UI。
- 为 JNI 提供单元测试（gtest）与 Android Instrumented Tests，保障稳定性。
- 配置 GitHub Actions / GitLab CI 跑自动化构建与静态检测（待补充）。

## 路线图（初稿）
1. 集成 MNN 推理引擎（源码或预编译库）。
2. 实现基础模型导出脚本（PyTorch → ONNX → MNN）。
3. 搭建 JNI/C++ 推理管线与单元测试。
4. 构建 Android Demo 应用，完成模型调用与结果展示。
5. 增加模型版本与资源管理机制，完善 CI/CD。

## 贡献
欢迎通过 Issue 或 Pull Request 参与建设。提交前请先同步最新分支，并确保通过现有检查。

## 许可证
本项目基于 `LICENSE` 文件中的协议进行分发。
