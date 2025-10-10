# JNI 模块

该目录包含 LiteMind 的原生推理实现与 CMake 配置。

## 目录说明
- `include/`：对外暴露的头文件，便于 Android NDK 或其他模块引用。
- `src/`：核心 JNI/C++ 源码。
- `CMakeLists.txt`：构建配置，默认输出 `liblitemind_core.so`。

## 本地构建
```bash
cd jni
cmake -B build -S . -DANDROID_ABI=arm64-v8a # 需通过 Android NDK 工具链
cmake --build build
```

Android Gradle 插件会自动调用该配置，无需手动复制 `.so` 文件。
