# Android 模块

该目录存放 LiteMind Android 客户端源码，负责完成以下职责：

- 提供与用户交互的界面，并展示推理结果。
- 管理模型资源的下载、缓存与版本控制。
- 通过 JNI 调用 `jni/` 目录中的原生推理库。
- 集成日志、权限管理与调试工具。

## 打开工程

1. 使用 Android Studio 2022.2+（Flamingo 及以上）。
2. 选择“Open an existing project”，指向 `android/` 目录。
3. 等待 Gradle 同步完成后即可运行 `app` 模块。

## 模块说明

- `app`：主应用模块，负责 UI、数据流与原生交互。
- `src/main/cpp`：CMake 相关配置，可调用 `jni/` 中的代码生成 `.so` 库。
- `src/main/java`：Kotlin/Java 源码，可根据业务规划进行分层（例如 `ui/`、`core/`、`data/`）。

后续可根据需求拆分更多功能模块或引入 Jetpack 组件。
