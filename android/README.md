# Android 模块

`android/` 目录存放 LiteMind 的 Android 客户端代码，负责 UI 展示、权限处理以及与 JNI 推理库的交互。

## 打开工程
- 推荐直接在 Android Studio 中打开仓库根目录 `LiteMind/android`，这样 IDE 会同时索引 `android/` 与 `jni/` 两个模块，Gradle 同步后即可调试 JNI 与 Kotlin 代码。
- 若只需浏览 Android 层，也可以单独打开 `android/` 目录；但进行原生开发时仍需依赖根目录下的 `jni/` 配置与模型资源。
- 请提前安装 Android Studio Flamingo 及以上版本，并在 `local.properties` 中配置 `ndk.dir`（NDK r23+）与 `cmake.dir`（3.22.1）。

## 目录结构
```
android/
├── app/
│   ├── build.gradle          # 模块级 Gradle 配置
│   └── src/main/
│       ├── java/…            # Kotlin 代码（UI、NativeBridge 等）
│       ├── res/…             # 布局与字符串资源
│       ├── assets/birefnet/  # BiRefNet MNN 模型
│       └── jniLibs/<abi>/    # 预编译原生依赖（libMNN.so 等）
├── build.gradle              # 项目级 Gradle 配置
├── gradle.properties
└── settings.gradle
```
 ce
## 运行 BiRefNet 演示
1. 确认 `app/src/main/assets/birefnet/` 下存在 `birefnet_w8_nostat.mnn`（已从 `resources/birefnet/mnn/` 拷贝）。
2. `app/src/main/jniLibs/<abi>/` 需包含对应架构的 `libMNN.so` 及其依赖（示例工程已预置）。
3. Gradle 同步后直接运行 `app` 模块。主页可选择图片并调用 JNI 进行抠图，界面会展示原图与生成的 Mask。

## 原生代码联动
- JNI 实现在仓库根目录 `jni/`，通过 `externalNativeBuild` 与 Android 工程关联。
- 如需扩展模型或新增原生接口，可修改：
  1. `jni/src/BiRefNetEngine.cpp` / `BiRefNetEngine.h` —— C++ 推理封装；
  2. `jni/src/litemind_jni.cpp` —— JNI 导出方法；
  3. `android/app/src/main/java/com/litemind/app/NativeBridge.kt` —— Kotlin 层桥接。
- 若替换模型文件，请同步更新 `BiRefNetEngineManager` 的默认资产路径，或在初始化时传入新的 asset 名称。
