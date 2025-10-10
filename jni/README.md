# JNI 模块

该目录包含 LiteMind 的原生推理实现与 CMake 配置，当前聚焦于 BiRefNet 的 MNN 推理封装。

## 目录结构
- `include/`
  - `BiRefNetEngine.h`：BiRefNet 推理引擎头文件。
  - `MNN/...`：从 MNN SDK 裁剪的公开头文件，便于在仓库内直接引用。
- `src/`
  - `BiRefNetEngine.cpp`：封装模型加载、预处理、推理与后处理。
  - `litemind_jni.cpp`：JNI 桥接实现，对外暴露 Native 方法。
- `CMakeLists.txt`：生成 `liblitemind_core.so` 的构建脚本。

## 构建准备
1. 确保已安装 Android NDK r23+ 与 CMake ≥ 3.22.1，并在 Android Studio 的 `local.properties` 中配置 `ndk.dir` 与 `cmake.dir`。
2. 准备 MNN 预编译库：
   - 默认从 `third_party/MNN` 读取（结构为 `include/`、`lib/<abi>/libMNN.so`）。
   - 如果直接打包到 App，可将 `libMNN.so` 放在 `android/app/src/main/jniLibs/<abi>/`，CMake 会自动导入。
3. Android 模块通过 `externalNativeBuild` 引用本目录，无需单独复制 `.so`。

## JNI 接口
- `long createEngine(String modelPath, int threads)`：加载指定路径的 MNN 模型。
- `void destroyEngine(long handle)`：释放原生引擎资源。
- `byte[] runInference(long handle, Bitmap bitmap)`：输入 `RGBA_8888` 位图，返回同尺寸的单通道掩码。
- `int[] getModelInputSize(long handle)`：返回模型期望的输入宽高（默认 512×512）。

### Kotlin 侧调用示例
```kotlin
object NativeBridge {
    init { System.loadLibrary("litemind_core") }

    external fun createEngine(modelPath: String, threads: Int = 4): Long
    external fun destroyEngine(handle: Long)
    external fun runInference(handle: Long, bitmap: Bitmap): ByteArray
    external fun getModelInputSize(handle: Long): IntArray
}
```

## 本地命令行构建
```bash
cd jni
cmake -B build -S . \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DANDROID_NDK=$ANDROID_NDK_HOME \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
cmake --build build
```

编译成功后会在 `jni/build/` 下生成 `liblitemind_core.so`，Android Gradle 插件在构建 APK 时会自动包含该库。
