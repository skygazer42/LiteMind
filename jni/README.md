# JNI 模块

该目录包含 LiteMind 的原生推理实现与 CMake 配置，当前重点集成了 BiRefNet 模型（MNN 推理）。

## 目录说明
- `include/`
  - `BiRefNetEngine.h`：BiRefNet 推理引擎头文件。
  - `LiteMind.h`：占位示例，可根据需要拓展。
- `src/`
  - `BiRefNetEngine.cpp`：封装 MNN Interpreter，完成预处理 / 推理 / 后处理。
  - `litemind_jni.cpp`：JNI 桥接，向 Android 层暴露 C++ 能力。
- `CMakeLists.txt`：构建配置，输出 `liblitemind_core.so`。

## 依赖准备
1. 下载 MNN 预编译包或自行编译 Android 版本，将头文件和库放置在 `third_party/MNN`：
   ```
   third_party/MNN/
   ├── include/       # MNN 头文件
   └── lib/
       └── arm64-v8a/ # 各 ABI 对应的 libMNN.so / libMNN.a
   ```
   > 如果希望使用 `android/app/src/main/jniLibs/<abi>/libMNN.so`，亦可保持默认结构，无需修改 CMake。
2. Gradle 侧在 `android/app/build.gradle` 中开启 CMake：
   ```groovy
   android {
       defaultConfig {
           externalNativeBuild {
               cmake {
                   abiFilters "arm64-v8a"
               }
           }
       }
       externalNativeBuild {
           cmake {
               path file("../../jni/CMakeLists.txt")
           }
       }
   }
   ```

## C++/JNI 接口概览
- `long createEngine(String modelPath, int threads)`：加载 `libMNN.so` 和模型文件，返回句柄。
- `void destroyEngine(long handle)`：释放对应引擎资源。
- `byte[] runInference(long handle, Bitmap bitmap)`：输入 `RGBA_8888` 位图，返回与原图同尺寸的掩码（0-255 灰度）。
- `int[] getModelInputSize(long handle)`：返回模型期望输入尺寸（默认 512x512）。

> JNI 会对 Bitmap 进行双线性缩放和归一化，输出掩码时保持原始大小。

## Android 层调用示例（Kotlin）
```kotlin
class NativeBridge {
    companion object {
        init {
            System.loadLibrary("litemind_core")
        }
    }

    external fun createEngine(modelPath: String, threads: Int = 4): Long
    external fun destroyEngine(handle: Long)
    external fun runInference(handle: Long, bitmap: Bitmap): ByteArray
    external fun getModelInputSize(handle: Long): IntArray
}
```

```kotlin
class BiRefNetViewModel : ViewModel() {
    private val bridge = NativeBridge()
    private var engineHandle: Long = 0

    fun initEngine(context: Context) {
        if (engineHandle != 0L) return
        val modelFile = copyAssetIfNeeded(context, "birefnet_w8_nostat.mnn")
        engineHandle = bridge.createEngine(modelFile.absolutePath, threads = 4)
    }

    fun release() {
        if (engineHandle != 0L) {
            bridge.destroyEngine(engineHandle)
            engineHandle = 0
        }
    }

    fun run(bitmap: Bitmap): Bitmap {
        require(engineHandle != 0L) { "engine not initialized" }
        val mask = bridge.runInference(engineHandle, bitmap)
        return mask.toAlphaBitmap(bitmap.width, bitmap.height)
    }
}
```

`toAlphaBitmap` 可将 ByteArray 转换为 `Bitmap.Config.ALPHA_8`，再与原图合成 RGBA 预览。

## 本地调试
```bash
cd jni
cmake -B build -S . \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DANDROID_NDK=$ANDROID_NDK_HOME \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
cmake --build build
```

Android Gradle 插件会自动调用该配置，无需手动复制 `.so` 文件。
