#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BiRefNetEngine.h"

namespace {

constexpr const char* kLogTag = "LiteMindJNI";

[[nodiscard]] std::string JStringToUtf8(JNIEnv* env, jstring value) {
    if (!value) {
        return {};
    }
    const char* utf = env->GetStringUTFChars(value, nullptr);
    std::string result(utf ? utf : "");
    if (utf) {
        env->ReleaseStringUTFChars(value, utf);
    }
    return result;
}

void ThrowIfException(JNIEnv* env, const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception& ex) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "Exception: %s", ex.what());
        jclass exception_class = env->FindClass("java/lang/RuntimeException");
        if (exception_class) {
            env->ThrowNew(exception_class, ex.what());
        }
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag, "Unknown native exception");
        jclass exception_class = env->FindClass("java/lang/RuntimeException");
        if (exception_class) {
            env->ThrowNew(exception_class, "Native error");
        }
    }
}

inline litemind::BiRefNetEngine* FromHandle(jlong handle) {
    return reinterpret_cast<litemind::BiRefNetEngine*>(handle);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_litemind_app_NativeBridge_createEngine(JNIEnv* env,
                                                jobject /*thiz*/,
                                                jstring model_path,
                                                jint threads) {
    jlong handle = 0;
    ThrowIfException(env, [&]() {
        auto engine = std::make_unique<litemind::BiRefNetEngine>();
        engine->Initialize(JStringToUtf8(env, model_path), threads);
        handle = reinterpret_cast<jlong>(engine.release());
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "BiRefNet engine created");
    });
    return handle;
}

extern "C" JNIEXPORT void JNICALL
Java_com_litemind_app_NativeBridge_destroyEngine(JNIEnv* env,
                                                 jobject /*thiz*/,
                                                 jlong handle) {
    ThrowIfException(env, [&]() {
        auto* engine = FromHandle(handle);
        delete engine;
        __android_log_print(ANDROID_LOG_INFO, kLogTag, "BiRefNet engine destroyed");
    });
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_litemind_app_NativeBridge_runInference(JNIEnv* env,
                                                jobject /*thiz*/,
                                                jlong handle,
                                                jobject bitmap) {
    jbyteArray result = nullptr;
    ThrowIfException(env, [&]() {
        auto* engine = FromHandle(handle);
        if (!engine) {
            throw std::runtime_error("Engine handle is null. Did you call createEngine()?");
        }
        if (!bitmap) {
            throw std::invalid_argument("Bitmap is null.");
        }

        AndroidBitmapInfo info;
        if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
            throw std::runtime_error("Failed to query bitmap info.");
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            throw std::runtime_error("Bitmap must be RGBA_8888.");
        }

        void* pixels = nullptr;
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
            throw std::runtime_error("Failed to lock bitmap pixels.");
        }

        litemind::Size src_size{static_cast<int>(info.width), static_cast<int>(info.height)};
        litemind::Size dst_size = src_size;  // 输出掩码保持原图尺寸
        std::vector<uint8_t> mask;
        try {
            mask = engine->Run(static_cast<uint8_t*>(pixels), src_size, dst_size);
        } catch (...) {
            AndroidBitmap_unlockPixels(env, bitmap);
            throw;
        }
        AndroidBitmap_unlockPixels(env, bitmap);

        result = env->NewByteArray(static_cast<jsize>(mask.size()));
        if (!result) {
            throw std::runtime_error("Failed to allocate result byte array.");
        }
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(mask.size()),
                                reinterpret_cast<const jbyte*>(mask.data()));
    });
    return result;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_litemind_app_NativeBridge_getModelInputSize(JNIEnv* env,
                                                     jobject /*thiz*/,
                                                     jlong handle) {
    jintArray result = env->NewIntArray(2);
    if (!result) {
        return nullptr;
    }
    jint buffer[2] = {0, 0};
    ThrowIfException(env, [&]() {
        auto* engine = FromHandle(handle);
        if (!engine) {
            throw std::runtime_error("Engine handle is null.");
        }
        const auto size = engine->model_input_size();
        buffer[0] = size.width;
        buffer[1] = size.height;
        env->SetIntArrayRegion(result, 0, 2, buffer);
    });
    return result;
}
