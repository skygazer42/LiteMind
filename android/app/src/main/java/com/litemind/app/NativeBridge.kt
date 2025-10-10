package com.litemind.app

import android.graphics.Bitmap

/**
 * Kotlin 层的 JNI 桥接封装，对应 `jni/src/litemind_jni.cpp` 中暴露的方法。
 */
object NativeBridge {

    init {
        System.loadLibrary("litemind_core")
    }

    @JvmStatic external fun createEngine(modelPath: String, threads: Int = 4): Long
    @JvmStatic external fun destroyEngine(handle: Long)
    @JvmStatic external fun runInference(handle: Long, bitmap: Bitmap): ByteArray
    @JvmStatic external fun getModelInputSize(handle: Long): IntArray
}
