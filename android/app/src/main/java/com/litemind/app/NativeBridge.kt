package com.litemind.app

/**
 * NativeBridge 负责封装 JNI 方法，后续可扩展更多推理接口。
 */
object NativeBridge {

    init {
        System.loadLibrary("litemind_core")
    }

    external fun stringFromJNI(): String
}
