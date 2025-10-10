#include <jni.h>

#include <string>

#include "LiteMind.h"

namespace {

constexpr const char* kHelloMessage = "LiteMind JNI pipeline ready";

}  // namespace

namespace litemind {

std::string hello() {
    return kHelloMessage;
}

}  // namespace litemind

extern "C" JNIEXPORT jstring JNICALL
Java_com_litemind_app_NativeBridge_stringFromJNI(JNIEnv* env, jobject /* thiz */) {
    const std::string message = litemind::hello();
    return env->NewStringUTF(message.c_str());
}
