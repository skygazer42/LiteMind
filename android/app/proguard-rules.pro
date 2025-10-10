# Keep MNN JNI entry points and native interfaces
-keep class com.litemind.app.** { *; }
-keep class com.litemind.jni.** { *; }
-keepclassmembers class * {
    native <methods>;
}
