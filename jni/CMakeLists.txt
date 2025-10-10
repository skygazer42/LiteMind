cmake_minimum_required(VERSION 3.22.1)

project(litemind_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(litemind_core SHARED
    src/litemind_jni.cpp
    src/BiRefNetEngine.cpp
)

target_include_directories(litemind_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 配置 MNN 头文件路径，可通过 -DLITEMIND_MNN_ROOT=<path> 覆盖
set(LITEMIND_MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN" CACHE PATH "Path to MNN prebuilt package")
set(LITEMIND_MNN_INCLUDE_DIR "${LITEMIND_MNN_ROOT}/include")
if(EXISTS "${LITEMIND_MNN_INCLUDE_DIR}")
    target_include_directories(litemind_core PRIVATE "${LITEMIND_MNN_INCLUDE_DIR}")
endif()

# 查找/导入 MNN 库，优先使用 third_party/MNN/lib/<abi>/libMNN.so
set(_mnn_candidates
    "${LITEMIND_MNN_ROOT}/lib/${ANDROID_ABI}/libMNN.so"
    "${LITEMIND_MNN_ROOT}/lib/${ANDROID_ABI}/libMNN.a"
    "${CMAKE_CURRENT_SOURCE_DIR}/../android/app/src/main/jniLibs/${ANDROID_ABI}/libMNN.so"
)
foreach(_mnn_path IN LISTS _mnn_candidates)
    if(EXISTS "${_mnn_path}")
        add_library(MNN SHARED IMPORTED GLOBAL)
        set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${_mnn_path}")
        get_filename_component(_mnn_dir "${_mnn_path}" DIRECTORY)
        target_link_directories(litemind_core PRIVATE "${_mnn_dir}")
        message(STATUS "Found MNN library: ${_mnn_path}")
        break()
    endif()
endforeach()

if(NOT TARGET MNN)
    message(WARNING "MNN library not found. Set LITEMIND_MNN_ROOT or place libMNN.so under android/app/src/main/jniLibs/${ANDROID_ABI}")
endif()

find_library(log-lib log)
find_library(graphics-lib jnigraphics)
find_library(android-lib android)

target_link_libraries(litemind_core
    PRIVATE
        ${graphics-lib}
        ${android-lib}
        ${log-lib}
)

if(TARGET MNN)
    target_link_libraries(litemind_core PRIVATE MNN)
endif()
