cmake_minimum_required(VERSION 3.22.1)

project(litemind_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(litemind_core SHARED
    src/litemind_jni.cpp
)

target_include_directories(litemind_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_library(log-lib log)

target_link_libraries(litemind_core
    PRIVATE
        ${log-lib}
)
