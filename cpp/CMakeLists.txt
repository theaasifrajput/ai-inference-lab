cmake_minimum_required(VERSION 3.20)

project(ai_inference_cpp
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ai_inference
    src/main.cpp
)

target_include_directories(ai_inference
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)