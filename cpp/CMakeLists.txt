cmake_minimum_required(VERSION 3.20)

project(ai_inference_cpp
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

add_executable(ai_inference
    src/main.cpp
    src/preprocessing.cpp
    src/inference.cpp
    src/postprocessing.cpp
    src/postprocessing.cpp
    src/utils.cpp
)

target_include_directories(ai_inference
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ai_inference
    PRIVATE
        ${OpenCV_LIBS}
        onnxruntime::onnxruntime
)