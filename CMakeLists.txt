cmake_minimum_required(VERSION 3.20)

project(
    AIInferenceLab
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AILAB_BUILD_TESTS ON)
option(AILAB_BUILD_BENCHMARKS ON)

add_subdirectory(src)
add_subdirectory(app)

if(AILAB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(AILAB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()