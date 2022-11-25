cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES "61")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -W -Wall -Wextra -pedantic -fsanitize=address -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -W -Wall -Wextra -pedantic -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -W -O3")
set(CMAKE_CUDA_STANDARD "14")

find_package(PNG REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(lib)
target_compile_features(lib PUBLIC cxx_std_20)

target_sources(lib PRIVATE    
    src/pipeline.hh
    src/pipeline.cuh
    src/pipeline.cc
    src/pipeline.cu

    src/greyscale/greyscale.cc
    src/greyscale/greyscale.cu
    src/smoothing/smoothing.cc
    src/smoothing/smoothing.cu
    src/difference/diff.cc
    src/difference/diff.cu
    src/closing_opening/closing_opening.cc
    src/closing_opening/closing_opening.cu
    src/connected_components/binary.cc
    src/connected_components/binary.cu
    src/connected_components/connected_components.cc
    src/connected_components/connected_components.cu

    src/debug/debug.cc
    src/debug/debug.cu
    src/debug/debug.hh

    src/mains.cc
    src/mains.hh
)

include_directories("src")

### Main ###

add_executable(main src/main.cc)
target_compile_features(main PUBLIC cxx_std_20)

target_link_libraries(main PRIVATE lib ${OpenCV_LIBS} PNG::PNG nlohmann_json::nlohmann_json)


### Test ###

# set_target_properties(lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test
  tests/test_cpu.cc
)

target_link_libraries(test GTest::gtest_main lib PNG::PNG)


### Bench ###

add_executable(bench bench/bench.cc)
target_compile_features(bench PUBLIC cxx_std_20)
target_link_libraries(bench PRIVATE lib benchmark::benchmark ${OpenCV_LIBS} PNG::PNG)

add_executable(bench_unit bench/bench_unit.cc)
target_compile_features(bench_unit PUBLIC cxx_std_20)
target_link_libraries(bench_unit PRIVATE lib benchmark::benchmark ${OpenCV_LIBS})