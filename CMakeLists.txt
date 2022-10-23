cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES "61")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -W -Wall -Werror -Wextra -pedantic -fsanitize=address -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -W -Wall -Werror -Wextra -pedantic -O3")

find_package(PNG REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)
find_package(benchmark REQUIRED)

add_library(lib)
target_compile_features(lib PUBLIC cxx_std_20)

target_sources(lib PRIVATE
    src/debug.cc
    src/debug.hh
    src/pipeline_cpu.cc
    src/pipeline_cpu.hh
    src/test.cu
    src/test.cuh
)

include_directories(".")

### Main ###

add_executable(main src/main.cc)
target_compile_features(main PUBLIC cxx_std_20)

target_link_libraries(main PRIVATE lib PNG::PNG nlohmann_json::nlohmann_json)


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
target_link_libraries(bench PRIVATE lib benchmark::benchmark PNG::PNG)