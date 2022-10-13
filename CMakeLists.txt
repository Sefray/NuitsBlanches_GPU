cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(PNG REQUIRED)

add_library(lib)
target_compile_features(lib PUBLIC cxx_std_20)

target_sources(lib PRIVATE
    src/test.cu
    src/test.cuh
)
# set_target_properties(lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(main src/main.cc)
target_compile_features(main PUBLIC cxx_std_20)

target_link_libraries(main PRIVATE lib PNG::PNG)