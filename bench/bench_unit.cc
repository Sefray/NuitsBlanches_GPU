#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "pipeline.cuh"
#include "pipeline.hh"

int* get_ref_image(int kernel_size)
{
  std::string path = std::string("../data/nuits_blanches_81/nb-01.png");
  auto        ref  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = ref.cols;
  int height = ref.rows;

  auto h_ref_greyscale = cpu::greyscale(ref.data, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  std::free(h_ref_greyscale);

  auto d_ref_smoothed = gpu::one::smoothing(d_ref_greyscale, width, height, kernel_size);

  return d_ref_smoothed;
}


void BM_Greyscale(benchmark::State& state)
{
  using namespace gpu::one::two;

  std::string path = std::string("../data/nuits_blanches_81/nb-55.png");
  cv::Mat     img  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = img.cols;
  int height = img.rows;

  unsigned char* h_input = img.data;

  int* d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int* d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);

  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));
  cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

  for (auto _ : state)
  {
    greyscale(d_buffer_uc, d_buffer_A, width, height);
  }

  cudaFree(d_buffer_uc);
  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);
}

void BM_Blur(benchmark::State& state)
{
  using namespace gpu::one::two;

  int kernel_size = 5;

  std::string path = std::string("../data/nuits_blanches_81/nb-55.png");
  cv::Mat     img  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = img.cols;
  int height = img.rows;

  unsigned char* h_input = img.data;

  int*   d_buffer_A      = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*   d_buffer_B      = gpu::my_cuda_malloc(sizeof(int) * width * height);
  float* d_kernel_smooth = gpu::init_gaussian_kernel(kernel_size);

  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));
  cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

  greyscale(d_buffer_uc, d_buffer_A, width, height);

  for (auto _ : state)
  {
    smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);
  }

  cudaFree(d_buffer_uc);
  cudaFree(d_kernel_smooth);
  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);
}

void BM_Diff(benchmark::State& state)
{
  using namespace gpu::one::two;

  int kernel_size         = 5;
  int kernel_size_opening = 41;
  int kernel_size_closing = 31;
  int binary_threshold    = 20;
  int high_pick_threshold = 10;
  int minimum_pixel       = 1920 * 1080 * 0.5f / 100;

  int* d_ref_in = get_ref_image(kernel_size);

  std::string path = std::string("../data/nuits_blanches_81/nb-55.png");
  cv::Mat     img  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = img.cols;
  int height = img.rows;

  unsigned char* h_input = img.data;

  int*   d_buffer_A      = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*   d_buffer_B      = gpu::my_cuda_malloc(sizeof(int) * width * height);
  float* d_kernel_smooth = gpu::init_gaussian_kernel(kernel_size);

  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));
  cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

  greyscale(d_buffer_uc, d_buffer_A, width, height);
  gpu::my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

  smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

  for (auto _ : state)
  {
    compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);
  }

  cudaFree(d_ref_in);
  cudaFree(d_buffer_uc);
  cudaFree(d_kernel_smooth);
  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);
}

void BM_Closing_Opening(benchmark::State& state)
{
  using namespace gpu::one::two;

  int kernel_size         = 5;
  int kernel_size_opening = 41;
  int kernel_size_closing = 31;
  int binary_threshold    = 20;
  int high_pick_threshold = 10;
  int minimum_pixel       = 1920 * 1080 * 0.5f / 100;

  int* d_ref_in = get_ref_image(kernel_size);

  std::string path = std::string("../data/nuits_blanches_81/nb-55.png");
  cv::Mat     img  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = img.cols;
  int height = img.rows;

  unsigned char* h_input = img.data;

  int*   d_buffer_A      = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*   d_buffer_B      = gpu::my_cuda_malloc(sizeof(int) * width * height);
  float* d_kernel_smooth = gpu::init_gaussian_kernel(kernel_size);

  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));
  cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

  greyscale(d_buffer_uc, d_buffer_A, width, height);
  gpu::my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

  smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

  compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);
  for (auto _ : state)
  {
    gpu::one::two::three::four::five::closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening,
                                                      kernel_size_closing);
  }

  cudaFree(d_ref_in);
  cudaFree(d_buffer_uc);
  cudaFree(d_kernel_smooth);
  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);
}

void BM_Threshold(benchmark::State& state)
{
  using namespace gpu::one::two;

  int kernel_size         = 5;
  int kernel_size_opening = 41;
  int kernel_size_closing = 31;
  int binary_threshold    = 20;
  int high_pick_threshold = 10;
  int minimum_pixel       = 1920 * 1080 * 0.5f / 100;

  int* d_ref_in = get_ref_image(kernel_size);

  std::string path = std::string("../data/nuits_blanches_81/nb-55.png");
  cv::Mat     img  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = img.cols;
  int height = img.rows;

  unsigned char* h_input = img.data;

  int* d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int* d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);

  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));
  cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
  float* d_kernel_smooth = gpu::init_gaussian_kernel(kernel_size);

  greyscale(d_buffer_uc, d_buffer_A, width, height);
  gpu::my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

  smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

  compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);
  closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);
  for (auto _ : state)
  {
    gpu::one::binary_image(d_buffer_A, width, height, binary_threshold);
  }

  cudaFree(d_ref_in);
  cudaFree(d_buffer_uc);
  cudaFree(d_kernel_smooth);
  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);
}

void BM_Connectic_components(benchmark::State& state)
{
  using namespace gpu::one::two;

  int kernel_size         = 5;
  int kernel_size_opening = 41;
  int kernel_size_closing = 31;
  int binary_threshold    = 20;
  int high_pick_threshold = 10;
  int minimum_pixel       = 1920 * 1080 * 0.5f / 100;

  int* d_ref_in = get_ref_image(kernel_size);

  std::string path = std::string("../data/nuits_blanches_81/nb-55.png");
  cv::Mat     img  = cv::imread(path, cv::IMREAD_COLOR);

  int width  = img.cols;
  int height = img.rows;

  unsigned char* h_input = img.data;

  int*           d_buffer_A     = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_B     = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_C     = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_image_values = gpu::my_cuda_malloc(sizeof(int) * width * height);
  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));
  cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
  float* d_kernel_smooth = gpu::init_gaussian_kernel(kernel_size);

  greyscale(d_buffer_uc, d_buffer_A, width, height);
  gpu::my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

  smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

  compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);
  closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);

  cudaMemcpy(d_image_values, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
  gpu::one::binary_image(d_buffer_A, width, height, binary_threshold);
  for (auto _ : state)
  {
    cudaMemcpy(d_buffer_C, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
    gpu::one::two::three::four::get_connected_components(d_buffer_C, d_buffer_B, d_image_values, width, height,
                                                         high_pick_threshold, minimum_pixel);
  }

  cudaFree(d_ref_in);
  cudaFree(d_buffer_uc);
  cudaFree(d_kernel_smooth);
  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);
  gpu::my_cuda_free(d_buffer_C);
  gpu::my_cuda_free(d_image_values);
}

BENCHMARK(BM_Greyscale)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Blur)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Diff)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Closing_Opening)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Threshold)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_Connectic_components)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
