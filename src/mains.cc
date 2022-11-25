#include "mains.hh"

#include <cuda_runtime.h>
#include <tuple>

using json = nlohmann::json;

json main_cpu(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
              int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
              int minimum_pixel)
{
  auto ref_greyscale = cpu::greyscale(ref, width, height);
  auto ref_smoothed  = cpu::smoothing(ref_greyscale, width, height, kernel_size);

  json ret;
  for (size_t image_index = 0; image_index < images.size(); image_index++)
  {
    auto [name, image] = images[image_index];
    ret[name] = cpu::pipeline(ref_smoothed, image, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                              binary_threshold, minimum_pixel);
  }

  std::free(ref_smoothed);

  return ret;
}
json main_gpu_1(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel)
{
  using namespace gpu::one;

  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  auto d_ref_smoothed  = smoothing(d_ref_greyscale, width, height, kernel_size);

  json ret;
  for (size_t image_index = 0; image_index < images.size(); image_index++)
  {
    auto [name, image] = images[image_index];
    ret[name] = pipeline(d_ref_smoothed, image, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                         binary_threshold, minimum_pixel);
  }

  std::free(h_ref_greyscale);
  gpu::my_cuda_free(d_ref_smoothed);

  return ret;
}

template <typename T>
json main_gpu_buffer(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width,
                     int height, int kernel_size, int kernel_size_opening, int kernel_size_closing,
                     int binary_threshold, int minimum_pixel, T p)
{
  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  std::free(h_ref_greyscale);

  auto d_ref_smoothed = gpu::one::smoothing(d_ref_greyscale, width, height, kernel_size);

  int*           d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);
  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));

  json ret;
  for (size_t image_index = 0; image_index < images.size(); image_index++)
  {
    auto [name, image] = images[image_index];
    ret[name]          = p(d_ref_smoothed, image, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                           binary_threshold, minimum_pixel, d_buffer_uc, d_buffer_A, d_buffer_B);
  }

  gpu::my_cuda_free(d_ref_smoothed);

  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);

  return ret;
}

json main_gpu_2(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel)
{
  using namespace gpu::one::two;
  return main_gpu_buffer(images, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                         binary_threshold, minimum_pixel, pipeline);
}

json main_gpu_3(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel)
{
  using namespace gpu::one::two::three;
  return main_gpu_buffer(images, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                         binary_threshold, minimum_pixel, pipeline);
}

json main_gpu_4(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel)
{
  using namespace gpu::one::two::three::four;
  return main_gpu_buffer(images, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                         binary_threshold, minimum_pixel, pipeline);
}

json main_gpu_5(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel)
{
  using namespace gpu::one::two::three::four::five;
  return main_gpu_buffer(images, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                         binary_threshold, minimum_pixel, pipeline);
}

json main_gpu_6(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel)
{
  using namespace gpu::one::two::three::four::five::six;

  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  std::free(h_ref_greyscale);

  auto d_ref_smoothed = gpu::one::smoothing(d_ref_greyscale, width, height, kernel_size);

  int*           d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);
  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));

  // Streams creation
  std::vector<cudaStream_t> streams;
  for (size_t i = 0; i < nb_stream; i++)
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }

  json ret;
  for (size_t image_index = 0; image_index < images.size(); image_index++)
  {
    auto [name, image] = images[image_index];
    ret[name] = pipeline(d_ref_smoothed, image, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                         binary_threshold, minimum_pixel, d_buffer_uc, d_buffer_A, d_buffer_B, streams);
  }

  gpu::my_cuda_free(d_ref_smoothed);

  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);

  return ret;
}