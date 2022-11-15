#include "mains.hh"

#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

json main_cpu(std::vector<std::string> vargv, unsigned char* ref, int width, int height, int kernel_size,
              int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
              int minimum_pixel)
{
  auto ref_greyscale = cpu::greyscale(ref, width, height);
  auto ref_smoothed  = cpu::smoothing(ref_greyscale, width, height, kernel_size);

  json ret;
  for (size_t img = 0; img < vargv.size(); img++)
  {
    auto modified   = cv::imread(vargv[img], cv::IMREAD_COLOR);
    ret[vargv[img]] = cpu::pipeline(ref_smoothed, modified.data, width, height, kernel_size, kernel_size_opening,
                                    kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
  }

  std::free(ref_smoothed);

  return ret;
}

json main_gpu_1(std::vector<std::string> vargv, unsigned char* ref, int width, int height, int kernel_size,
                int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
                int minimum_pixel)
{
  using namespace gpu::one;

  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  auto d_ref_smoothed  = smoothing(d_ref_greyscale, width, height, kernel_size);

  json ret;
  for (size_t img = 0; img < vargv.size(); img++)
  {
    auto modified   = cv::imread(vargv[img], cv::IMREAD_COLOR);
    ret[vargv[img]] = pipeline(d_ref_smoothed, modified.data, width, height, kernel_size, kernel_size_opening,
                               kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
  }

  std::free(h_ref_greyscale);
  gpu::my_cuda_free(d_ref_smoothed);

  return ret;
}

json main_gpu_2(std::vector<std::string> vargv, unsigned char* ref, int width, int height, int kernel_size,
                int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
                int minimum_pixel)
{
  using namespace gpu::one::two;

  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  std::free(h_ref_greyscale);

  auto d_ref_smoothed = gpu::one::smoothing(d_ref_greyscale, width, height, kernel_size);

  int*           d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);
  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));

  json ret;
  for (size_t img = 0; img < vargv.size(); img++)
  {
    auto modified = cv::imread(vargv[img], cv::IMREAD_COLOR);
    ret[vargv[img]] =
        pipeline(d_ref_smoothed, modified.data, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                 binary_threshold, mode_cc, minimum_pixel, d_buffer_uc, d_buffer_A, d_buffer_B);
  }

  gpu::my_cuda_free(d_ref_smoothed);

  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);

  return ret;
}

json main_gpu_3(std::vector<std::string> vargv, unsigned char* ref, int width, int height, int kernel_size,
                int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
                int minimum_pixel)
{
  using namespace gpu::one::two::three;

  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  std::free(h_ref_greyscale);

  auto d_ref_smoothed = gpu::one::smoothing(d_ref_greyscale, width, height, kernel_size);

  int*           d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);
  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));

  json ret;
  for (size_t img = 0; img < vargv.size(); img++)
  {
    auto modified = cv::imread(vargv[img], cv::IMREAD_COLOR);
    ret[vargv[img]] =
        pipeline(d_ref_smoothed, modified.data, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                 binary_threshold, mode_cc, minimum_pixel, d_buffer_uc, d_buffer_A, d_buffer_B);
  }

  gpu::my_cuda_free(d_ref_smoothed);

  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);

  return ret;
}

json main_gpu_4(std::vector<std::string> vargv, unsigned char* ref, int width, int height, int kernel_size,
                int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
                int minimum_pixel)
{
  using namespace gpu::one::two::three::four;

  auto h_ref_greyscale = cpu::greyscale(ref, width, height);

  auto d_ref_greyscale = gpu::malloc_and_copy(h_ref_greyscale, width, height);
  std::free(h_ref_greyscale);

  auto d_ref_smoothed = gpu::one::smoothing(d_ref_greyscale, width, height, kernel_size);

  int*           d_buffer_A = gpu::my_cuda_malloc(sizeof(int) * width * height);
  int*           d_buffer_B = gpu::my_cuda_malloc(sizeof(int) * width * height);
  unsigned char* d_buffer_uc =
      static_cast<unsigned char*>(static_cast<void*>(gpu::my_cuda_malloc(sizeof(unsigned char) * width * height * 3)));

  json ret;
  for (size_t img = 0; img < vargv.size(); img++)
  {
    auto modified = cv::imread(vargv[img], cv::IMREAD_COLOR);
    ret[vargv[img]] =
        pipeline(d_ref_smoothed, modified.data, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                 binary_threshold, mode_cc, minimum_pixel, d_buffer_uc, d_buffer_A, d_buffer_B);
  }

  gpu::my_cuda_free(d_ref_smoothed);

  gpu::my_cuda_free(d_buffer_A);
  gpu::my_cuda_free(d_buffer_B);

  return ret;
}