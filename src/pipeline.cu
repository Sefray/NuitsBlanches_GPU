#include "pipeline.hh"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <set>

#include "debug/debug.hh"
#include <err.h>

namespace gpu
{
  int* malloc_and_copy(const int* h, int width, int height)
  {
    cudaError_t rc = cudaSuccess;

    int* d;

    rc = cudaMalloc((void**)&d, sizeof(int) * width * height);
    if (rc)
      errx(1, "Fail buffer allocation");

    cudaMemcpy(d, h, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    if (rc)
      errx(1, "Fail buffer copy to device");

    return d;
  }

  void my_cuda_free(int* d)
  {
    cudaError_t rc = cudaSuccess;
    rc             = cudaFree(d);
    if (rc)
      errx(1, "Fail to free memory");
  }

  int* my_cuda_calloc(size_t n)
  {
    int* d_out = my_cuda_malloc(n);
    int   rc    = cudaMemset(d_out, 0, n);
    if (rc)
      errx(1, "Fail buffer set to 0 for d_out");
    return d_out;
  }

  int* my_cuda_malloc(size_t n)
  {
    int* d_out;
    int  rc = cudaMalloc(&d_out, n);
    if (rc)
      errx(1, "Fail buffer allocation for d_out");
    return d_out;
  }

  namespace one
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, png::pixel_buffer<png::rgb_pixel> h_input, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel)
    {
      // 1.Greyscale
      auto h_greyscale = cpu::greyscale(h_input, width, height);

      int* d_greyscale = malloc_and_copy(h_greyscale, width, height);

      // 2.Smooth (gaussian filter)
      auto d_smoothed = smoothing(d_greyscale, width, height, kernel_size);

      // 3.Difference
      auto d_diff = compute_difference(d_ref_in, d_smoothed, width, height);

      // 4.Closing/opening with disk or rectangle
      auto d_closed_opened = closing_opening(d_diff, width, height, kernel_size_opening, kernel_size_closing);

      // 5.1.Thresh image
      binary_image(d_closed_opened, width, height, binary_threshold);

      // 5.2.Lakes
      auto components = get_connected_components(d_closed_opened, width, height, minimum_pixel);

      cudaFree(d_closed_opened);

      return components;
    }
  } // namespace one

  namespace two
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, png::pixel_buffer<png::rgb_pixel> h_input, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel)
    {
      cudaError_t rc = cudaSuccess;

      // 1.Greyscale
      auto h_greyscale = cpu::greyscale(h_input, width, height);

      // Buffer Allocation
      int*  d_buffer_A = malloc_and_copy(h_greyscale, width, height);
      int* d_buffer_B = my_cuda_calloc(sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, kernel_size);

#ifndef NDEBUG
      rc = cudaMemcpy(h_greyscale, d_buffer_B, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
      if (rc)
        errx(1, "Fail buffer copy to host");
      save_img(h_greyscale, width, height, "gpu_smoothed.png");
#endif

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

#ifndef NDEBUG
      rc = cudaMemcpy(h_greyscale, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
      if (rc)
        errx(1, "Fail buffer copy to host");
      save_img(h_greyscale, width, height, "gpu_diff.png");
#endif

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);
#ifndef NDEBUG
      rc = cudaMemcpy(h_greyscale, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
      if (rc)
        errx(1, "Fail buffer copy to host");
      save_img(h_greyscale, width, height, "gpu_closing_opening.png");
#endif

      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);
#ifndef NDEBUG
      rc = cudaMemcpy(h_greyscale, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
      if (rc)
        errx(1, "Fail buffer copy to host");
      save_img(h_greyscale, width, height, "gpu_binary.png", 255);
#endif

      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, h_greyscale, width, height, minimum_pixel);

      rc = cudaFree(d_buffer_A);
      if (rc)
        errx(1, "Fail to free memory");
      rc = cudaFree(d_buffer_B);
      if (rc)
        errx(1, "Fail to free memory");
      std::free(h_greyscale);

      return components;
    }
  } // namespace two
} // namespace gpu