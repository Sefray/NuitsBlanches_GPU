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
  void my_cuda_mem_copy(const int* h, int* d, size_t n)
  {
    int rc = cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
    if (rc)
      errx(1, "Fail buffer copy to device");
  }

  void my_cuda_mem_set(int* d, int v, size_t n)
  {
    int rc = cudaMemset((void*)d, 0, n);
    if (rc)
      errx(1, "Fail buffer copy to device");
  }

  int* malloc_and_copy(const int* h, int width, int height)
  {
    int* d = my_cuda_malloc(sizeof(int) * width * height);
    my_cuda_mem_copy(h, d, sizeof(int) * width * height);
    return d;
  }

  void my_cuda_free(int* d)
  {
    cudaError_t rc = cudaSuccess;
    rc             = cudaFree(d);
    if (rc)
      errx(1, "Fail to free memory");
  }

  int* my_cuda_malloc(size_t n)
  {
    int* d_out;
    int  rc = cudaMalloc(&d_out, n);
    if (rc)
      errx(1, "Fail buffer allocation for d_out");
    return d_out;
  }

  int* my_cuda_calloc(size_t n)
  {
    int* d_out = my_cuda_malloc(n);
    my_cuda_mem_set(d_out, 0, n);
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
      std::free(h_greyscale);

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
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel, int* d_buffer_A,
                                        int* d_buffer_B)
    {
      // 1.Greyscale
      auto h_greyscale = cpu::greyscale(h_input, width, height);

      // Buffer Allocation
      my_cuda_mem_copy(h_greyscale, d_buffer_A, sizeof(int) * width * height);
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);

      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);

      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, h_greyscale, width, height, minimum_pixel);

      std::free(h_greyscale);

      return components;
    }
  } // namespace two
} // namespace gpu