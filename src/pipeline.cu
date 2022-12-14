#include "pipeline.cuh"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <set>

#include "debug/debug.hh"
#include <err.h>

namespace gpu
{
  void my_cuda_mem_set(int* d, int v, size_t n)
  {
    int rc = cudaMemset((void*)d, 0, n);
    if (rc)
      errx(1, "Fail buffer copy to device");
  }

  int* malloc_and_copy(const int* h, int width, int height)
  {
    int* d  = my_cuda_malloc(sizeof(int) * width * height);
    int  rc = cudaMemcpy(d, h, sizeof(int) * width * height, cudaMemcpyHostToDevice);
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
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel)
    {
      unsigned char* d_input;

      int rc = cudaMalloc(&d_input, sizeof(unsigned char) * width * height * 3);
      if (rc)
        errx(1, "Error in buffer_uc allocation");
      rc = cudaMemcpy(d_input, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
      if (rc)
        errx(1, "Error in buffer_uc copy");

      // 1.Greyscale
      auto d_greyscale = greyscale(d_input, width, height);
#ifndef NDEBUG
      save_img_gpu(d_greyscale, width, height, "greyscaled_gpu.png");
#endif

      // 2.Smooth (gaussian filter)
      auto d_smoothed = smoothing(d_greyscale, width, height, kernel_size);
#ifndef NDEBUG
      save_img_gpu(d_smoothed, width, height, "smoothed_gpu.png");
#endif
      // 3.Difference
      auto d_diff = compute_difference(d_ref_in, d_smoothed, width, height);
#ifndef NDEBUG
      save_img_gpu(d_diff, width, height, "diff_gpu.png");
#endif
      // 4.Closing/opening with disk or rectangle
      auto d_closed_opened = closing_opening(d_diff, width, height, kernel_size_opening, kernel_size_closing);
#ifndef NDEBUG
      save_img_gpu(d_closed_opened, width, height, "closed_opened_gpu.png");
#endif

      int* d_image_values = my_cuda_malloc(sizeof(int) * width * height);
      rc = cudaMemcpy(d_image_values, d_closed_opened, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      if (rc)
        errx(1, "Error memcpy h->d d_image_values");

      // 5.1.Thresh image
      binary_image(d_closed_opened, width, height, binary_threshold);
#ifndef NDEBUG
      save_img_gpu(d_closed_opened, width, height, "binary_gpu.png", 255);
#endif

      // 5.2.Lakes
      auto components =
          get_connected_components(d_closed_opened, d_image_values, width, height, high_pick_threshold, minimum_pixel);

      cudaFree(d_closed_opened);

      return components;
    }
  } // namespace one

  namespace one::two
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel, unsigned char* d_buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, int* d_buffer_image_values,
                                        float* d_kernel_smooth)
    {
      cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

      // 1.Greyscale
      greyscale(d_buffer_uc, d_buffer_A, width, height);

      // Buffer Allocation
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);

      cudaMemcpy(d_buffer_image_values, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);

      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, d_buffer_image_values, width, height,
                                                 high_pick_threshold, minimum_pixel);

      return components;
    }
  } // namespace one::two

  namespace one::two::three
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel, unsigned char* d_buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, int* d_buffer_image_values,
                                        float* d_kernel_smooth)
    {
      cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

      // 1.Greyscale
      greyscale(d_buffer_uc, d_buffer_A, width, height);

      // Buffer Allocation
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      three::closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);

      // 5
      cudaMemcpy(d_buffer_image_values, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);
      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, d_buffer_image_values, width, height,
                                                 high_pick_threshold, minimum_pixel);

      return components;
    }
  } // namespace one::two::three

  namespace one::two::three::four
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel, unsigned char* d_buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, int* d_buffer_image_values,
                                        float* d_kernel_smooth)
    {
      cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

      // 1.Greyscale
      greyscale(d_buffer_uc, d_buffer_A, width, height);

      // Buffer Allocation
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);

      // 5
      cudaMemcpy(d_buffer_image_values, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);
      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, d_buffer_image_values, width, height,
                                                 high_pick_threshold, minimum_pixel);

      return components;
    }
  } // namespace one::two::three::four
  namespace one::two::three::four::five
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel, unsigned char* d_buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, int* d_buffer_image_values,
                                        float* d_kernel_smooth)
    {
      cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

      // 1.Greyscale
      greyscale(d_buffer_uc, d_buffer_A, width, height);

      // Buffer Allocation
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing);

      // 5
      cudaMemcpy(d_buffer_image_values, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);
      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, d_buffer_image_values, width, height,
                                                 high_pick_threshold, minimum_pixel);

      return components;
    }
  } // namespace one::two::three::four::five

  namespace one::two::three::four::five::six
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel, unsigned char* d_buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, int* d_buffer_image_values,
                                        float* d_kernel_smooth, std::vector<cudaStream_t>& streams)
    {
      cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

      // 1.Greyscale
      greyscale(d_buffer_uc, d_buffer_A, width, height);

      // Buffer Allocation
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing, streams);

      // 5
      cudaMemcpy(d_buffer_image_values, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);
      // 5.2.Lakes
      auto components = get_connected_components(d_buffer_A, d_buffer_B, d_buffer_image_values, width, height,
                                                 high_pick_threshold, minimum_pixel);

      return components;
    }
  } // namespace one::two::three::four::five::six
  namespace one::two::three::four::five::six::seven
  {
    std::set<std::vector<int>> pipeline(int* d_ref_in, unsigned char* h_input, int width, int height, int kernel_size,
                                        int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                                        int high_pick_threshold, int minimum_pixel, unsigned char* d_buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, int* d_buffer_image_values,
                                        float* d_kernel_smooth, std::vector<cudaStream_t>& streams)
    {
      cudaMemcpy(d_buffer_uc, h_input, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

      // 1.Greyscale
      greyscale(d_buffer_uc, d_buffer_A, width, height);

      // Buffer Allocation
      my_cuda_mem_set(d_buffer_B, 0, sizeof(int) * width * height);

      // 2.Smooth (gaussian filter)
      smoothing(d_buffer_A, d_buffer_B, width, height, d_kernel_smooth, kernel_size);

      // 3.Difference
      compute_difference(d_ref_in, d_buffer_B, d_buffer_A, width, height);

      // 4.Closing/opening with disk or rectangle
      closing_opening(d_buffer_A, d_buffer_B, width, height, kernel_size_opening, kernel_size_closing, streams);

      // 5
      cudaMemcpy(d_buffer_B, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToDevice);
      // 5.1.Thresh image
      binary_image(d_buffer_A, width, height, binary_threshold);
      // 5.2.Lakes
      auto components =
          get_connected_components(d_buffer_A, d_buffer_B, width, height, high_pick_threshold, minimum_pixel);

      return components;
    }
  } // namespace one::two::three::four::five::six::seven
} // namespace gpu