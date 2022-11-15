#pragma once

#include <err.h>
#include <png++/png.hpp>
#include <set>
#include <vector>

#include "debug/debug.hh"

enum mode
{
  CPU,
  GPU_1,
  GPU_2,
  GPU_3,
  GPU_4,
};

enum mask_type
{
  square,
  disk,
};

enum mode_cc
{
  slide,
  union_find,
};

struct Box
{
  int xmin;
  int ymin;
  int xmax;
  int ymax;

  int size;
};

namespace cpu
{
  int* greyscale(unsigned char* image, int width, int height);

  float* init_gaussian_kernel(int kernel_size, float sigma = 1.0f);
  int*   smoothing(int* greyscale_image, int width, int height, int kernel_size);
  int*   compute_difference(int* ref_smoothed, int* modified_smoothed, int width, int height);

  int* create_mask(int kernel_size, enum mask_type type = square);
  int* dilatation(int* img, int width, int height, int* kernel, int kernel_size);
  int* erosion(int* img, int width, int height, int* kernel, int kernel_size);
  int* closing_opening(int* img, int width, int height, int kernel_size_opening, int kernel_size_closing);

  void binary_image(int* image, int width, int height, int threshold);

  std::set<std::vector<int>> compute_find(int* image, int width, int height, int minimum_pixel, int nb_boxes);
  std::set<std::vector<int>> get_connected_components(int* image, int width, int height, enum mode_cc mode_cc,
                                                      int minimum_pixel);

  std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                      int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                      int binary_threshold, enum mode_cc mode_cc, int minimum_pixel);
} // namespace cpu

namespace gpu
{
  int* malloc_and_copy(const int* h, int width, int height);
  void my_cuda_free(int* d);
  int* my_cuda_malloc(size_t n);
  int* my_cuda_calloc(size_t n);

  void binary_image(int* d_in_out, int width, int height, int threshold);

  namespace one
  {
    int* smoothing(int* d_in, int width, int height, int kernel_size);
    int* compute_difference(int* d_ref_in, int* d_in, int width, int height);
    int* closing_opening(int* d_A, int width, int height, int kernel_size_opening, int kernel_size_closing);
    std::set<std::vector<int>> get_connected_components(int* d_in, int width, int height, int minimum_pixel);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel);
  } // namespace one

  namespace one::two
  {
    void smoothing(int* d_in, int* d_out, int width, int height, int kernel_size);
    void compute_difference(int* d_ref_in, int* d_in, int* d_out, int width, int height);
    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing);
    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int* h, int width, int height,
                                                        int minimum_pixel);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel, int* d_buffer_A,
                                        int* d_buffer_B);
  } // namespace one::two

  namespace one::two::three
  {
    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel, int* d_buffer_A,
                                        int* d_buffer_B);
  } // namespace one::two::three

  namespace one::two::three::four
  {
    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int* h, int width, int height,
                                                        int minimum_pixel);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, enum mode_cc mode_cc, int minimum_pixel, int* d_buffer_A,
                                        int* d_buffer_B);
  } // namespace one::two::three::four
} // namespace gpu