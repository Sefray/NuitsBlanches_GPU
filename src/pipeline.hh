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
  GPU_5,
  GPU_6,
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

  int* create_mask(int kernel_size);
  int* dilatation(int* img, int width, int height, int* kernel, int kernel_size);
  int* erosion(int* img, int width, int height, int* kernel, int kernel_size);
  int* closing_opening(int* img, int width, int height, int kernel_size_opening, int kernel_size_closing);

  void binary_image(int* image, int width, int height, int threshold);

  std::set<std::vector<int>> compute_find(int* image, int width, int height, int minimum_pixel, int nb_boxes);
  std::set<std::vector<int>> get_connected_components(int* image, int width, int height, int minimum_pixel);

  std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                      int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                      int binary_threshold, int minimum_pixel);
} // namespace cpu
