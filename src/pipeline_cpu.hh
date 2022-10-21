#pragma once

#include <png++/png.hpp>
#include <set>
#include <vector>

namespace cpu
{
    int *greyscale(png::pixel_buffer<png::rgb_pixel> image, int width, int height);
    int *smoothing(int *greyscale_image, int width, int height, int kernel_size = 5);

    namespace internal
    {
        enum mask_type
        {
            square,
            // disk,
        };

        int *create_mask(int kernel_size, enum mask_type type = square);
        int *dilatation(int *img, int width, int height, int *kernel, int kernel_size);
        int *erosion(int *img, int width, int height, int *kernel, int kernel_size);
        int *closing_opening(int *img, int width, int height, int kernel_size_opening = 11, int kernel_size_closing = 7);
      
        void binary_image(int *image, int width, int height, int threshold);
        std::set<std::vector<int>> lakes(int *image, int width, int height, int minimum_pixel = 12);
    }

    std::set<std::vector<int>> pipeline(int *ref_smoothed, png::pixel_buffer<png::rgb_pixel> modified, int width, int height);
}
