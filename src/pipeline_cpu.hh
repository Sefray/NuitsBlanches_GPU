#pragma once

#include <png++/png.hpp>

namespace cpu
{
    int *greyscale(png::pixel_buffer<png::rgb_pixel> image, int width, int height);
    int *smoothing(int *greyscale_image, int width, int height, int kernel_size = 5);

    void pipeline(int *ref_smoothed, png::pixel_buffer<png::rgb_pixel> modified, int width, int height);
}