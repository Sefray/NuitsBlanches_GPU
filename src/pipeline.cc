#include "pipeline.hh"

#include <cmath>
#include <cstdlib>
#include <numbers>
#include <set>

#include "debug/debug.hh"

namespace cpu
{
  std::set<std::vector<int>> pipeline(int* ref_smoothed, png::pixel_buffer<png::rgb_pixel> modified, int width,
                                      int height, int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                      int binary_threshold, enum mode_cc mode_cc, int minimum_pixel)
  {
    // 1.Greyscale
    auto modified_greyscale = greyscale(modified, width, height);
#ifndef NDEBUG
    save_img(modified_greyscale, width, height, "greyscaled.png");
#endif
    // 2.Smooth (gaussian filter)
    auto modified_smoothed = smoothing(modified_greyscale, width, height, kernel_size);
#ifndef NDEBUG
    save_img(modified_smoothed, width, height, "greyscale_smoothed.png");
#endif
    // 3.Difference
    auto diff = compute_difference(ref_smoothed, modified_smoothed, width, height);
#ifndef NDEBUG
    save_img(diff, width, height, "diff.png");
#endif
    // 4.Closing/opening with disk or rectangle
    auto img = closing_opening(diff, width, height, kernel_size_opening, kernel_size_closing);
#ifndef NDEBUG
    save_img(img, width, height, "closed_opened.png");
    compute_and_display_histogramme(img, width, height);
#endif
    // 5.1.Thresh image
    binary_image(img, width, height, binary_threshold);
#ifndef NDEBUG
    save_img(img, width, height, "binary.png", 255);
#endif
    // 5.2.Lakes
    auto components = get_connected_components(img, width, height, mode_cc, minimum_pixel);

    std::free(img);

    return components;
  }
} // namespace cpu