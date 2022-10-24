#include "pipeline.hh"

#include <cstdlib>
#include <cmath>
#include <numbers>
#include <set>

#include "debug/debug.hh"

namespace cpu
{
        std::set<std::vector<int>> pipeline(int *ref_smoothed, png::pixel_buffer<png::rgb_pixel> modified, int width, int height)
        {
                // 1.Greyscale
                auto modified_greyscale = greyscale(modified, width, height);
#ifndef NDEBUG
                save_img(modified_greyscale, width, height, "greyscaled.png");
#endif
                // 2.Smooth (gaussian filter)
                auto modified_smoothed = smoothing(modified_greyscale, width, height);
#ifndef NDEBUG
                save_img(modified_smoothed, width, height, "greyscale_smoothed.png");
#endif
                // 3.Difference
                auto diff = compute_difference(ref_smoothed, modified_smoothed, width, height);
#ifndef NDEBUG
                save_img(diff, width, height, "diff.png");
#endif
                // 4.Closing/opening with disk or rectangle
                auto img = closing_opening(diff, width, height);
#ifndef NDEBUG
                save_img(img, width, height, "closed_opened.png");
                compute_and_display_histogramme(img, width, height);
#endif
                // 5.1.Thresh image
                auto threshold = 30;
                binary_image(img, width, height, threshold);
#ifndef NDEBUG
                save_img(img, width, height, "binary.png", 255);
#endif
                // 5.2.Lakes
                auto components = get_connected_components(img, width, height);

                std::free(img);

                return components;
        }
}

