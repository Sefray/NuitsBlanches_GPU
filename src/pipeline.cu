#include "pipeline.hh"

#include <cstdlib>
#include <cmath>
#include <numbers>
#include <set>

#include "debug.hh"

namespace gpu
{
        std::set<std::vector<int>> pipeline(int *ref_smoothed, png::pixel_buffer<png::rgb_pixel> h_input, int width, int height)
        {
                // 1.Greyscale
                auto h_greyscale = cpu::greyscale(h_input, width, height);
#ifndef NDEBUG
                save_img(h_greyscale, width, height, "greyscaled.png");
#endif

                int *d_buffer_A;
                cudaMalloc(&d_buffer_A, sizeof(int) * width * height);
                int *d_buffer_B;
                cudaMalloc(&d_buffer_B, sizeof(int) * width * height);

                cudaMemcpy(d_buffer_A, h_greyscale, sizeof(int) * width * height, cudaMemcpyHostToDevice);

                (void)d_buffer_A;
                (void)d_buffer_B;

                // 2.Smooth (gaussian filter)
                auto modified_smoothed = smoothing(h_greyscale, width, height);
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

                cudaFree(d_buffer_A);
                cudaFree(d_buffer_B);

                return components;
        }
}