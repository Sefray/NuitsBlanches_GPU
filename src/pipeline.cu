#include "pipeline.hh"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <numbers>
#include <set>

#include "debug/debug.hh"
#include <err.h>

namespace gpu
{
        int *malloc_and_copy(const int *h, int width, int height)
        {
                cudaError_t rc = cudaSuccess;

                int *d;

                rc = cudaMalloc((void **)&d, sizeof(int) * width * height);
                if (rc)
                        errx(1, "Fail buffer allocation");

                cudaMemcpy(d, h, sizeof(int) * width * height, cudaMemcpyHostToDevice);
                if (rc)
                        errx(1, "Fail buffer copy to device");

                return d;
        }

        void my_cuda_free(int *d)
        {
                cudaError_t rc = cudaSuccess;
                rc = cudaFree(d);
                if (rc)
                        errx(1, "Fail to free memory");
        }

        std::set<std::vector<int>> pipeline(int *d_ref_in, png::pixel_buffer<png::rgb_pixel> h_input, int width, int height, int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc, int minimum_pixel)
        {
                cudaError_t rc = cudaSuccess;

                // 1.Greyscale
                auto h_greyscale = cpu::greyscale(h_input, width, height);

                // Buffer Allocation
                int *d_buffer_A;
                rc = cudaMalloc(&d_buffer_A, sizeof(int) * width * height);
                if (rc)
                        errx(1, "Fail buffer allocation for A");

                int *d_buffer_B;
                rc = cudaMalloc(&d_buffer_B, sizeof(int) * width * height);
                if (rc)
                        errx(1, "Fail buffer allocation for B");
                cudaMemset(d_buffer_B, 0, sizeof(int) * width * height);

                rc = cudaMemcpy(d_buffer_A, h_greyscale, sizeof(int) * width * height, cudaMemcpyHostToDevice);
                if (rc)
                        errx(1, "Fail buffer copy to device");

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
                rc = cudaMemcpy(h_greyscale, d_buffer_A, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
                if (rc)
                        errx(1, "Fail buffer copy to host");
                rc = cudaFree(d_buffer_A);
                if (rc)
                        errx(1, "Fail to free memory");
                rc = cudaFree(d_buffer_B);
                if (rc)
                        errx(1, "Fail to free memory");

                // 5.2.Lakes
                auto components = cpu::get_connected_components(h_greyscale, width, height, mode_cc, minimum_pixel);

                // TMP
                std::free(h_greyscale);

                return components;
        }
}