#include <iostream>
#include <png++/png.hpp>
#include <cstdlib>
#include <cmath>
#include <numbers>

int *greyscale(png::pixel_buffer<png::rgb_pixel> image, int width, int height)
{
    int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            ret[x + y * width] = static_cast<int>(image[y][x].red + image[y][x].green + image[y][x].blue) / 3;
    return ret;
}

float *init_gaussian_kernel(int kernel_size, float sigma = 1.0f)
{
    float *ret = static_cast<float *>(std::malloc(sizeof(float) * kernel_size * kernel_size));
    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++)
        {
            float x = i - kernel_size / 2;
            float y = j - kernel_size / 2;
            ret[i * kernel_size + j] = std::exp2f(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * sigma * sigma * std::numbers::pi);
        }
    return ret;
}

int *smoothing(int *greyscale_image, int width, int height, int kernel_size = 5)
{
    float *kernel = init_gaussian_kernel(kernel_size);
    int ks2 = kernel_size / 2;

    int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));
    for (int x = ks2; x < width - ks2; x++)
        for (int y = ks2; y < height - ks2; y++)
        {
            float v = 0;
            for (int i = -ks2; i <= ks2; i++)
                for (int j = -ks2; j <= ks2; j++)
                {
                    int cx = x + i;
                    int cy = y + j;
                    int ci = i + ks2;
                    int cj = j + ks2;
                    v += kernel[cj * kernel_size + ci] * greyscale_image[cy * width + cx];
                }
            ret[x + y * width] = static_cast<int>(v);
        }

    free(kernel);
    free(greyscale_image);

    return ret;
}

int *compute_difference(int *ref_smoothed, int *modified_smoothed, int width, int height)
{
    int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
            ret[y * width + x] = std::abs(ref_smoothed[y * width + x] - modified_smoothed[y * width + x]);

    free(modified_smoothed);

    return ret;
}

void pipeline(int *ref_smoothed, png::pixel_buffer<png::rgb_pixel> modified, int width, int height)
{
    // 1.Greyscale
    auto modified_greyscale = greyscale(modified, width, height);

    // 2.Smooth (gaussian filter)
    auto modified_smoothed = smoothing(modified_greyscale, width, height);

    // 3.Difference
    auto difference = compute_difference(ref_smoothed, modified_smoothed, width, height);
    (void) difference;
    // 4.Closing/opening with disk or rectangle
    // 5.1.Thresh image
    // 5.2.Lakes
    // 6.Output Json
    /*
    Json format example
    {
        "input-0001.jpg" : [
            [0, 0, 10, 10],
            [15, 15, 42, 42]
        ],
        "input-0002.jpg" : [],
        "input-0003.jpg" : [
            [0, 0, 10, 10],
            [15, 15, 42, 42],
            [51, 42, 69, 99]
        ]
    }
    */
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./main image_ref.png image1.png [imageN.png]" << std::endl;
        return 1;
    }

    png::image<png::rgb_pixel> ref(argv[1]);

    int width = ref.get_width();
    int height = ref.get_height();

    auto ref_greyscale = greyscale(ref.get_pixbuf(), width, height);
    auto ref_smoothed = smoothing(ref_greyscale, width, height);

    for (int i = 2; i < argc; i++)
    {
        png::image<png::rgb_pixel> modified(argv[i]);
        pipeline(ref_smoothed, modified.get_pixbuf(), width, height);
    }

    free(ref_smoothed);
}
