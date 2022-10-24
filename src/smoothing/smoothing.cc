#include "src/pipeline.hh"

#include <cmath>
#include <numbers>

namespace cpu
{
    float *init_gaussian_kernel(int kernel_size, float sigma)
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

    int *smoothing(int *greyscale_image, int width, int height, int kernel_size)
    {
        assert(kernel_size % 2 == 1);

        float *kernel = init_gaussian_kernel(kernel_size);
        int ks2 = kernel_size / 2;

        int *ret = static_cast<int *>(std::calloc(width * height, sizeof(int)));
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

        std::free(kernel);
        std::free(greyscale_image);

        return ret;
    }
}