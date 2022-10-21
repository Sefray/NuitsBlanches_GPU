#include "pipeline_cpu.hh"

#include <cstdlib>
#include <cmath>
#include <numbers>
#include <set>

#include "debug.hh"

namespace cpu::internal
{
    int *compute_difference(int *ref_smoothed, int *modified_smoothed, int width, int height)
    {
        int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                ret[y * width + x] = std::abs(ref_smoothed[y * width + x] - modified_smoothed[y * width + x]);

        std::free(modified_smoothed);

        return ret;
    }

    int *create_mask(int kernel_size, enum mask_type type)
    {
        int *ret = static_cast<int *>(std::calloc(kernel_size * kernel_size, sizeof(int)));

        switch (type)
        {
        case square:
            for (int x = 0; x < kernel_size; x++)
                for (int y = 0; y < kernel_size; y++)
                    ret[y * kernel_size + x] = 1;
            break;
        }

        return ret;
    }

    template <typename T>
    int *kernel_func(int *img, int width, int height, int *kernel, int kernel_size, T func)
    {
        int *ret = static_cast<int *>(std::calloc(width * height, sizeof(int)));

        int ks2 = kernel_size / 2;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                int v = kernel[ks2 * kernel_size + ks2] * img[y * width + x];

                for (int i = -ks2; i <= ks2; i++)
                {
                    int cx = x + i;
                    if (cx < 0 || cx >= width)
                        continue;

                    for (int j = -ks2; j <= ks2; j++)
                    {
                        int cy = y + j;
                        if (cy < 0 || cy >= height)
                            continue;

                        int ci = i + ks2;
                        int cj = j + ks2;

                        v = func(kernel[cj * kernel_size + ci] * img[cy * width + cx], v);
                    }
                }

                ret[y * width + x] = v;
            }

        std::free(img);

        return ret;
    }

    int *dilatation(int *img, int width, int height, int *kernel, int kernel_size)
    {
        auto lambda = [](int a, int b)
        { return a < b ? b : a; };
        return kernel_func(img, width, height, kernel, kernel_size, lambda);
    }

    int *erosion(int *img, int width, int height, int *kernel, int kernel_size)
    {
        auto lambda = [](int a, int b)
        { return a < b ? a : b; };
        return kernel_func(img, width, height, kernel, kernel_size, lambda);
    }

    int *closing_opening(int *img, int width, int height, int kernel_size_opening, int kernel_size_closing)
    {
        // Closing
        auto mask = create_mask(kernel_size_closing);
        display_img(img, width, height);
        auto a = erosion(img, width, height, mask, kernel_size_closing);
        display_img(a, width, height);
        auto b = dilatation(a, width, height, mask, kernel_size_closing);
        display_img(b, width, height);
        std::free(mask);

        // Opening
        mask = create_mask(kernel_size_opening);
        auto c = dilatation(b, width, height, mask, kernel_size_opening);
        display_img(c, width, height);
        auto ret = erosion(c, width, height, mask, kernel_size_opening);
        display_img(ret, width, height);
        std::free(mask);

        return ret;
    }

    void binary_image(int *image, int width, int height, int threshold)
    {
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                image[y * width + x] = image[y * width + x] < threshold ? -1 : 0;
    }

    struct Box
    {
        int x;
        int y;
        int width;
        int height;
        int size;
    };

    void rec_lakes(int *image, int x, int y, int width, int height, int value, Box &box)
    {
        if (image[y * width + height] == 0)
        {
            image[y * width + height] = value;
            rec_lakes(image, x - 1, y, width, height, value, box);
            rec_lakes(image, x + 1, y, width, height, value, box);
            rec_lakes(image, x - 1, y + 1, width, height, value, box);

            box.x = std::min(box.x, x);
            box.y = std::min(box.y, y);
            box.height = std::max(y - box.y, box.height);
            box.width = std::max(x - box.x, box.width);
            box.size += 1;
        }
    }

    std::set<std::vector<int>> lakes(int *image, int width, int height, int minimum_pixel)
    {
        std::set<std::vector<int>> boxes;
        for (int x = 0; x < width; x++)
            for (int y = 0; y < width; y++)
                if (image[y * width + x] == 0)
                {
                    Box box = {.x = x, .y = y, .width = 1, .height = 1, .size = 0};
                    rec_lakes(image, x, y, width, height, y * width + x, box);
                    if (box.size > minimum_pixel)
                        boxes.insert({box.x, box.y, box.width, box.height});
                }

        return boxes;
    }
}

namespace cpu
{
    using namespace cpu::internal;

    int *greyscale(png::pixel_buffer<png::rgb_pixel> image, int width, int height)
    {
        int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                // TODO: Tenter de changer les coeff de conversion
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

    int *smoothing(int *greyscale_image, int width, int height, int kernel_size)
    {
        assert(kernel_size % 2 == 1);

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

        std::free(kernel);
        std::free(greyscale_image);

        return ret;
    }

    std::set<std::vector<int>> pipeline(int *ref_smoothed, png::pixel_buffer<png::rgb_pixel> modified, int width, int height)
    {
        // 1.Greyscale
        auto modified_greyscale = greyscale(modified, width, height);

        // 2.Smooth (gaussian filter)
        auto modified_smoothed = smoothing(modified_greyscale, width, height);

        // 3.Difference
        auto diff = compute_difference(ref_smoothed, modified_smoothed, width, height);

        // 4.Closing/opening with disk or rectangle
        auto img = closing_opening(diff, width, height);

        // 5.1.Thresh image
        auto threshold = 10;
        binary_image(img, width, height, threshold);
        // 5.2.Lakes
        auto components = lakes(img, width, height);
        // 6.Output Json
        return components;
    }
}