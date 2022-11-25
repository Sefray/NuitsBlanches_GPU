#include "pipeline.hh"

#include <cstdlib>

namespace cpu
{
  int* create_mask(int kernel_size)
  {
    int* ret = static_cast<int*>(std::calloc(kernel_size * kernel_size, sizeof(int)));

    for (int x = 0; x < kernel_size; x++)
      for (int y = 0; y < kernel_size; y++)
        ret[y * kernel_size + x] = 1;

    return ret;
  }

  template <typename T>
  int* kernel_func(int* img, int width, int height, int* kernel, int kernel_size, T func)
  {
    int* ret = static_cast<int*>(std::calloc(width * height, sizeof(int)));

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

  int* dilatation(int* img, int width, int height, int* kernel, int kernel_size)
  {
    auto lambda = [](int a, int b) { return a < b ? b : a; };
    return kernel_func(img, width, height, kernel, kernel_size, lambda);
  }

  int* erosion(int* img, int width, int height, int* kernel, int kernel_size)
  {
    auto lambda = [](int a, int b) { return a < b ? a : b; };
    return kernel_func(img, width, height, kernel, kernel_size, lambda);
  }

  int* closing_opening(int* img, int width, int height, int kernel_size_opening, int kernel_size_closing)
  {
    assert(kernel_size_opening % 2 == 1);
    assert(kernel_size_closing % 2 == 1);

    // Closing
    auto mask = create_mask(kernel_size_closing);
    auto a    = dilatation(img, width, height, mask, kernel_size_closing);
#ifndef NDEBUG
    save_img(a, width, height, "closing_dillation.png");
#endif
    auto b = erosion(a, width, height, mask, kernel_size_closing);
    std::free(mask);

#ifndef NDEBUG
    save_img(b, width, height, "closing_erosion.png");
#endif

    // Opening
    mask   = create_mask(kernel_size_opening);
    auto c = erosion(b, width, height, mask, kernel_size_opening);
#ifndef NDEBUG
    save_img(c, width, height, "opening_erosion.png");
#endif
    auto ret = dilatation(c, width, height, mask, kernel_size_opening);
    std::free(mask);

    return ret;
  }
} // namespace cpu