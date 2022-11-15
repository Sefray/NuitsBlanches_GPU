#include "pipeline.hh"

namespace cpu
{
  int* greyscale(unsigned char* image, int width, int height)
  {
    int* ret = static_cast<int*>(std::malloc(sizeof(int) * width * height));

    for (int x = 0; x < width; x++)
      for (int y = 0; y < height; y++)
      {
        auto p = x + y * width;

        // For rgb
        auto image_p = 3 * p;

        // OpenCV order {B, G, R}
        float b = image[image_p];
        float g = image[image_p + 1];
        float r = image[image_p + 2];

        ret[p] = static_cast<int>(0.2126 * r + 0.7152 * g + 0.0722 * b);
      }

    return ret;
  }
} // namespace cpu