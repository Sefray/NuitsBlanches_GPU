#include "pipeline.hh"

namespace cpu
{
  void binary_image(int* image, int width, int height, int threshold)
  {
    for (int x = 0; x < width; x++)
      for (int y = 0; y < height; y++)
        image[y * width + x] = image[y * width + x] < threshold ? 0 : 1;
  }
} // namespace cpu