#include "pipeline.hh"

namespace cpu
{
  int* greyscale(png::pixel_buffer<png::rgb_pixel> image, int width, int height)
  {
    int* ret = static_cast<int*>(std::malloc(sizeof(int) * width * height));
    for (int x = 0; x < width; x++)
      for (int y = 0; y < height; y++)
        ret[x + y * width] =
            static_cast<int>(0.2126 * image[y][x].red + 0.7152 * image[y][x].green + 0.0722 * image[y][x].blue);
    return ret;
  }
} // namespace cpu