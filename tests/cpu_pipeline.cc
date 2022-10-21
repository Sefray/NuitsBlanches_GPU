#include <gtest/gtest.h>

#include <cstdlib>

#include "src/pipeline_cpu.hh"

using namespace cpu;
using namespace cpu::internal;

int *malloc_img(int *ori, int width, int height)
{
  int *img = static_cast<int *>(std::malloc(sizeof(int) * width * height));
  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      img[y * width + x] = ori[y * width + x];
  return img;
}

void check_img(int *out, int *ref, int width, int height)
{
  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      EXPECT_EQ(out[y * width + x], ref[y * width + x]);
}

TEST(Opening_Closing, erosion_single)
{
  int size_mask = 3;
  auto mask = create_mask(size_mask);

  int img_width = 5;
  int img_height = 5;
  int ori[] = {0, 0, 0, 0, 0,
               0, 1, 1, 1, 0,
               0, 1, 1, 1, 0,
               0, 1, 1, 1, 0,
               0, 0, 0, 0, 0};

  auto out = erosion(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 0, 0};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}
