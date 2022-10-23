#include <gtest/gtest.h>

#include <cstdlib>

#include "src/pipeline_cpu.hh"
#include "src/debug.hh"

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

TEST(Closing_Opening, erosion_single_square)
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

TEST(Closing_Opening, erosion_two)
{
  int size_mask = 3;
  auto mask = create_mask(size_mask);

  int img_width = 5;
  int img_height = 5;
  int ori[] = {1, 1, 1, 0, 0,
               1, 1, 1, 0, 0,
               1, 1, 1, 1, 1,
               0, 0, 1, 1, 1,
               0, 0, 1, 1, 1};

  auto out = erosion(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {1, 1, 0, 0, 0,
               1, 1, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 1, 1,
               0, 0, 0, 1, 1};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, dilatation_single_square)
{
  int size_mask = 3;
  auto mask = create_mask(size_mask);

  int img_width = 5;
  int img_height = 5;
  int ori[] = {0, 0, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 0, 0};

  auto out = dilatation(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {0, 0, 0, 0, 0,
               0, 1, 1, 1, 0,
               0, 1, 1, 1, 0,
               0, 1, 1, 1, 0,
               0, 0, 0, 0, 0};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, dilatation_two_square)
{
  int size_mask = 3;
  auto mask = create_mask(size_mask);

  int img_width = 5;
  int img_height = 5;
  int ori[] = {0, 0, 0, 0, 0,
               0, 1, 0, 0, 0,
               0, 0, 0, 0, 0,
               0, 0, 0, 1, 0,
               0, 0, 0, 0, 0};

  auto out = dilatation(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {1, 1, 1, 0, 0,
               1, 1, 1, 0, 0,
               1, 1, 1, 1, 1,
               0, 0, 1, 1, 1,
               0, 0, 1, 1, 1};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, opening5_closing3)
{
  int kernel_size_opening = 5;
  int kernel_size_closing = 3;

  int img_width = 10;
  int img_height = 10;
  int ori[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
      0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
      0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto out = closing_opening(malloc_img(ori, img_width, img_height), img_width, img_height, kernel_size_opening, kernel_size_closing);

  int ref[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  check_img(ref, out, img_width, img_height);

  std::free(out);
}

TEST(Closing_Opening, opening5_closing3_hard)
{
  int kernel_size_opening = 5;
  int kernel_size_closing = 3;

  int img_width = 20;
  int img_height = 20;
  int ori[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto out = closing_opening(malloc_img(ori, img_width, img_height), img_width, img_height, kernel_size_opening, kernel_size_closing);

  int ref[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  check_img(ref, out, img_width, img_height);

  std::free(out);
}

void check_lakes(std::set<std::vector<int>> &ref, std::set<std::vector<int>> &out)
{
  ASSERT_EQ(ref.size(), out.size());

  for (auto &rbox : ref)
  {
    auto it = out.begin();
    while (it != out.end())
    {
      auto current = it++;
      auto obox = *current;
      int i;
      for (i = 0; i < 4; i++)
        if (obox[i] != rbox[i])
          break;
      if (i == 4)
        out.erase(obox);
    }
  }

  ASSERT_EQ(out.size(), 0);
}

TEST(Connectic_component, small_one_no_bounderies)
{
  int width = 5;
  int height = 5;
  int img[] = {
      0, 0, 0, 0, 0,
      0, 1, 1, 1, 0,
      0, 1, 1, 1, 0,
      0, 1, 1, 1, 0,
      0, 0, 0, 0, 0};

  int minimum_pixel = 0;

  auto out = lakes(img, width, height, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 3, 3},
  };

  check_lakes(ref, out);
}

TEST(Connectic_component, small_one_no_bounderies_holl)
{
  int width = 5;
  int height = 5;
  int img[] = {
      0, 0, 0, 0, 0,
      0, 1, 1, 1, 0,
      0, 1, 0, 1, 0,
      0, 1, 1, 1, 0,
      0, 0, 0, 0, 0};

  int minimum_pixel = 0;

  auto out = lakes(img, width, height, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 3, 3},
  };

  check_lakes(ref, out);
}

TEST(Connectic_component, small_two_no_bounderies)
{
  int width = 5;
  int height = 5;
  int img[] = {
      0, 0, 0, 0, 0,
      0, 1, 0, 1, 0,
      0, 1, 0, 1, 0,
      0, 1, 0, 1, 0,
      0, 0, 0, 0, 0};

  int minimum_pixel = 0;

  auto out = lakes(img, width, height, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 1, 3},
      {3, 1, 1, 3},
  };

  check_lakes(ref, out);
}

TEST(Connectic_component, small_full)
{
  int width = 5;
  int height = 5;
  int img[] = {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1};

  int minimum_pixel = 0;

  auto out = lakes(img, width, height, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {0, 0, 5, 5},
  };

  check_lakes(ref, out);
}

TEST(Connectic_component, one_complexe)
{
  int width = 9;
  int height = 5;
  int img[] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 1, 1, 1, 0, 1, 0,
      0, 1, 0, 1, 0, 1, 0, 1, 0,
      0, 1, 1, 1, 0, 1, 1, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0};

  int minimum_pixel = 0;

  auto out = lakes(img, width, height, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 7, 3},
  };

  check_lakes(ref, out);
}