#include <gtest/gtest.h>

#include <cstdlib>

#include "debug/debug.hh"
#include "pipeline.hh"

using namespace cpu;

int* malloc_img(int* ori, int width, int height)
{
  int* img = static_cast<int*>(std::malloc(sizeof(int) * width * height));
  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      img[y * width + x] = ori[y * width + x];
  return img;
}

void check_img(int* out, int* ref, int width, int height)
{
  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      EXPECT_EQ(out[y * width + x], ref[y * width + x]);
}

void sum_as_one(float* kernel, int kernel_size)
{
  float f = 0;
  for (int x = 0; x < kernel_size; x++)
    for (int y = 0; y < kernel_size; y++)
      f += kernel[x + y * kernel_size];

  EXPECT_NEAR(f, 1, 0.00001f);
}

TEST(Smoothing_kernel, gaussian_5)
{
  int  kernel_size = 5;
  auto kernel      = init_gaussian_kernel(kernel_size);
  sum_as_one(kernel, kernel_size);
  std::free(kernel);
}

TEST(Smoothing_kernel, gaussian_11)
{
  int  kernel_size = 11;
  auto kernel      = init_gaussian_kernel(kernel_size);
  sum_as_one(kernel, kernel_size);
  std::free(kernel);
}

TEST(Smoothing, smoothing_one)
{
  int kernel_size = 5;

  int width  = 5;
  int height = 5;
  int ori[]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto out = smoothing(malloc_img(ori, width, height), width, height, kernel_size);

  int ref[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  check_img(ref, out, width, height);

  std::free(out);
}

TEST(Smoothing, smoothing_ones)
{
  int kernel_size = 5;

  int width  = 5;
  int height = 5;
  int ori[]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto out = smoothing(malloc_img(ori, width, height), width, height, kernel_size);

  int ref[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  check_img(ref, out, width, height);

  std::free(out);
}

TEST(Create_mask, create_mask)
{
  int size_mask = 3;

  int ref[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto out = create_mask(size_mask);

  check_img(ref, out, size_mask, size_mask);

  std::free(out);
}


TEST(Closing_Opening, erosion_single_square)
{
  int  size_mask = 3;
  auto mask      = create_mask(size_mask);

  int img_width  = 5;
  int img_height = 5;
  int ori[]      = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  auto out = erosion(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, erosion_two)
{
  int  size_mask = 3;
  auto mask      = create_mask(size_mask);

  int img_width  = 5;
  int img_height = 5;
  int ori[]      = {1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1};

  auto out = erosion(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, dilatation_single_square)
{
  int  size_mask = 3;
  auto mask      = create_mask(size_mask);

  int img_width  = 5;
  int img_height = 5;
  int ori[]      = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto out = dilatation(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, dilatation_two_square)
{
  int  size_mask = 3;
  auto mask      = create_mask(size_mask);

  int img_width  = 5;
  int img_height = 5;
  int ori[]      = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};

  auto out = dilatation(malloc_img(ori, img_width, img_height), img_width, img_height, mask, size_mask);

  int ref[] = {1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1};

  check_img(ref, out, img_width, img_height);

  std::free(mask);
  std::free(out);
}

TEST(Closing_Opening, opening5_closing3)
{
  int kernel_size_opening = 5;
  int kernel_size_closing = 3;

  int img_width  = 10;
  int img_height = 10;
  int ori[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
               1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
               1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto out = closing_opening(malloc_img(ori, img_width, img_height), img_width, img_height, kernel_size_opening,
                             kernel_size_closing);

  int ref[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  check_img(ref, out, img_width, img_height);

  std::free(out);
}

TEST(Closing_Opening, opening3_closing3_hard)
{
  int kernel_size_opening = 3;
  int kernel_size_closing = 3;

  int img_width  = 20;
  int img_height = 20;
  int ori[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
               1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  auto out = closing_opening(malloc_img(ori, img_width, img_height), img_width, img_height, kernel_size_opening,
                             kernel_size_closing);

  int ref[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
               1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  check_img(ref, out, img_width, img_height);

  std::free(out);
}

void check_connected_components(std::set<std::vector<int>>& ref, std::set<std::vector<int>>& out)
{
  ASSERT_EQ(ref.size(), out.size());

  for (auto& rbox : ref)
  {
    auto it = out.begin();
    while (it != out.end())
    {
      auto current = it++;
      auto obox    = *current;
      int  i;
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
  int width  = 5;
  int height = 5;
  int img[]  = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  int minimum_pixel       = 0;
  int high_pick_threshold = 1;

  auto out = get_connected_components(img, img, width, height, high_pick_threshold, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 3, 3},
  };

  check_connected_components(ref, out);
}

TEST(Connectic_component, small_one_no_bounderies_holl)
{
  int width  = 5;
  int height = 5;
  int img[]  = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0};

  int minimum_pixel       = 0;
  int high_pick_threshold = 1;

  auto out = get_connected_components(img, img, width, height, high_pick_threshold, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 3, 3},
  };

  check_connected_components(ref, out);
}

TEST(Connectic_component, small_two_no_bounderies)
{
  int width  = 5;
  int height = 5;
  int img[]  = {0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};

  int minimum_pixel       = 0;
  int high_pick_threshold = 1;

  auto out = get_connected_components(img, img, width, height, high_pick_threshold, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 1, 3},
      {3, 1, 1, 3},
  };

  check_connected_components(ref, out);
}

TEST(Connectic_component, small_full)
{
  int width  = 5;
  int height = 5;
  int img[]  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  int minimum_pixel       = 0;
  int high_pick_threshold = 1;

  auto out = get_connected_components(img, img, width, height, high_pick_threshold, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {0, 0, 5, 5},
  };

  check_connected_components(ref, out);
}

TEST(Connectic_component, one_complexe)
{
  int width  = 9;
  int height = 5;
  int img[]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int minimum_pixel       = 0;
  int high_pick_threshold = 1;

  auto out = get_connected_components(img, img, width, height, high_pick_threshold, minimum_pixel);

  std::set<std::vector<int>> ref = {
      {1, 1, 7, 3},
  };

  check_connected_components(ref, out);
}

TEST(Binary, small_cross)
{
  int threshold = 5;

  int width  = 5;
  int height = 5;
  int img[]  = {1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 7, 9, 8, 1, 1, 1, 8, 1, 0, 1, 1, 1, 2, 1};

  binary_image(img, width, height, threshold);

  int ref[] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};

  check_img(img, ref, width, height);
}
