#include <benchmark/benchmark.h>

#include "pipeline.hh"
#include <png++/png.hpp>

void BM_Detection_cpu(benchmark::State& st, std::string ref_filename, std::string modified_filename,
                      enum mode_cc mode_cc)
{
  png::image<png::rgb_pixel> ref(ref_filename);
  png::image<png::rgb_pixel> modified(modified_filename);
  auto                       pixbuf = modified.get_pixbuf();

  int width  = ref.get_width();
  int height = ref.get_height();

  int kernel_size         = 5;
  int kernel_size_opening = 101;
  int kernel_size_closing = 41;
  int binary_threshold    = 10;
  int minimum_pixel       = 30;

  auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
  auto ref_smoothed  = cpu::smoothing(ref_greyscale, width, height, kernel_size);

  for (auto _ : st)
    auto components = cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height, kernel_size,
                                    kernel_size_opening, kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);

  std::free(ref_smoothed);
}

void BM_Detection_gpu_one(benchmark::State& st, std::string ref_filename, std::string modified_filename,
                          enum mode_cc mode_cc)
{
  png::image<png::rgb_pixel> ref(ref_filename);
  png::image<png::rgb_pixel> modified(modified_filename);
  auto                       pixbuf = modified.get_pixbuf();

  int width  = ref.get_width();
  int height = ref.get_height();

  int kernel_size         = 5;
  int kernel_size_opening = 101;
  int kernel_size_closing = 41;
  int binary_threshold    = 20;
  int minimum_pixel       = 30;

  auto ref_greyscale  = cpu::greyscale(ref.get_pixbuf(), width, height);
  auto ref_smoothed   = cpu::smoothing(ref_greyscale, width, height, kernel_size);
  auto d_ref_smoothed = gpu::malloc_and_copy(ref_smoothed, width, height);

  for (auto _ : st)
    auto components =
        gpu::one::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                           kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);

  std::free(ref_smoothed);
  gpu::my_cuda_free(d_ref_smoothed);
}

void BM_Detection_gpu_two(benchmark::State& st, std::string ref_filename, std::string modified_filename,
                          enum mode_cc mode_cc)
{
  png::image<png::rgb_pixel> ref(ref_filename);
  png::image<png::rgb_pixel> modified(modified_filename);
  auto                       pixbuf = modified.get_pixbuf();

  int width  = ref.get_width();
  int height = ref.get_height();

  int kernel_size         = 5;
  int kernel_size_opening = 101;
  int kernel_size_closing = 41;
  int binary_threshold    = 20;
  int minimum_pixel       = 30;

  auto ref_greyscale  = cpu::greyscale(ref.get_pixbuf(), width, height);
  auto ref_smoothed   = cpu::smoothing(ref_greyscale, width, height, kernel_size);
  auto d_ref_smoothed = gpu::malloc_and_copy(ref_smoothed, width, height);

  for (auto _ : st)
    auto components =
        gpu::two::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                           kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);

  std::free(ref_smoothed);
  gpu::my_cuda_free(d_ref_smoothed);
}


BENCHMARK_CAPTURE(BM_Detection_cpu, scia_premium_slide, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), slide);
BENCHMARK_CAPTURE(BM_Detection_cpu, scia_premium_union_find, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), union_find);
BENCHMARK_CAPTURE(BM_Detection_gpu_one, scia_premium_first, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), union_find);
BENCHMARK_CAPTURE(BM_Detection_gpu_two, scia_premium_first, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), union_find);

BENCHMARK_MAIN();