#include <benchmark/benchmark.h>

#include "pipeline.hh"
#include <png++/png.hpp>

void BM_Detection_one(benchmark::State& st, std::string ref_filename, std::string modified_filename, enum mode mode,
                      enum mode_cc mode_cc)
{
  int kernel_size         = 5;
  int kernel_size_opening = 101;
  int kernel_size_closing = 41;
  int binary_threshold    = 10;
  int minimum_pixel       = 30;

  for (auto _ : st)
  {
    png::image<png::rgb_pixel> ref(ref_filename);

    int width  = ref.get_width();
    int height = ref.get_height();

    auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
    auto ref_smoothed  = cpu::smoothing(ref_greyscale, width, height, kernel_size);

    int* d_ref_smoothed = NULL;
    if (mode != CPU)
      d_ref_smoothed = gpu::malloc_and_copy(ref_smoothed, width, height);

    png::image<png::rgb_pixel> modified(modified_filename);
    auto                       pixbuf = modified.get_pixbuf();
    switch (mode)
    {
    case CPU:
      cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                    kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
      break;
    case GPU_1:
      gpu::one::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                         kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
      break;
    case GPU_2:
      gpu::two::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                         kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
      break;
    }

    if (mode != CPU)
      gpu::my_cuda_free(d_ref_smoothed);

    std::free(ref_smoothed);
  }
}

BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_cpu_slide, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), CPU, slide);
BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_cpu_union_find, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), CPU, union_find);

BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_gpu_first, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), GPU_1, union_find);
BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_gpu_second, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), GPU_2, union_find);

BENCHMARK_MAIN();