#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <png++/png.hpp>
#include <string>
#include <vector>

#include "mains.hh"
#include "pipeline.hh"

namespace fs = std::filesystem;

void BM_Detection_one(benchmark::State& st, std::string ref_filename, std::string modified_filename, enum mode mode,
                      enum mode_cc mode_cc)
{
  int kernel_size         = 5;
  int kernel_size_opening = 101;
  int kernel_size_closing = 41;
  int binary_threshold    = 10;
  int minimum_pixel       = 30;

  std::vector<std::string> vargv = {modified_filename};

  for (auto _ : st)
  {
    png::image<png::rgb_pixel> ref(ref_filename);

    int width  = ref.get_width();
    int height = ref.get_height();

    std::vector<std::function<decltype(main_cpu)>> main_func = {main_cpu, main_gpu_1, main_gpu_2, main_gpu_3};

    main_func[mode](vargv, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing, binary_threshold,
                    mode_cc, minimum_pixel);
  }
}

void BM_Detection_multiple(benchmark::State& st, std::string ref_filename, std::string folder, enum mode mode,
                           enum mode_cc mode_cc)
{
  int kernel_size         = 5;
  int kernel_size_opening = 101;
  int kernel_size_closing = 41;
  int binary_threshold    = 10;
  int minimum_pixel       = 30;

  std::vector<std::string> vargv;
  for (const auto& entry : fs::directory_iterator(folder))
    vargv.push_back(entry.path());

  for (auto _ : st)
  {
    png::image<png::rgb_pixel> ref(ref_filename);

    int width  = ref.get_width();
    int height = ref.get_height();

    std::vector<std::function<decltype(main_cpu)>> main_func = {main_cpu, main_gpu_1, main_gpu_2, main_gpu_3};

    main_func[mode](vargv, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing, binary_threshold,
                    mode_cc, minimum_pixel);
  }
}

BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_cpu_slide, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), CPU, slide);
BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_cpu_union_find, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), CPU, union_find);
BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_gpu_one, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), GPU_1, slide);
BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_gpu_two, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), GPU_2, slide);
BENCHMARK_CAPTURE(BM_Detection_one, scia_premium_gpu_three, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_0002.png"), GPU_3, slide);


BENCHMARK_CAPTURE(BM_Detection_multiple, scia_premium_cpu_slide, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_2/"), CPU, slide);
BENCHMARK_CAPTURE(BM_Detection_multiple, scia_premium_cpu_union_find, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_2/"), CPU, union_find);
BENCHMARK_CAPTURE(BM_Detection_multiple, scia_premium_gpu_one, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_2/"), GPU_1, slide);
BENCHMARK_CAPTURE(BM_Detection_multiple, scia_premium_gpu_two, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_2/"), GPU_2, slide);
BENCHMARK_CAPTURE(BM_Detection_multiple, scia_premium_gpu_three, std::string("../data/scia_premium_0001.png"),
                  std::string("../data/scia_premium_2/"), GPU_3, slide);

BENCHMARK_MAIN();