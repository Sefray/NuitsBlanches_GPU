#include <benchmark/benchmark.h>

#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "mains.hh"
#include "pipeline.hh"


namespace fs = std::filesystem;

void bench_func(benchmark::State& state, std::string const& ref_filename, std::vector<std::string> const& vargv,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                enum mode_cc mode_cc, enum mode mode, int minimum_pixel)
{
  for (auto _ : state)
  {
    auto ref = cv::imread(ref_filename, cv::IMREAD_COLOR);

    int width  = ref.cols;
    int height = ref.rows;

    std::vector<std::function<decltype(main_cpu)>> main_func = {main_cpu, main_gpu_1, main_gpu_2, main_gpu_3,
                                                                main_gpu_4};

    main_func[mode](vargv, ref.data, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                    binary_threshold, mode_cc, minimum_pixel);
  }
}

// Register the function as a benchmark : SCIA Premium

void Detection_sp(benchmark::State& state, std::string const& ref_filename, std::vector<std::string> const& vargv,
                  enum mode mode, enum mode_cc mode_cc)
{
  int kernel_size         = 5;
  int kernel_size_opening = 51;
  int kernel_size_closing = 41;
  int binary_threshold    = 50;
  int minimum_pixel       = 30;

  bench_func(state, ref_filename, vargv, kernel_size, kernel_size_opening, kernel_size_closing, binary_threshold,
             mode_cc, mode, minimum_pixel);
}

void BM_Detection_file_sp(benchmark::State& state, enum mode mode, enum mode_cc mode_cc)
{
  std::string ref_filename      = std::string("../data/scia_premium_91/sp-0001.png");
  std::string modified_filename = std::string("../data/scia_premium_91/sp-0055.png");

  std::vector<std::string> vargv = {modified_filename};

  Detection_sp(state, ref_filename, vargv, mode, mode_cc);
}

void BM_Detection_folder_sp(benchmark::State& state, enum mode mode, enum mode_cc mode_cc)
{
  std::string ref_filename = std::string("../data/scia_premium_91/sp-0001.png");
  std::string folder       = std::string("../data/scia_premium_91/");

  std::vector<std::string> vargv;
  for (const auto& entry : fs::directory_iterator(folder))
    vargv.push_back(entry.path());

  Detection_sp(state, ref_filename, vargv, mode, mode_cc);
}

// Register the function as a benchmark : Nuits blanches

void Detection_nb(benchmark::State& state, std::string const& ref_filename, std::vector<std::string> const& vargv,
                  enum mode mode, enum mode_cc mode_cc)
{
  int kernel_size         = 5;
  int kernel_size_opening = 51;
  int kernel_size_closing = 41;
  int binary_threshold    = 20;
  int minimum_pixel       = 30;

  bench_func(state, ref_filename, vargv, kernel_size, kernel_size_opening, kernel_size_closing, binary_threshold,
             mode_cc, mode, minimum_pixel);
}

void BM_Detection_file_nb(benchmark::State& state, enum mode mode, enum mode_cc mode_cc)
{
  std::string ref_filename      = std::string("../data/nuit_blanches_81/nb-01.png");
  std::string modified_filename = std::string("../data/nuit_blanches_81/nb-65.png");

  std::vector<std::string> vargv = {modified_filename};

  Detection_nb(state, ref_filename, vargv, mode, mode_cc);
}

void BM_Detection_folder_nb(benchmark::State& state, enum mode mode, enum mode_cc mode_cc)
{
  std::string ref_filename = std::string("../data/nuit_blanches_81/nb-01.png");
  std::string folder       = std::string("../data/nuit_blanches_81/");

  std::vector<std::string> vargv;
  for (const auto& entry : fs::directory_iterator(folder))
    vargv.push_back(entry.path());

  Detection_nb(state, ref_filename, vargv, mode, mode_cc);
}

// SCIA Premium (one objet + framesize = (340, 640))

BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_cpu_slide, CPU, slide);
BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_cpu_union_find, CPU, union_find);
BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_one, GPU_1, slide);
BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_two, GPU_2, slide);
BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_three, GPU_3, slide);
BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_four, GPU_4, slide);

BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_one, GPU_1, slide);
BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_two, GPU_2, slide);
BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_three, GPU_3, slide);
BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_four, GPU_4, slide);

// Nuit Blanches (multiple objects + framesize = (1080, 1920))
BENCHMARK_CAPTURE(BM_Detection_file_nb, scia_premium_cpu_slide, CPU, slide);
BENCHMARK_CAPTURE(BM_Detection_file_nb, scia_premium_cpu_union_find, CPU, union_find);
BENCHMARK_CAPTURE(BM_Detection_file_nb, scia_premium_gpu_one, GPU_1, slide);
BENCHMARK_CAPTURE(BM_Detection_file_nb, scia_premium_gpu_two, GPU_2, slide);
BENCHMARK_CAPTURE(BM_Detection_file_nb, scia_premium_gpu_three, GPU_3, slide);
BENCHMARK_CAPTURE(BM_Detection_file_nb, scia_premium_gpu_four, GPU_4, slide);

BENCHMARK_CAPTURE(BM_Detection_folder_nb, scia_premium_gpu_one, GPU_1, slide);
BENCHMARK_CAPTURE(BM_Detection_folder_nb, scia_premium_gpu_two, GPU_2, slide);
BENCHMARK_CAPTURE(BM_Detection_folder_nb, scia_premium_gpu_three, GPU_3, slide);
BENCHMARK_CAPTURE(BM_Detection_folder_nb, scia_premium_gpu_four, GPU_4, slide);


BENCHMARK_MAIN();