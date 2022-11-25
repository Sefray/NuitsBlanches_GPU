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
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode mode,
                int minimum_pixel)
{
  std::vector<std::tuple<std::string, unsigned char*>> images;

  auto ref = cv::imread(ref_filename, cv::IMREAD_COLOR);

  int width  = ref.cols;
  int height = ref.rows;

  for (auto& file : vargv)
  {
    auto           img  = cv::imread(file, cv::IMREAD_COLOR);
    unsigned char* data = new unsigned char[width * height * 3];
    memcpy(data, img.data, width * height * 3);
    images.emplace_back(file, data);
  }

  std::vector<std::function<decltype(main_cpu)>> main_func = {main_cpu,   main_gpu_1, main_gpu_2, main_gpu_3,
                                                              main_gpu_4, main_gpu_5, main_gpu_6};

  for (auto _ : state)
  {
    main_func[mode](images, ref.data, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                    binary_threshold, minimum_pixel);
  }

  for (auto& img : images)
  {
    delete[] std::get<1>(img);
  }

  if (vargv.size() > 1)
    state.SetItemsProcessed(state.iterations() * vargv.size());
  else
    state.SetItemsProcessed(state.iterations());
}

// Register the function as a benchmark : SCIA Premium
void Detection_sp(benchmark::State& state, std::string const& ref_filename, std::vector<std::string> const& vargv,
                  enum mode mode)
{
  int kernel_size         = 5;
  int kernel_size_opening = 51;
  int kernel_size_closing = 41;
  int binary_threshold    = 50;
  int minimum_pixel       = 30;

  bench_func(state, ref_filename, vargv, kernel_size, kernel_size_opening, kernel_size_closing, binary_threshold, mode,
             minimum_pixel);
}

void BM_Detection_file_sp(benchmark::State& state, enum mode mode)
{
  std::string ref_filename      = std::string("../data/scia_premium_91/sp-0001.png");
  std::string modified_filename = std::string("../data/scia_premium_91/sp-0055.png");

  std::vector<std::string> vargv = {modified_filename};

  Detection_sp(state, ref_filename, vargv, mode);
}

void BM_Detection_folder_sp(benchmark::State& state, enum mode mode)
{
  std::string ref_filename = std::string("../data/scia_premium_91/sp-0001.png");
  std::string folder       = std::string("../data/scia_premium_91/");

  std::vector<std::string> vargv;
  for (const auto& entry : fs::directory_iterator(folder))
    vargv.push_back(entry.path());

  Detection_sp(state, ref_filename, vargv, mode);
}

// Register the function as a benchmark : Nuits blanches
void Detection_nb(benchmark::State& state, std::string const& ref_filename, std::vector<std::string> const& vargv,
                  enum mode mode)
{
  int kernel_size         = 5;
  int kernel_size_opening = 51;
  int kernel_size_closing = 41;
  int binary_threshold    = 20;
  int minimum_pixel       = 30;

  bench_func(state, ref_filename, vargv, kernel_size, kernel_size_opening, kernel_size_closing, binary_threshold, mode,
             minimum_pixel);
}

void BM_Detection_file_nb(benchmark::State& state, enum mode mode)
{
  std::string ref_filename      = std::string("../data/nuits_blanches_81/nb-01.png");
  std::string modified_filename = std::string("../data/nuits_blanches_81/nb-65.png");

  std::vector<std::string> vargv = {modified_filename};

  Detection_nb(state, ref_filename, vargv, mode);
}

void BM_Detection_folder_nb(benchmark::State& state, enum mode mode)
{
  std::string ref_filename = std::string("../data/nuits_blanches_81/nb-01.png");
  std::string folder       = std::string("../data/nuits_blanches_81/");

  std::vector<std::string> vargv;
  for (const auto& entry : fs::directory_iterator(folder))
    vargv.push_back(entry.path());

  Detection_nb(state, ref_filename, vargv, mode);
}

// SCIA Premium(one objet + framesize = (340, 640))
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_cpu, CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_one, GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_two, GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_three, GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_four, GPU_4)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_five, GPU_5)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_sp, scia_premium_gpu_six, GPU_6)->Unit(benchmark::kMillisecond)->UseRealTime();

// BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_one, GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_two, GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_three, GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_four, GPU_4)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_five, GPU_5)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_sp, scia_premium_gpu_six, GPU_6)->Unit(benchmark::kMillisecond)->UseRealTime();

// Nuit Blanches(multiple objects + framesize = (1080, 1920))
BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_cpu, CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_gpu_one, GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_gpu_two, GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_gpu_three, GPU_3)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_gpu_four, GPU_4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_gpu_five, GPU_5)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_file_nb, nuits_blanches_gpu_six, GPU_6)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_CAPTURE(BM_Detection_folder_nb, nuits_blanches_gpu_one, GPU_1)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_nb, nuits_blanches_gpu_two, GPU_2)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_nb, nuits_blanches_gpu_three, GPU_3)
//     ->Unit(benchmark::kMillisecond)
//     ->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_nb, nuits_blanches_gpu_four, GPU_4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_CAPTURE(BM_Detection_folder_nb, nuits_blanches_gpu_five, GPU_5)->Unit(benchmark::kMillisecond)->UseRealTime();
// BENCHMARK_CAPTURE(BM_Detection_folder_nb, nuits_blanches_gpu_six, GPU_6)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();