#include <benchmark/benchmark.h>

#include <png++/png.hpp>
#include "src/pipeline_cpu.hh"

void BM_Detection_cpu(benchmark::State &st)
{
    png::image<png::rgb_pixel> ref("../data/01.png");
    png::image<png::rgb_pixel> modified("../data/02.png");
    auto pixbuf = modified.get_pixbuf();

    int width = ref.get_width();
    int height = ref.get_height();

    auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
    auto ref_smoothed = cpu::smoothing(ref_greyscale, width, height);

    for (auto _ : st)
        auto components = cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height);

    std::free(ref_smoothed);
}

BENCHMARK(BM_Detection_cpu)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();