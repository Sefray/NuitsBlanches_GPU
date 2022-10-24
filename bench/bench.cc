#include <benchmark/benchmark.h>

#include <png++/png.hpp>
#include "src/pipeline.hh"

void BM_Detection_cpu(benchmark::State &st, std::string ref_filename, std::string modified_filename)
{
    png::image<png::rgb_pixel> ref(ref_filename);
    png::image<png::rgb_pixel> modified(modified_filename);
    auto pixbuf = modified.get_pixbuf();

    int width = ref.get_width();
    int height = ref.get_height();

    auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
    auto ref_smoothed = cpu::smoothing(ref_greyscale, width, height);

    for (auto _ : st)
        auto components = cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height);

    std::free(ref_smoothed);
}

BENCHMARK_CAPTURE(BM_Detection_cpu, nuit_blanche_02, std::string("../data/01.png"), std::string("../data/02.png"));
BENCHMARK_CAPTURE(BM_Detection_cpu, nuit_blanche_03, std::string("../data/01.png"), std::string("../data/03.png"));
BENCHMARK_CAPTURE(BM_Detection_cpu, nuit_blanche_04, std::string("../data/01.png"), std::string("../data/04.png"));

BENCHMARK_MAIN();