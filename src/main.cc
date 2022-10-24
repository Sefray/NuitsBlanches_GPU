#include <iostream>
#include <png++/png.hpp>
#include <nlohmann/json.hpp>

#include "pipeline.hh"
#include "test.cuh"

using json = nlohmann::json;

enum mode
{
    CPU,
    GPU
};

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./main image_ref.png image1.png [imageN.png]" << std::endl;
        return 1;
    }

    enum mode mode = CPU;

    png::image<png::rgb_pixel> ref(argv[1]);

    int width = ref.get_width();
    int height = ref.get_height();

    auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
    auto ref_smoothed = cpu::smoothing(ref_greyscale, width, height);

    int *d_ref_smoothed = NULL;
    if (mode == GPU)
        d_ref_smoothed = gpu::malloc_and_copy(ref_smoothed, width, height);

    json ret;
    for (int i = 2; i < argc; i++)
    {
        png::image<png::rgb_pixel> modified(argv[i]);
        switch (mode)
        {
        case CPU:
            ret[argv[i]] = cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height);
            break;
        case GPU:
            ret[argv[i]] = gpu::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height);
            break;
        }
    }

    std::free(ref_smoothed);
    if (mode == GPU)
        gpu::my_cuda_free(d_ref_smoothed);

    std::cout << ret.dump(2) << std::endl;
}
