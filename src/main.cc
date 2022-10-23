#include <iostream>
#include <png++/png.hpp>
#include <nlohmann/json.hpp>

#include "pipeline_cpu.hh"

using json = nlohmann::json;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./main image_ref.png image1.png [imageN.png]" << std::endl;
        return 1;
    }

    png::image<png::rgb_pixel> ref(argv[1]);

    int width = ref.get_width();
    int height = ref.get_height();

    auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
    auto ref_smoothed = cpu::smoothing(ref_greyscale, width, height);

    json ret;
    for (int i = 2; i < argc; i++)
    {
        png::image<png::rgb_pixel> modified(argv[i]);
        auto components = cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height);

        /*
        Json format example
        {
            "input-0001.jpg" : [
                [0, 0, 10, 10],
                [15, 15, 42, 42]
            ],
            "input-0002.jpg" : [],
            "input-0003.jpg" : [
                [0, 0, 10, 10],
                [15, 15, 42, 42],
                [51, 42, 69, 99]
            ]
        }
        */
        ret[argv[i]] = components;
    }
    std::free(ref_smoothed);

    std::cout << ret.dump(2) << std::endl;
}
