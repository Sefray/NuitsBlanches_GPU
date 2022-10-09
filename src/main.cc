#include <iostream>
#include <png++/png.hpp>
#include <cstdlib>

int *greyscale(png::pixel_buffer<png::rgb_pixel> image, int width, int height)
{
    int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));
    for (uint_fast32_t x = 0; x < width; x++)
        for (uint_fast32_t y = 0; y < height; y++)
            ret[x + y * width] = static_cast<int>(image[y][x].red + image[y][x].green + image[y][x].blue) / 3;
    return ret;
}

int *smoothing(int *image)
{

}

void pipeline(int * ref, png::image<png::rgb_pixel> modified, int width, int height)
{
    (void)ref;
    (void)modified;
    // 0.Pixel buffer

    // 1.Greyscale

    // 2.Smooth (gaussian filter)

    // 3.Difference
    // 4.Closing/opening with disk or rectangle
    // 5.1.Thresh image
    // 5.2.Lakes
    // 6.Output Json
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
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./main image_ref.png image1.png [imageN.png]" << std::endl;
        return 1;
    }

    png::image<png::rgb_pixel> ref(argv[1]);

    auto buf = ref.get_pixbuf();
    std::cout << buf[0][0].blue << std::endl;

    for (int i = 2; i < argc; i++)
    {
        png::image<png::rgb_pixel> modified(argv[i]);
        pipeline(ref, modified);
    }
}
