#include <iostream>
#include <png++/png.hpp>



void pipeline(png::image<png::rgb_pixel> ref, png::image<png::rgb_pixel> modified)
{
    (void) ref;
    (void) modified;

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
