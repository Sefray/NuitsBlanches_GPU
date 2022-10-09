#include <iostream>
#include <png++/png.hpp>

void pipeline(png::image<png::rgb_pixel> ref, png::image<png::rgb_pixel> modified)
{
    (void) ref;
    (void) modified;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./main image_ref.png image1.png [imageN.png]" << std::endl;
        return 1;
    }

    png::image<png::rgb_pixel> ref(argv[1]);

    for (int i = 2; i < argc; i++)
    {
        png::image<png::rgb_pixel> modified(argv[i]);
        pipeline(ref, modified);
    }
}
