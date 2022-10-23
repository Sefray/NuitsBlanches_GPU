#include <iostream>

#include <png++/png.hpp>

void display_img_stdout(int *img, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            printf("%03d ", img[y * width + x]);
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void save_img(int *img, int width, int height, std::string filename, int factor)
{
    png::image<png::gray_pixel> image(width, height);

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            image[y][x] = img[y * width + x] * factor;

    image.write(filename);
}