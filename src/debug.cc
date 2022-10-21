#include <iostream>

void display_img(int *img, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
            printf("%03d ", img[y * width + x]);
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
}