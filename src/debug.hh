#pragma once

#include <string>

void display_img_stdout(int *img, int width, int height);
void save_img(int *img, int width, int height, std::string filename, int factor = 1);