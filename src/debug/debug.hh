#pragma once

#include <string>

void display_img_stdout(int* img, int width, int height);
void display_imgf_stdout(float* img, int width, int height);
void save_img(int* img, int width, int height, std::string filename, int factor = 1);
void compute_and_display_histogramme(int* img, int width, int height);