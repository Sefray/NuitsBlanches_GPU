#pragma once

#include <nlohmann/json.hpp>
#include <png++/png.hpp>

#include <string>
#include <vector>

#include "pipeline.cuh"
#include "pipeline.hh"

using json = nlohmann::json;

json main_cpu(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
              int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
              int minimum_pixel);

json main_gpu_1(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel);

json main_gpu_2(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel);

json main_gpu_3(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel);

json main_gpu_4(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel);

json main_gpu_5(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel);

json main_gpu_6(std::vector<std::tuple<std::string, unsigned char*>> images, unsigned char* ref, int width, int height,
                int kernel_size, int kernel_size_opening, int kernel_size_closing, int binary_threshold,
                int minimum_pixel);