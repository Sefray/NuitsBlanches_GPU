#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <png++/png.hpp>

#include "mains.hh"
#include "pipeline.hh"

using json = nlohmann::json;

std::vector<std::string> get_files(int argc, char* argv[], bool is_path)
{
  std::vector<std::string> files;
  if (is_path)
  {
    if (!std::filesystem::is_directory(argv[1]))
    {
      std::cerr << "Error: " << argv[1] << " is not a directory" << std::endl;
      exit(1);
    }

    for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
      files.push_back(entry.path());
  }
  else
  {
    for (int i = 1; i < argc; ++i)
      files.push_back(argv[i]);
  }

  return files;
}

int main(int argc, char* argv[])
{
  cv::CommandLineParser parser(
      argc, argv,
      "{mode                     |0|0:CPU 1:GPU1 2:GPU2 3:GPU3 4:GPU4 5:GPU5 6:GPU6}"

      "{kernel_size              |5|Size of the kernel for the gaussian blur}"
      "{kernel_size_opening      |101|Should be odd}"
      "{kernel_size_closing      |41|Should be odd}"
      "{binary_threshold         |12|Minimum value for a pixel to be considered as a binary pixel}"
      "{minimum_pixel_percentage |1.0|Percentage of the space occupied by the object to be considered as a detection}"

      "{folder                   |false|Is the path a folder}"

      "{help                 h|false|show help message}");

  parser.about("Usage: ./main [OPTIONS] -- REFENCE_IMAGE_PATH ([IMAGE_PATH]*|DIRECTORY_PATH)");

  int i = 1;
  while (argv[i] && strcmp(argv[i++], "--"))
    continue;

  if (parser.get<bool>("help") || i + 2 > argc)
  {
    parser.printMessage();
    return 1;
  }

  argv += i;
  argc -= i;

  std::vector<std::string> vargv = get_files(argc, argv, parser.get<bool>("folder"));

  int mode = parser.get<int>("mode");

  int   kernel_size              = parser.get<int>("kernel_size");
  int   kernel_size_opening      = parser.get<int>("kernel_size_opening");
  int   kernel_size_closing      = parser.get<int>("kernel_size_closing");
  int   binary_threshold         = parser.get<int>("binary_threshold");
  float minimum_pixel_percentage = parser.get<float>("minimum_pixel_percentage");

  auto ref = cv::imread(*argv, cv::IMREAD_COLOR);

  int width  = ref.cols;
  int height = ref.rows;

  int minimum_pixel = width * height * minimum_pixel_percentage / 100;

  std::vector<std::function<decltype(main_cpu)>> main_func = {main_cpu,   main_gpu_1, main_gpu_2, main_gpu_3,
                                                              main_gpu_4, main_gpu_5, main_gpu_6};

  std::vector<std::tuple<std::string, unsigned char*>> images;
  for (auto& file : vargv)
  {
    auto           img  = cv::imread(file, cv::IMREAD_COLOR);
    unsigned char* data = new unsigned char[width * height * 3];
    memcpy(data, img.data, width * height * 3);
    images.emplace_back(file, data);
  }

  json ret = main_func[mode](images, ref.data, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                             binary_threshold, minimum_pixel);

  for (auto& img : images)
  {
    delete[] std::get<1>(img);
  }

  std::cout << ret.dump(2) << std::endl;
}
