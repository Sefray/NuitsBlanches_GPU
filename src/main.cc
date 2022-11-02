#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core/utility.hpp>
#include <png++/png.hpp>

#include "pipeline.hh"

using json = nlohmann::json;


json main_cpu(int argc, char* argv[], png::image<png::rgb_pixel> ref, int width, int height, int kernel_size,
              int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
              int minimum_pixel)
{
  auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
  auto ref_smoothed  = cpu::smoothing(ref_greyscale, width, height, kernel_size);

  json ret;
  for (int img = 1; img < argc; img++)
  {
    png::image<png::rgb_pixel> modified(argv[img]);
    ret[argv[img]] = cpu::pipeline(ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                                   kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
  }

  std::free(ref_smoothed);

  return ret;
}

json main_gpu_1(int argc, char* argv[], png::image<png::rgb_pixel> ref, int width, int height, int kernel_size,
                int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
                int minimum_pixel)
{
  auto ref_greyscale = cpu::greyscale(ref.get_pixbuf(), width, height);
  auto ref_smoothed  = cpu::smoothing(ref_greyscale, width, height, kernel_size);

  auto d_ref_smoothed = gpu::malloc_and_copy(ref_smoothed, width, height);

  json ret;
  for (int img = 1; img < argc; img++)
  {
    png::image<png::rgb_pixel> modified(argv[img]);
    ret[argv[img]] =
        gpu::one::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                           kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
  }

  std::free(ref_smoothed);
  gpu::my_cuda_free(d_ref_smoothed);

  return ret;
}

json main_gpu_2(int argc, char* argv[], png::image<png::rgb_pixel> ref, int width, int height, int kernel_size,
                int kernel_size_opening, int kernel_size_closing, int binary_threshold, enum mode_cc mode_cc,
                int minimum_pixel)
{
  auto ref_greyscale  = cpu::greyscale(ref.get_pixbuf(), width, height);
  auto ref_smoothed   = cpu::smoothing(ref_greyscale, width, height, kernel_size);
  auto d_ref_smoothed = gpu::malloc_and_copy(ref_smoothed, width, height);

  json ret;
  for (int img = 1; img < argc; img++)
  {
    png::image<png::rgb_pixel> modified(argv[img]);
    ret[argv[img]] =
        gpu::one::pipeline(d_ref_smoothed, modified.get_pixbuf(), width, height, kernel_size, kernel_size_opening,
                           kernel_size_closing, binary_threshold, mode_cc, minimum_pixel);
  }

  std::free(ref_smoothed);
  gpu::my_cuda_free(d_ref_smoothed);

  return ret;
}

int main(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv,
                               "{mode                   |0|0:CPU 1:GPU}"

                               "{kernel_size |5|Should be odd}"

                               "{kernel_size_opening       |101|Should be odd}"
                               "{kernel_size_closing       |41|Should be odd}"

                               "{binary_threshold        |12|}"

                               "{minimum_pixel      |30| Angle in degree}"
                               "{mode_cc            |0|0:slide 1:union_find}"

                               "{help    h|false|show help message}");

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

  int mode = parser.get<int>("mode");

  int          kernel_size         = parser.get<int>("kernel_size");
  int          kernel_size_opening = parser.get<int>("kernel_size_opening");
  int          kernel_size_closing = parser.get<int>("kernel_size_closing");
  int          binary_threshold    = parser.get<int>("binary_threshold");
  enum mode_cc mode_cc             = static_cast<enum mode_cc>(parser.get<int>("mode_cc"));
  int          minimum_pixel       = parser.get<int>("minimum_pixel");

  png::image<png::rgb_pixel> ref(*argv);

  int width  = ref.get_width();
  int height = ref.get_height();

  std::vector<std::function<decltype(main_cpu)>> main_func = {main_cpu, main_gpu_1, main_gpu_2};

  json ret = main_func[mode](argc, argv, ref, width, height, kernel_size, kernel_size_opening, kernel_size_closing,
                             binary_threshold, mode_cc, minimum_pixel);

  std::cout << ret.dump(2) << std::endl;
}
