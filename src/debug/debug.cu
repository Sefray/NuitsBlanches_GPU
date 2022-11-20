#include "debug.hh"

#include <err.h>

void save_img_gpu(int* d_img, int width, int height, std::string filename, int factor)
{
  int* h_img = new int[width * height];
  int  rc  = cudaMemcpy(h_img, d_img, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
  if (rc)
    errx(1, "Fail buffer copy to device");
  save_img(h_img, width, height, filename, factor);
  delete h_img;
}