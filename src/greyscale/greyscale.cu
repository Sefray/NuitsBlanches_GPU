#include "pipeline.hh"

namespace gpu
{
  __global__ void gpu_greyscale(unsigned char* d_in, int* d_out, int width, int height)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    int image_p = p * 3;

    float b = d_in[image_p];
    float g = d_in[image_p + 1];
    float r = d_in[image_p + 2];

    d_out[p] = static_cast<int>(0.2126 * r + 0.7152 * g + 0.0722 * b);
  }

  namespace one
  {
    int* greyscale(unsigned char* d_in, int width, int height)
    {
      int* d_out = my_cuda_malloc(sizeof(int) * width * height);

      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_greyscale<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");

      cudaFree(d_in);

      return d_out;
    }
  } // namespace one

  namespace one::two
  {
    void greyscale(unsigned char* d_in, int *d_out, int width, int height)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_greyscale<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }
  } // namespace one::two
} // namespace gpu