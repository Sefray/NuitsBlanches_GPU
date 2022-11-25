#include "pipeline.cuh"

namespace gpu
{
  __global__ void gpu_difference(int* d_ref_in, int* d_in, int* d_out, int width, int height)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    int c    = x + y * width;
    d_out[c] = std::abs(d_in[c] - d_ref_in[c]);
  }

  namespace one
  {
    int* compute_difference(int* d_ref_in, int* d_in, int width, int height)
    {
      int* d_out = my_cuda_malloc(sizeof(int) * width * height);

      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_difference<<<dimGrid, dimBlock>>>(d_ref_in, d_in, d_out, width, height);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");

      cudaFree(d_in);

      return d_out;
    }
  } // namespace one

  namespace one::two
  {
    void compute_difference(int* d_ref_in, int* d_in, int* d_out, int width, int height)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_difference<<<dimGrid, dimBlock>>>(d_ref_in, d_in, d_out, width, height);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }
  } // namespace one::two
} // namespace gpu