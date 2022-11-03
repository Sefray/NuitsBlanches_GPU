#include "pipeline.hh"

#include <cstdlib>

namespace gpu
{
  float* init_gaussian_kernel(int kernel_size, float sigma = 1.0f)
  {
    float* h_ret = cpu::init_gaussian_kernel(kernel_size, sigma);

    float* d_ret;
    cudaMalloc(&d_ret, sizeof(float) * kernel_size * kernel_size);
    cudaMemcpy(d_ret, h_ret, sizeof(float) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    std::free(h_ret);

    return d_ret;
  }

  namespace one
  {
    __global__ void gpu_smoothing(int* d_in, int* d_out, float* kernel, int width, int height, int kernel_size, int ks2)
    {
      int p = blockDim.x * blockIdx.x + threadIdx.x;

      int x = p % width;
      int y = p / width;

      if (x >= width - ks2 || y >= height - ks2 || x < ks2 || y < ks2)
        return;

      float v = 0;
      for (int i = -ks2; i <= ks2; i++)
        for (int j = -ks2; j <= ks2; j++)
        {
          int cx = x + i;
          int cy = y + j;
          int ci = i + ks2;
          int cj = j + ks2;

          v += kernel[cj * kernel_size + ci] * d_in[cy * width + cx];
        }

      d_out[x + y * width] = static_cast<int>(v);
    }

    int* smoothing(int* d_in, int width, int height, int kernel_size)
    {
      assert(kernel_size % 2 == 1);

      float* kernel = init_gaussian_kernel(kernel_size);
      int    ks2    = kernel_size / 2;

      int* d_out = my_cuda_calloc(sizeof(int) * width * height);

      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_smoothing<<<dimGrid, dimBlock>>>(d_in, d_out, kernel, width, height, kernel_size, ks2);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");

      cudaFree(kernel);
      cudaFree(d_in);

      return d_out;
    }
  } // namespace one

  namespace two
  {
    __global__ void gpu_smoothing(int* d_in, int* d_out, float* kernel, int width, int height, int kernel_size, int ks2)
    {
      int p = blockDim.x * blockIdx.x + threadIdx.x;

      int x = p % width;
      int y = p / width;

      if (x >= width - ks2 || y >= height - ks2 || x < ks2 || y < ks2)
        return;

      float v = 0;
      for (int i = -ks2; i <= ks2; i++)
        for (int j = -ks2; j <= ks2; j++)
        {
          int cx = x + i;
          int cy = y + j;
          int ci = i + ks2;
          int cj = j + ks2;

          v += kernel[cj * kernel_size + ci] * d_in[cy * width + cx];
        }

      d_out[x + y * width] = static_cast<int>(v);
    }

    void smoothing(int* d_in, int* d_out, int width, int height, int kernel_size)
    {
      assert(kernel_size % 2 == 1);

      float* kernel = init_gaussian_kernel(kernel_size);
      int    ks2    = kernel_size / 2;

      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_smoothing<<<dimGrid, dimBlock>>>(d_in, d_out, kernel, width, height, kernel_size, ks2);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");

      cudaFree(kernel);
    }
  } // namespace two
} // namespace gpu