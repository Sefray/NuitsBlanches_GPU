#include "pipeline.hh"

namespace gpu
{
  int* create_mask(int kernel_size, enum mask_type type = mask_type::square)
  {
    int* h_ret = cpu::create_mask(kernel_size, type);

    int* d_ret;
    cudaMalloc(&d_ret, sizeof(int) * kernel_size * kernel_size);
    cudaMemcpy(d_ret, h_ret, sizeof(int) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    std::free(h_ret);

    return d_ret;
  }

  enum type_oc
  {
    EROSION,
    DILATATION,
  };

  __device__ int dilatation(int a, int b)
  {
    return a < b ? b : a;
  }

  __device__ int erosion(int a, int b)
  {
    return a < b ? a : b;
  }

  __global__ void gpu_kernel_func(int* d_in, int* d_out, int width, int height, int* kernel, int kernel_size, int ks2,
                                  enum type_oc type)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    auto f = type == EROSION ? erosion : dilatation;

    int v = kernel[ks2 * kernel_size + ks2] * d_in[y * width + x];

    for (int j = -ks2; j <= ks2; j++)
    {
      int cy = y + j;
      if (cy < 0 || cy >= height)
        continue;

      for (int i = -ks2; i <= ks2; i++)
      {
        int cx = x + i;
        if (cx < 0 || cx >= width)
          continue;

        int ci = i + ks2;
        int cj = j + ks2;

        v = f(kernel[cj * kernel_size + ci] * d_in[cy * width + cx], v);
      }
    }

    d_out[y * width + x] = v;
  }

  namespace one
  {
    int* kernel_func(int* d_in, int width, int height, int* kernel, int kernel_size, enum type_oc type)
    {
      int* d_out = my_cuda_calloc(sizeof(int) * width * height);

      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_kernel_func<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, kernel, kernel_size, kernel_size / 2, type);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");

      my_cuda_free(d_in);

      return d_out;
    }

    int* closing_opening(int* d_in, int width, int height, int kernel_size_opening, int kernel_size_closing)
    {
      // Closing
      auto mask = create_mask(kernel_size_closing);
      auto d_a  = kernel_func(d_in, width, height, mask, kernel_size_closing, DILATATION);
      auto d_b  = kernel_func(d_a, width, height, mask, kernel_size_closing, EROSION);
      cudaFree(mask);

      // Opening
      mask       = create_mask(kernel_size_opening);
      auto d_c   = kernel_func(d_b, width, height, mask, kernel_size_opening, EROSION);
      auto d_out = kernel_func(d_c, width, height, mask, kernel_size_opening, DILATATION);
      cudaFree(mask);

      return d_out;
    }
  } // namespace one

  namespace one::two
  {
    void kernel_func(int* d_in, int* d_out, int width, int height, int* kernel, int kernel_size, enum type_oc type)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_kernel_func<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, kernel, kernel_size, kernel_size / 2, type);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }

    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing)
    {
      // Closing
      auto mask = create_mask(kernel_size_closing);
      kernel_func(d_A, d_B, width, height, mask, kernel_size_closing, DILATATION);
      kernel_func(d_B, d_A, width, height, mask, kernel_size_closing, EROSION);
      cudaFree(mask);

      // Opening
      mask = create_mask(kernel_size_opening);
      kernel_func(d_A, d_B, width, height, mask, kernel_size_opening, EROSION);
      kernel_func(d_B, d_A, width, height, mask, kernel_size_opening, DILATATION);
      cudaFree(mask);
    }
  } // namespace one::two

  namespace one::two::three
  {
    __global__ void gpu_kernel_func(int* d_in, int* d_out, int width, int height, int* kernel, int kernel_size, int mx,
                                    int my, enum type_oc type)
    {
      int p = blockDim.x * blockIdx.x + threadIdx.x;

      int x = p % width;
      int y = p / width;

      if (x >= width || y >= height)
        return;

      int  ks2 = kernel_size / 2;
      auto f   = type == EROSION ? erosion : dilatation;

      int v = kernel[ks2 * kernel_size + ks2] * d_in[y * width + x];

      for (int j = -ks2 * my; j <= ks2 * my; j++)
      {
        int cy = y + j;
        if (cy < 0 || cy >= height)
          continue;

        for (int i = -ks2 * mx; i <= ks2 * mx; i++)
        {
          int cx = x + i;
          if (cx < 0 || cx >= width)
            continue;

          int ci = i + ks2;
          int cj = j + ks2;

          v = f(kernel[cj * kernel_size + ci] * d_in[cy * width + cx], v);
        }
      }

      d_out[y * width + x] = v;
    }

    void kernel_func(int* d_in_out, int* d_tmp, int width, int height, int* kernel, int kernel_size, enum type_oc type)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_kernel_func<<<dimGrid, dimBlock>>>(d_in_out, d_tmp, width, height, kernel, kernel_size, 1, 0, type);
      gpu_kernel_func<<<dimGrid, dimBlock>>>(d_tmp, d_in_out, width, height, kernel, kernel_size, 0, 1, type);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }

    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing)
    {
      // Closing
      auto mask = create_mask(kernel_size_closing);
      kernel_func(d_A, d_B, width, height, mask, kernel_size_closing, DILATATION);
      kernel_func(d_A, d_B, width, height, mask, kernel_size_closing, EROSION);
      cudaFree(mask);

      // Opening
      mask = create_mask(kernel_size_opening);
      kernel_func(d_A, d_B, width, height, mask, kernel_size_opening, EROSION);
      kernel_func(d_A, d_B, width, height, mask, kernel_size_opening, DILATATION);
      cudaFree(mask);
    }
  } // namespace one::two::three
} // namespace gpu