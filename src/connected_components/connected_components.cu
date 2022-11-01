#include "pipeline.hh"

#include <map>

namespace gpu
{
  struct Box
  {
    int xmin;
    int ymin;
    int xmax;
    int ymax;

    int size;
  };

  __global__ void gpu_init_label(int* d_in_out, int width, int height)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    d_in_out[p] *= (p + 2);
  }

  void init_label(int* d_in_out, int width, int height)
  {
    int bsize = 256;
    int g     = std::ceil(((float)(width * height)) / bsize);

    dim3 dimBlock(bsize);
    dim3 dimGrid(g);

    gpu_init_label<<<dimGrid, dimBlock>>>(d_in_out, width, height);
    cudaDeviceSynchronize();

    if (cudaPeekAtLastError())
      errx(1, "Computation Error");
  }

  namespace one
  {
    __global__ void gpu_propaged_label(int* d_in_out, bool* changed, int width, int height)
    {
      int p = blockDim.x * blockIdx.x + threadIdx.x;

      int x = p % width;
      int y = p / width;

      if (x >= width || y >= height || d_in_out[p] == 0)
        return;

      int cmin = d_in_out[p];

      int min = d_in_out[p];
      for (int j = -1; j < 2; j++)
      {
        int cy = y + j;
        if (!(0 <= cy && cy < height))
          continue;

        for (int i = -1; i < 2; i++)
        {
          int cx = x + i;
          if (!(0 <= cx && cx < width))
            continue;

          int pos  = cx + cy * width;
          int cpos = d_in_out[pos];
          if (cpos && cpos < min)
            min = cpos;
        }
      }

      if (min < cmin)
      {
        *changed    = true;
        d_in_out[p] = min;
      }
    }

    void propaged_label(int* d_in_out, bool* d_changed, int width, int height)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_propaged_label<<<dimGrid, dimBlock>>>(d_in_out, d_changed, width, height);
      cudaDeviceSynchronize();

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }

    std::set<std::vector<int>> get_connected_components(int* d_in_out, int width, int height, int minimum_pixel)
    {
      init_label(d_in_out, width, height);

      bool  changed   = true;
      bool* h_changed = &changed;

      bool* d_changed;
      int   rc = cudaMalloc(&d_changed, sizeof(bool));
      if (rc)
        errx(1, "Fail buffer allocation for d_changed");

      while (changed)
      {
        changed = false;
        rc      = cudaMemcpy(d_changed, h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        if (rc)
          errx(1, "Fail buffer copy to device");

        propaged_label(d_in_out, d_changed, width, height);

        rc = cudaMemcpy(h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (rc)
          errx(1, "Fail buffer copy to host");
      }

      int* h = static_cast<int*>(std::malloc(sizeof(int) * width * height));
      rc     = cudaMemcpy(h, d_in_out, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
      if (rc)
        errx(1, "Fail buffer copy to host");

      auto ret = cpu::compute_find(h, width, height, minimum_pixel, false);

      std::free(h);

      return ret;
    }
  } // namespace one

  namespace two
  {
    __global__ void gpu_propaged_label(int* d_in, int* d_out, bool* changed, int width, int height)
    {
      int p = blockDim.x * blockIdx.x + threadIdx.x;

      int x = p % width;
      int y = p / width;

      if (x >= width || y >= height || d_in[p] == 0)
        return;

      int cmin = d_in[p];

      int min = d_in[p];
      for (int j = -1; j < 2; j++)
      {
        int cy = y + j;
        if (!(0 <= cy && cy < height))
          continue;

        for (int i = -1; i < 2; i++)
        {
          int cx = x + i;
          if (!(0 <= cx && cx < width))
            continue;

          int pos  = cx + cy * width;
          int cpos = d_in[pos];
          if (cpos && cpos < min)
            min = cpos;
        }
      }

      if (min < cmin)
      {
        *changed = true;
        cmin     = min;
      }

      d_out[p] = cmin;
    }

    void propaged_label(int* d_in, int* d_out, bool* d_changed, int width, int height)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_propaged_label<<<dimGrid, dimBlock>>>(d_in, d_out, d_changed, width, height);
      cudaDeviceSynchronize();

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }

    void swap(int** a, int** b)
    {
      auto tmp = *a;
      *b       = *a;
      *a       = tmp;
    }

    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int* h, int width, int height,
                                                        int minimum_pixel)
    {
      init_label(d_A, width, height);
      cudaMemset((void*)d_B, 0, sizeof(int) * width * height);

      bool  changed   = true;
      bool* h_changed = &changed;

      bool* d_changed;
      int   rc = cudaMalloc(&d_changed, sizeof(bool));
      if (rc)
        errx(1, "Fail changed allocation");

      while (changed)
      {
        changed = false;
        rc      = cudaMemcpy(d_changed, h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        if (rc)
          errx(1, "Fail buffer copy to device");

        propaged_label(d_A, d_B, d_changed, width, height);
        swap(&d_A, &d_B);

        rc = cudaMemcpy(h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (rc)
          errx(1, "Fail buffer copy to host");
      }

      rc = cudaMemcpy(h, d_A, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
      if (rc)
        errx(1, "Fail buffer copy to host");

      auto ret = cpu::compute_find(h, width, height, minimum_pixel, false);

      return ret;
    }
  } // namespace two
} // namespace gpu