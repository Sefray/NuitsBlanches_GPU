#include "pipeline.cuh"

#include <map>

namespace gpu
{
  __global__ void gpu_init_label(int* d_in_out, int width, int height)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    d_in_out[p] *= -(p + 2);
  }

  void init_label(int* d_in_out, int width, int height)
  {
    int bsize = 256;
    int g     = std::ceil(((float)(width * height)) / bsize);

    dim3 dimBlock(bsize);
    dim3 dimGrid(g);

    gpu_init_label<<<dimGrid, dimBlock>>>(d_in_out, width, height);

    if (cudaPeekAtLastError())
      errx(1, "Computation Error");
  }

  __global__ void gpu_relabel(int* d_in_out, int* d_r, int width, int height)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    if (d_in_out[p] < 0)
    {
      int v       = atomicAdd(d_r, 1) + 1;
      d_in_out[p] = -v;
    }
  }

  int relabel(int* d_in_out, int width, int height)
  {
    int  r   = 0;
    int* h_r = &r;
    int* d_r = my_cuda_calloc(sizeof(int));

    int bsize = 256;
    int g     = std::ceil(((float)(width * height)) / bsize);

    dim3 dimBlock(bsize);
    dim3 dimGrid(g);

    gpu_relabel<<<dimGrid, dimBlock>>>(d_in_out, d_r, width, height);

    if (cudaPeekAtLastError())
      errx(1, "Computation Error");

    cudaMemcpy(h_r, d_r, sizeof(int), cudaMemcpyDeviceToHost);

    my_cuda_free(d_r);

    return *h_r;
  }

  __global__ void gpu_compute_find(int* d_image, int* d_image_values, int width, int height, Box* d_boxes)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height)
      return;

    int l = d_image[x + y * width];
    if (!l)
      return;

    while (l > 0)
      l = d_image[l - 2];

    atomicMin(&(d_boxes[-l].xmin), x);
    atomicMin(&(d_boxes[-l].ymin), y);
    atomicMax(&(d_boxes[-l].xmax), x);
    atomicMax(&(d_boxes[-l].ymax), y);

    atomicMax(&(d_boxes[-l].high_pick), d_image_values[x + y * width]);
    atomicAdd(&(d_boxes[-l].size), 1);
  }

  std::set<std::vector<int>> compute_find(int* d_in, int* d_image_values, int width, int height,
                                          int high_pick_threshold, int minimum_pixel, int nb_boxes)
  {
    int  rc      = cudaSuccess;
    Box* h_boxes = static_cast<Box*>(std::calloc(nb_boxes + 1, sizeof(Box)));
    for (int i = 1; i < nb_boxes + 1; i++)
    {
      h_boxes[i].xmin = width;
      h_boxes[i].ymin = height;
    }

    Box* d_boxes;
    rc = cudaMalloc(&d_boxes, sizeof(Box) * (nb_boxes + 1));
    if (rc)
      errx(1, "Error malloc boxes");
    rc = cudaMemcpy(d_boxes, h_boxes, sizeof(Box) * (nb_boxes + 1), cudaMemcpyHostToDevice);
    if (rc)
      errx(1, "Error memcpy h->d boxes");

    int bsize = 256;
    int g     = std::ceil(((float)(width * height)) / bsize);

    dim3 dimBlock(bsize);
    dim3 dimGrid(g);

    gpu_compute_find<<<dimGrid, dimBlock>>>(d_in, d_image_values, width, height, d_boxes);

    if (cudaPeekAtLastError())
      errx(1, "Computation Error");

    rc = cudaMemcpy(h_boxes, d_boxes, sizeof(Box) * (nb_boxes + 1), cudaMemcpyDeviceToHost);
    if (rc)
      errx(1, "Error memcpy d->h boxes");

    std::set<std::vector<int>> ret;
    for (int i = 1; i < nb_boxes + 1; i++)
    {
      auto& box = h_boxes[i];
      if (box.size >= minimum_pixel && box.high_pick >= high_pick_threshold)
        ret.insert({box.xmin, box.ymin, box.xmax - box.xmin + 1, box.ymax - box.ymin + 1});
    }

    std::free(h_boxes);
    cudaFree(d_boxes);

    return ret;
  }

  __device__ int get_min_neighbor(int* d_in_out, int p, int min, int x, int y, int width, int height, int shift)
  {
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

        int pos  = p + i + j * shift;
        int cpos = d_in_out[pos];
        if (cpos < 0)
          cpos *= -1;
        if (cpos && cpos < min)
          min = cpos;
      }
    }

    return min;
  }

  __global__ void gpu_propaged_label(int* d_in, int* d_out, bool* changed, int width, int height)
  {
    int p = blockDim.x * blockIdx.x + threadIdx.x;

    int x = p % width;
    int y = p / width;

    if (x >= width || y >= height || d_in[p] == 0)
      return;

    int cmin = d_in[p];
    if (cmin < 0)
      cmin *= -1;
    int min = get_min_neighbor(d_in, p, cmin, x, y, width, height, width);

    if (min < cmin)
    {
      *changed = true;
      d_out[p] = min;
    }
  }

  void propaged_label(int* d_in, int* d_out, bool* d_changed, int width, int height)
  {
    int bsize = 256;
    int g     = std::ceil(((float)(width * height)) / bsize);

    dim3 dimBlock(bsize);
    dim3 dimGrid(g);

    gpu_propaged_label<<<dimGrid, dimBlock>>>(d_in, d_out, d_changed, width, height);

    if (cudaPeekAtLastError())
      errx(1, "Computation Error");
  }

  namespace one
  {
    std::set<std::vector<int>> get_connected_components(int* d_in_out, int* d_image_values, int width, int height,
                                                        int high_pick_threshold, int minimum_pixel)
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

        propaged_label(d_in_out, d_in_out, d_changed, width, height);

        rc = cudaMemcpy(h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        if (rc)
          errx(1, "Fail buffer copy to host");
      }

      int r = relabel(d_in_out, width, height);

      auto ret = compute_find(d_in_out, d_image_values, width, height, high_pick_threshold, minimum_pixel, r);

      cudaFree(d_changed);

      return ret;
    }
  } // namespace one

  namespace one::two
  {
    void swap(int** a, int** b)
    {
      auto tmp = *a;
      *b       = *a;
      *a       = tmp;
    }

    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int* d_image_values, int width, int height,
                                                        int high_pick_threshold, int minimum_pixel)
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

      int r = relabel(d_A, width, height);

      auto ret = compute_find(d_A, d_image_values, width, height, high_pick_threshold, minimum_pixel, r);

      return ret;
    }
  } // namespace one::two

  namespace one::two::three::four
  {
    __global__ void gpu_propaged_label(int* d_in, int* d_out, bool* changed, int width, int height)
    {
      int p = blockDim.x * blockIdx.x + threadIdx.x;

      int x = p % width;
      int y = p / width;

      if (x >= width || y >= height)
        return;

      for (int s = 0; s < 32 && s + p < width * height; s++)
      {
        int cmin = d_in[p];
        if (cmin != 0)
        {
          if (cmin < 0)
            cmin *= -1;
          int min = get_min_neighbor(d_in, p, cmin, x, y, width, height, width);

          if (min < cmin)
          {
            *changed = true;
            d_out[p] = min;
          }
        }

        // Update (x, y) and p
        x++;
        if (!(x %= width))
        {
          y++;
          if (!(y %= height))
            break;
        }

        p++;
      }
    }

    void propaged_label(int* d_in, int* d_out, bool* d_changed, int width, int height)
    {
      int bsize = 256;
      int g     = std::ceil(((float)(width * height)) / bsize);

      dim3 dimBlock(bsize);
      dim3 dimGrid(g);

      gpu_propaged_label<<<dimGrid, dimBlock>>>(d_in, d_out, d_changed, width, height);

      if (cudaPeekAtLastError())
        errx(1, "Computation Error");
    }

    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int* d_image_values, int width, int height,
                                                        int high_pick_threshold, int minimum_pixel)
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
      cudaFree(d_changed);

      int r = relabel(d_A, width, height);

      auto ret = compute_find(d_A, d_image_values, width, height, high_pick_threshold, minimum_pixel, r);

      return ret;
    }
  } // namespace one::two::three::four
} // namespace gpu