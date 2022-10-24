#include "pipeline.hh"

namespace gpu
{
    __global__ void gpu_binary_image(int *d_in_out, int width, int height, int threshold)
    {
        int p = blockDim.x * blockIdx.x + threadIdx.x;

        int x = p % width;
        int y = p / width;

        if (x >= width || y >= height)
            return;

        d_in_out[p] = d_in_out[p] < threshold ? 0 : 1;
    }

    void binary_image(int *d_in_out, int width, int height, int threshold)
    {
        int bsize = 256;
        int g = std::ceil(((float)(width * height)) / bsize);

        dim3 dimBlock(bsize);
        dim3 dimGrid(g);

        gpu_binary_image<<<dimGrid, dimBlock>>>(d_in_out, width, height, threshold);
        cudaDeviceSynchronize();

        if (cudaPeekAtLastError())
            errx(1, "Computation Error");
    }
}