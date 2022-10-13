#include <stdio.h>

#include "test.cuh"

__global__ void kernel(void)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    printf("Hello from thread %d in block %d \n", tid, bid);
}

void kernel_cxx(void)
{
    kernel<<<4, 4>>>();
    cudaDeviceSynchronize();
}
