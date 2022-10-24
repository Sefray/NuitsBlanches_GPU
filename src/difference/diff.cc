#include "src/pipeline.hh"

namespace cpu
{
    int *compute_difference(int *ref_smoothed, int *modified_smoothed, int width, int height)
    {
        int *ret = static_cast<int *>(std::malloc(sizeof(int) * width * height));

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                ret[y * width + x] = std::abs(ref_smoothed[y * width + x] - modified_smoothed[y * width + x]);

        std::free(modified_smoothed);

        return ret;
    }
}