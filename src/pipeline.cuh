#include "pipeline.hh"

#include <cuda_runtime.h>

namespace gpu
{
  void my_cuda_mem_copy_dth(const int* h, int* d, size_t n);
  int* malloc_and_copy(const int* h, int width, int height);
  void my_cuda_free(int* d);
  int* my_cuda_malloc(size_t n);
  int* my_cuda_calloc(size_t n);

  void binary_image(int* d_in_out, int width, int height, int threshold);

  namespace one
  {
    int* greyscale(unsigned char* d_in, int width, int height);
    int* smoothing(int* d_in, int width, int height, int kernel_size);
    int* compute_difference(int* d_ref_in, int* d_in, int width, int height);
    int* closing_opening(int* d_A, int width, int height, int kernel_size_opening, int kernel_size_closing);
    std::set<std::vector<int>> get_connected_components(int* d_in, int width, int height, int minimum_pixel);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, int minimum_pixel);
  } // namespace one

  namespace one::two
  {
    void greyscale(unsigned char* d_in, int* d_out, int width, int height);
    void smoothing(int* d_in, int* d_out, int width, int height, int kernel_size);
    void compute_difference(int* d_ref_in, int* d_in, int* d_out, int width, int height);
    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing);
    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int width, int height, int minimum_pixel);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, int minimum_pixel, unsigned char* buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B);
  } // namespace one::two

  namespace one::two::three
  {
    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, int minimum_pixel, unsigned char* buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B);
  } // namespace one::two::three

  namespace one::two::three::four
  {
    std::set<std::vector<int>> get_connected_components(int* d_A, int* d_B, int width, int height, int minimum_pixel);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, int minimum_pixel, unsigned char* buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B);
  } // namespace one::two::three::four

  namespace one::two::three::four::five
  {
    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, int minimum_pixel, unsigned char* buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B);
  } // namespace one::two::three::four::five

  namespace one::two::three::four::five::six
  {
    constexpr int nb_stream = 4;

    void closing_opening(int* d_A, int* d_B, int width, int height, int kernel_size_opening, int kernel_size_closing,
                         std::vector<cudaStream_t>& streams);

    std::set<std::vector<int>> pipeline(int* ref_smoothed, unsigned char* modified, int width, int height,
                                        int kernel_size, int kernel_size_opening, int kernel_size_closing,
                                        int binary_threshold, int minimum_pixel, unsigned char* buffer_uc,
                                        int* d_buffer_A, int* d_buffer_B, std::vector<cudaStream_t>& streams);
  } // namespace one::two::three::four::five::six
} // namespace gpu