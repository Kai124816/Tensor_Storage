#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

template<typename T>
__global__ void mttkrp_4D_kernel_1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, 
T* m4_fmat, int rows, int cols, int depth, int num_blocks, int rank, int wavefront_size = 64)
{
    return;
}

template<typename T>
__global__ void mttkrp_4D_kernel_2(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, 
T* m4_fmat, int rows, int cols, int depth, int num_blocks, int rank, int wf_indices, int wavefront_size = 64)
{
    return;
}


template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_4D(int mode, const Blco_Tensor<T,S>& sparse_tensor, int iter = 1, std::vector<int> times = {0})
{
    std::vector<T> temp;
    return temp;
}