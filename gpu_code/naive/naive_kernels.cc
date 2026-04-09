#include <hip/hip_runtime.h>
#include "../../tensor_implementations/blco_impl.h"
#include "../kernel_utils.h"


//======================================================================
// Kernel 1: 3D MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_3D_kernel_naive(
    int mode, 
    BLCO_BLOCK_GPU<T>* tensor, 
    uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, 
    T* m1_fmat, T* m2_fmat, T* m3_fmat, 
    int d1, int d2, int d3, 
    int num_blocks, 
    int rank)
{
    // 1. Determine which NNZ this thread handles
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= nnz) return;

    // 2. Extract Entry (utilizing existing helper)
    int block = 0; // extract_entry handles block identification internally
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    uint64_t lin_index = entry.index;
    T value = entry.value;

    // 3. Decode Coordinates
    int bit_widths[3] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3)};
    uint64_t masks[3] = {m1_mask, m2_mask, m3_mask};

    int coords[3];
    #pragma unroll
    for(int i = 0; i < 3; i++) {
        // extract_mode is a utility from your kernel_utils.h/blco_impl.h
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 3, block);
    }

    // 4. Identify target and non-target matrices
    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};
    const int nt1_idx = (mode == 1) ? 1 : 0;          // Mode 1 uses 2 & 3; Mode 2 uses 1 & 3; Mode 3 uses 1 & 2
    const int nt2_idx = (mode == 3) ? 1 : 2;

    int target_coord = coords[mode - 1];
    int nt1_coord = coords[nt1_idx];
    int nt2_coord = coords[nt2_idx];

    T* target_fmat = fmat_list[mode - 1];
    T* matB = fmat_list[nt1_idx];
    T* matC = fmat_list[nt2_idx];

    // 5. Compute and Output
    // Instead of shuffles, every thread loops through the entire rank
    for (int r = 0; r < rank; ++r) {
        // MTTKRP Contribution: V * B(i2, r) * C(i3, r)
        T contribution = value * matB[nt1_coord * rank + r] * matC[nt2_coord * rank + r];

        // Direct atomic update to global memory
        atomicAdd(&target_fmat[target_coord * rank + r], contribution);
    }
}

//======================================================================
// Kernel 2: 4D MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_4D_kernel_naive(
    int mode, 
    BLCO_BLOCK_GPU<T>* tensor, 
    uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask,
    T* m1_fmat, T* m2_fmat, T* m3_fmat, T* m4_fmat,
    int d1, int d2, int d3, int d4,
    int num_blocks, 
    int rank)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= nnz) return;

    int block = 0; 
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    uint64_t lin_index = entry.index;
    T value = entry.value;

    int bit_widths[4] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4)};
    uint64_t masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};
    T* fmat_list[4] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat};

    // Decode all 4 coordinates
    int coords[4];
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 4, block);
    }

    // Identify target and indices for the 3 source matrices
    int target_idx = mode - 1;
    int s1 = (mode % 4); 
    int s2 = (mode + 1) % 4;
    int s3 = (mode + 2) % 4;

    for (int r = 0; r < rank; ++r) {
        // Product of value and the 3 other factor matrix rows
        T product = value * fmat_list[s1][coords[s1] * rank + r] 
                          * fmat_list[s2][coords[s2] * rank + r] 
                          * fmat_list[s3][coords[s3] * rank + r];

        atomicAdd(&fmat_list[target_idx][coords[target_idx] * rank + r], product);
    }
}

//======================================================================
// Kernel 3: 5D MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_5D_kernel_naive(
    int mode, 
    BLCO_BLOCK_GPU<T>* tensor, 
    uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, uint64_t m5_mask,
    T* m1_fmat, T* m2_fmat, T* m3_fmat, T* m4_fmat, T* m5_fmat,
    int d1, int d2, int d3, int d4, int d5,
    int num_blocks, 
    int rank)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= nnz) return;

    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    uint64_t lin_index = entry.index;
    T value = entry.value;

    int bit_widths[5] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), 
                         ceiling_log2(d4), ceiling_log2(d5)};
    uint64_t masks[5] = {m1_mask, m2_mask, m3_mask, m4_mask, m5_mask};
    T* fmat_list[5] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat, m5_fmat};

    int coords[5];
    #pragma unroll
    for(int i = 0; i < 5; i++) {
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 5, block);
    }

    int target_idx = mode - 1;
    // Get the 4 non-target indices
    int sources[4];
    int s_count = 0;
    for(int i = 0; i < 5; i++) {
        if(i != target_idx) sources[s_count++] = i;
    }

    for (int r = 0; r < rank; ++r) {
        // Multi-way product across all dimensions except the target
        T product = value;
        #pragma unroll
        for(int s = 0; s < 4; s++) {
            int m = sources[s];
            product *= fmat_list[m][coords[m] * rank + r];
        }

        atomicAdd(&fmat_list[target_idx][coords[target_idx] * rank + r], product);
    }
}


//======================================================================
// Host Wrapper: MTTKRP_BLCO
//======================================================================
template<typename T, typename S>
void MTTKRP_BLCO_Naive(int mode, Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    //Dimensions
    const std::vector<int> dims = sparse_tensor.get_dims();
    int num_dimensions = dims.size();
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t non_zeros = sparse_tensor.get_nnz();
    int num_blocks = sparse_tensor.get_num_blocks();

    //Masks
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Allocate device pointers
    MTTKRP_Device_Resources<T> res = allocate_mttkrp_resources(sparse_tensor);

    bool collect_times = false;
    if(times.size() == 0) collect_times = true;
    std::pair<int,int> dimensions;

    if(num_dimensions == 3){
        dimensions = determine_dimensions_no_smem(non_zeros); //Determine Dimensions

        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_3D_kernel_naive<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, sparse_tensor.get_nnz(),
                masks[0], masks[1], masks[2],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
                dims[0], dims[1], dims[2],
                num_blocks, rank
            );

            // Record stop
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));

            // Compute elapsed time in ms
            float milliseconds = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

            if(collect_times) times.push_back(milliseconds);

            // Clean up
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }
    }
    else if(num_dimensions == 4){
        dimensions = determine_dimensions_no_smem(non_zeros); //Determine Dimensions

        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_4D_kernel_naive<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, sparse_tensor.get_nnz(),
                masks[0], masks[1], masks[2], masks[3],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2], res.d_fmats[3],
                dims[0], dims[1], dims[2], dims[3], num_blocks, rank
            );

            // Record stop
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));

            // Compute elapsed time in ms
            float milliseconds = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

            if(collect_times) times.push_back(milliseconds);

            // Clean up
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }
    }
    else if(num_dimensions == 5){
        dimensions = determine_dimensions_no_smem(non_zeros); //Determine Dimensions

        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_5D_kernel_naive<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, sparse_tensor.get_nnz(),
                masks[0], masks[1], masks[2], masks[3], masks[4],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2], res.d_fmats[3],
                res.d_fmats[4], dims[0], dims[1], dims[2], dims[3], dims[4],
                num_blocks, rank
            );

            // Record stop
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));

            // Compute elapsed time in ms
            float milliseconds = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

            if(collect_times) times.push_back(milliseconds);

            // Clean up
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }

    }
        
    size_t out_size = dims[mode-1] * rank;
    std::vector<T> result(out_size);
    HIP_CHECK(hipMemcpy(result.data(), res.d_fmats[mode-1], sizeof(T) * out_size, hipMemcpyDeviceToHost));
    sparse_tensor.reassign_fmat(mode, result);
    deallocate_mttkrp_resources(res, num_blocks);
}

//======================================================================
// Explicit template instantiations
//======================================================================
template void MTTKRP_BLCO_Naive<int, uint64_t>(int, Blco_Tensor<int, uint64_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<float, uint64_t>(int, Blco_Tensor<float, uint64_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<unsigned long long, uint64_t>(int, Blco_Tensor<unsigned long long, uint64_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<double, uint64_t>(int, Blco_Tensor<double, uint64_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<int, __uint128_t>(int, Blco_Tensor<int, __uint128_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<float, __uint128_t>(int, Blco_Tensor<float, __uint128_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<unsigned long long, __uint128_t>(int, Blco_Tensor<unsigned long long, __uint128_t>&, std::vector<float>&, int);
template void MTTKRP_BLCO_Naive<double, __uint128_t>(int, Blco_Tensor<double, __uint128_t>&, std::vector<float>&, int);
