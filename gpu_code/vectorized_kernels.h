#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

//======================================================================
// Vectorized Kernels
//======================================================================
template<typename T>
__global__ void mttkrp_3D_kernel_vectorized(
    int mode, 
    BLCO_BLOCK_GPU<T>* tensor, 
    uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, 
    T* m1_fmat, T* m2_fmat, T* m3_fmat, 
    int d1, int d2, int d3, 
    int num_blocks, 
    int rank, int stride)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Group's global ID and its assigned non-zeros
    int group_id = global_thread / rank;
    uint64_t total_groups = ((uint64_t)gridDim.x * blockDim.x) / rank;
    
    // Group's idx within the wavefront (handles multiple groups per wave)
    int wavefront_group = (threadIdx.x % 64) / rank;
    
    // Which column of the factor matrix is this specific thread computing?
    int r = global_thread % rank; 

    int bit_widths[3] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3)};
    uint64_t masks[3] = {m1_mask, m2_mask, m3_mask};
    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};
    const int nt1_idx = (mode == 1) ? 1 : 0;          
    const int nt2_idx = (mode == 3) ? 1 : 2;

    // Fixed stride loop to ensure no warp divergence
    for(int s = 0; s < stride; s++) {
        uint64_t idx = group_id + (uint64_t)s * total_groups;
        bool active = (idx < nnz) && (r < rank);

        int target_coord = -1;
        T contribution = (T)0;

        if (active) {
            int block = 0; 
            BLCO_ENTRY<T> entry = extract_entry_by_idx(tensor, num_blocks, idx, block); 
            uint64_t lin_index = entry.index;
            T value = entry.value;

            int coords[3];
            #pragma unroll
            for(int i = 0; i < 3; i++) {
                coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 3, block);
            }

            target_coord = coords[mode - 1];
            int nt1_coord = coords[nt1_idx];
            int nt2_coord = coords[nt2_idx];

            T matB_val = fmat_list[nt1_idx][nt1_coord * rank + r];
            T matC_val = fmat_list[nt2_idx][nt2_coord * rank + r];
            
            contribution = value * matB_val * matC_val;
        }

        // Leader keeps track if it's the valid thread to output
        bool is_leader = active;
        T total_contribution = contribution;

        for(int i = 0; i < 64; i += rank){
            int peer_target = __shfl(target_coord, i, 64);
            int shuffle_group = i / rank;
            
            if(peer_target == target_coord && target_coord != -1){
                int cousin_idx = (shuffle_group * rank) + r;
                total_contribution += __shfl(contribution, cousin_idx, 64) * (shuffle_group != wavefront_group);

                if(shuffle_group < wavefront_group){
                    is_leader = false;
                }
            }
        }

        if(is_leader && active) {
            atomicAdd(&fmat_list[mode - 1][target_coord * rank + r], total_contribution);
        }
    }
}

template<typename T>
__global__ void mttkrp_4D_kernel_vectorized(
    int mode, 
    BLCO_BLOCK_GPU<T>* tensor, 
    uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask,
    T* m1_fmat, T* m2_fmat, T* m3_fmat, T* m4_fmat,
    int d1, int d2, int d3, int d4, 
    int num_blocks, 
    int rank, int stride)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int group_id = global_thread / rank;
    uint64_t total_groups = ((uint64_t)gridDim.x * blockDim.x) / rank;
    
    int wavefront_group = (threadIdx.x % 64) / rank;
    int r = global_thread % rank; 

    int bit_widths[4] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4)};
    uint64_t masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};
    T* fmat_list[4] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat};
    
    int nt1_idx, nt2_idx, nt3_idx;
    if (mode == 1) { nt1_idx = 1; nt2_idx = 2; nt3_idx = 3; }
    else if (mode == 2) { nt1_idx = 0; nt2_idx = 2; nt3_idx = 3; }
    else if (mode == 3) { nt1_idx = 0; nt2_idx = 1; nt3_idx = 3; }
    else { nt1_idx = 0; nt2_idx = 1; nt3_idx = 2; }

    for(int s = 0; s < stride; s++) {
        uint64_t idx = group_id + (uint64_t)s * total_groups;
        bool active = (idx < nnz) && (r < rank);

        int target_coord = -1;
        T contribution = (T)0;

        if (active) {
            int block = 0; 
            BLCO_ENTRY<T> entry = extract_entry_by_idx(tensor, num_blocks, idx, block); 
            uint64_t lin_index = entry.index;
            T value = entry.value;

            int coords[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 4, block);
            }

            target_coord = coords[mode - 1];
            int nt1_coord = coords[nt1_idx];
            int nt2_coord = coords[nt2_idx];
            int nt3_coord = coords[nt3_idx];

            T matB_val = fmat_list[nt1_idx][nt1_coord * rank + r];
            T matC_val = fmat_list[nt2_idx][nt2_coord * rank + r];
            T matD_val = fmat_list[nt3_idx][nt3_coord * rank + r];
            
            contribution = value * matB_val * matC_val * matD_val;
        }

        bool is_leader = active;
        T total_contribution = contribution;

        for(int i = 0; i < 64; i += rank){
            int peer_target = __shfl(target_coord, i, 64);
            int shuffle_group = i / rank;
            
            if(peer_target == target_coord && target_coord != -1){
                int cousin_idx = (shuffle_group * rank) + r;
                total_contribution += __shfl(contribution, cousin_idx, 64) * (shuffle_group != wavefront_group);

                if(shuffle_group < wavefront_group){
                    is_leader = false;
                }
            }
        }

        if(is_leader && active) {
            atomicAdd(&fmat_list[mode - 1][target_coord * rank + r], total_contribution);
        }
    }
}

template<typename T>
__global__ void mttkrp_5D_kernel_vectorized(
    int mode, 
    BLCO_BLOCK_GPU<T>* tensor, 
    uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, uint64_t m5_mask,
    T* m1_fmat, T* m2_fmat, T* m3_fmat, T* m4_fmat, T* m5_fmat,
    int d1, int d2, int d3, int d4, int d5, 
    int num_blocks, 
    int rank, int stride)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int group_id = global_thread / rank;
    uint64_t total_groups = ((uint64_t)gridDim.x * blockDim.x) / rank;
    
    int wavefront_group = (threadIdx.x % 64) / rank;
    int r = global_thread % rank; 

    int bit_widths[5] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4), ceiling_log2(d5)};
    uint64_t masks[5] = {m1_mask, m2_mask, m3_mask, m4_mask, m5_mask};
    T* fmat_list[5] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat, m5_fmat};
    
    int nt1_idx, nt2_idx, nt3_idx, nt4_idx;
    if (mode == 1) { nt1_idx = 1; nt2_idx = 2; nt3_idx = 3; nt4_idx = 4; }
    else if (mode == 2) { nt1_idx = 0; nt2_idx = 2; nt3_idx = 3; nt4_idx = 4; }
    else if (mode == 3) { nt1_idx = 0; nt2_idx = 1; nt3_idx = 3; nt4_idx = 4; }
    else if (mode == 4) { nt1_idx = 0; nt2_idx = 1; nt3_idx = 2; nt4_idx = 4; }
    else { nt1_idx = 0; nt2_idx = 1; nt3_idx = 2; nt4_idx = 3; }

    for(int s = 0; s < stride; s++) {
        uint64_t idx = group_id + (uint64_t)s * total_groups;
        bool active = (idx < nnz) && (r < rank);

        int target_coord = -1;
        T contribution = (T)0;

        if (active) {
            int block = 0; 
            BLCO_ENTRY<T> entry = extract_entry_by_idx(tensor, num_blocks, idx, block); 
            uint64_t lin_index = entry.index;
            T value = entry.value;

            int coords[5];
            #pragma unroll
            for(int i = 0; i < 5; i++) {
                coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 5, block);
            }

            target_coord = coords[mode - 1];
            int nt1_coord = coords[nt1_idx];
            int nt2_coord = coords[nt2_idx];
            int nt3_coord = coords[nt3_idx];
            int nt4_coord = coords[nt4_idx];

            T matB_val = fmat_list[nt1_idx][nt1_coord * rank + r];
            T matC_val = fmat_list[nt2_idx][nt2_coord * rank + r];
            T matD_val = fmat_list[nt3_idx][nt3_coord * rank + r];
            T matE_val = fmat_list[nt4_idx][nt4_coord * rank + r];
            
            contribution = value * matB_val * matC_val * matD_val * matE_val;
        }

        bool is_leader = active;
        T total_contribution = contribution;

        for(int i = 0; i < 64; i += rank){
            int peer_target = __shfl(target_coord, i, 64);
            int shuffle_group = i / rank;
            
            if(peer_target == target_coord && target_coord != -1){
                int cousin_idx = (shuffle_group * rank) + r;
                total_contribution += __shfl(contribution, cousin_idx, 64) * (shuffle_group != wavefront_group);

                if(shuffle_group < wavefront_group){
                    is_leader = false;
                }
            }
        }

        if(is_leader && active) {
            atomicAdd(&fmat_list[mode - 1][target_coord * rank + r], total_contribution);
        }
    }
}


template<typename T, typename S>
void MTTKRP_BLCO_VEC(int mode, Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    //Dimensions
    const std::vector<int> dims = sparse_tensor.get_dims();
    int rows = dims[0];
    int cols = dims[1];
    int depth = dims[2];
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t non_zeros = sparse_tensor.get_nnz();
    int num_blocks = sparse_tensor.get_num_blocks();

    //Masks
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    uint64_t m1_mask = masks[0];
    uint64_t m2_mask = masks[1];
    uint64_t m3_mask = masks[2];

    // Allocate device pointers
    MTTKRP_Device_Resources<T> res = allocate_mttkrp_resources(sparse_tensor);

    bool collect_times = false;
    if(times.size() == 0) collect_times = true;
    
    // Each group of "rank" threads processes a batch of "stride" non-zeros.
    int total_simds = get_total_simd_units();
    int stride = determine_stride(total_simds, rank, non_zeros); 
    std::pair<int,int> dimensions = determine_dimensions_vectorized<T>(non_zeros, rank, stride, 256);

    if(dims.size() == 3){
        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_3D_kernel_vectorized<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, non_zeros,
                m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
                rows, cols, depth,
                num_blocks, rank, stride
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
    else if(dims.size() == 4){
        uint64_t m4_mask = masks[3];
        int dim_4 = dims[3];
        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_4D_kernel_vectorized<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, non_zeros,
                m1_mask, m2_mask, m3_mask, m4_mask,
                res.d_fmats[0], res.d_fmats[1], 
                res.d_fmats[2], res.d_fmats[3], rows, 
                cols, depth, dim_4, num_blocks, rank, stride
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
    else if(dims.size() == 5){
        uint64_t m4_mask = masks[3];
        int dim_4 = dims[3];
        uint64_t m5_mask = masks[4];
        int dim_5 = dims[4];
        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_5D_kernel_vectorized<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, non_zeros,
                m1_mask, m2_mask, m3_mask, m4_mask,
                m5_mask, res.d_fmats[0], res.d_fmats[1], 
                res.d_fmats[2], res.d_fmats[3], res.d_fmats[4], 
                rows, cols, depth, dim_4, dim_5, num_blocks, 
                rank, stride
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
