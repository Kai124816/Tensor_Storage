#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

//======================================================================
// Kernel 1: Non-Hierarchical 5D MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_5D_kernel_1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, uint64_t m5_mask, 
T* m1_fmat, T* m2_fmat, T* m3_fmat, T* m4_fmat, T* m5_fmat, 
int d1, int d2, int d3, int d4, int d5, int num_blocks, int rank, int wavefront_size = 64)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (global_idx < nnz);
    
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    uint64_t lin_index = entry.index;
    T value = entry.value;

    int bit_widths[5] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4), ceiling_log2(d5)};
    uint64_t masks[5] = {m1_mask, m2_mask, m3_mask, m4_mask, m5_mask};

    int coords[5];
    for(int i = 0; i < 5; i++) {
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 5, block);
    }

    int target_index = coords[mode - 1] - !active;

    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, value, total_modes, mode_num);
    __builtin_amdgcn_s_barrier();

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    // Non-target modes for 5D: if target is mode n, multiply the other 4
    const int nt1_list[5] = {2, 1, 1, 1, 1};
    const int nt2_list[5] = {3, 3, 2, 2, 2};
    const int nt3_list[5] = {4, 4, 4, 3, 3};
    const int nt4_list[5] = {5, 5, 5, 5, 4};
    
    int nt1 = nt1_list[mode - 1], nt2 = nt2_list[mode - 1], nt3 = nt3_list[mode - 1], nt4 = nt4_list[mode - 1];
    T* fmat_list[5] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat, m5_fmat};

    int peer_lanes[64];
    int count = 0;
    unsigned long long mask_copy = wavefront_mask;
    while(mask_copy) {
        int peer = __ffsll(mask_copy) - 1;
        mask_copy &= (mask_copy - 1);
        peer_lanes[count++] = peer;
    }

    int target_base = (coords[mode - 1]) * rank;
    bool acc;
    target_index += !active; //Set non target indices back to 0
    for (int j = 0; j < rank; ++j) {
        T sum = 0;
        acc = (j >= s1) && (j < e1);
        for (int i = 0; i < count; ++i) {
            int peer = peer_lanes[i];
            T val = __shfl(value, peer, wavefront_size);
            int idx1 = __shfl(coords[nt1-1], peer, wavefront_size) * rank + j;
            int idx2 = __shfl(coords[nt2-1], peer, wavefront_size) * rank + j;
            int idx3 = __shfl(coords[nt3-1], peer, wavefront_size) * rank + j;
            int idx4 = __shfl(coords[nt4-1], peer, wavefront_size) * rank + j;
            
            sum += fmat_list[nt1-1][idx1] * fmat_list[nt2-1][idx2] * fmat_list[nt3-1][idx3] * 
            fmat_list[nt4-1][idx4] * val * acc;
        }
        if(active) atomicAdd(&fmat_list[mode-1][target_base + j], sum);
    }
}

//======================================================================
// Kernel 2: Hierarchical 5D MTTKRP (Shared Memory)
//======================================================================
template<typename T>
__global__ void mttkrp_5D_kernel_2(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, uint64_t m5_mask, 
T* m1_fmat, T* m2_fmat, T* m3_fmat, T* m4_fmat, T* m5_fmat, 
int d1, int d2, int d3, int d4, int d5, int num_blocks, int rank, int smem_size, int wavefront_size = 64)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    for (int i = block_idx; i < smem_size; i += blockDim.x) smem[i] = 0;
    __syncthreads();

    bool active = (global_idx < nnz);
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    
    int bit_widths[5] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4), ceiling_log2(d5)};
    uint64_t masks[5] = {m1_mask, m2_mask, m3_mask, m4_mask, m5_mask};
    int coords[5];
    for(int i = 0; i < 5; i++) coords[i] = extract_mode(entry.index, i + 1, masks, bit_widths, 5, block);

    int target_index = coords[mode-1] - !active;
    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, entry.value, total_modes, mode_num);

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    const int nt_modes[4][5] = {
        {2,1,1,1,1}, // nt1
        {3,3,2,2,2}, // nt2
        {4,4,4,3,3}, // nt3
        {5,5,5,5,4}  // nt4
    };

    T* fmat_list[5] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat, m5_fmat};
    int fmat_sizes[5] = {d1*rank, d2*rank, d3*rank, d4*rank, d5*rank};

    unsigned long long mask_copy = wavefront_mask;
    int peer_lanes[64], count = 0;
    while(mask_copy) { peer_lanes[count++] = __ffsll(mask_copy) - 1; mask_copy &= (mask_copy - 1); }

    int target_base = coords[mode-1] * rank;
    bool acc;
    target_index += !active; //Set non target indices back to 0
    for (int j = 0; j < rank; ++j) {
        T sum = 0;
        acc = (j >= s1) && (j < e1);
        for (int i = 0; i < count; ++i) {
            int p = peer_lanes[i];
            int idx1 = __shfl(coords[nt_modes[0][mode-1]-1], p, wavefront_size) * rank + j;
            int idx2 = __shfl(coords[nt_modes[1][mode-1]-1], p, wavefront_size) * rank + j;
            int idx3 = __shfl(coords[nt_modes[2][mode-1]-1], p, wavefront_size) * rank + j;
            int idx4 = __shfl(coords[nt_modes[3][mode-1]-1], p, wavefront_size) * rank + j;
            
            sum += fmat_list[nt_modes[0][mode-1]-1][idx1] * fmat_list[nt_modes[1][mode-1]-1][idx2] * fmat_list[nt_modes[2][mode-1]-1][idx3] 
            * fmat_list[nt_modes[3][mode-1]-1][idx4] * __shfl(entry.value, p, wavefront_size) * acc;
        }
        int out_idx = target_base + j;
        if (active && out_idx < smem_size) atomicAdd(&smem[out_idx], sum);
        else if (active && out_idx < fmat_sizes[mode-1]) atomicAdd(&fmat_list[mode-1][out_idx], sum);
    }

    __syncthreads();
    for (int i = block_idx; i < fmat_sizes[mode-1]; i += blockDim.x) {
        if (smem[i] != 0) atomicAdd(&fmat_list[mode-1][i], smem[i]);
    }
}

//======================================================================
// Host Wrapper: MTTKRP_BLCO_5D
//======================================================================
template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_5D(int mode, const Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    const std::vector<int> dims = sparse_tensor.get_dims();
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t nnz = sparse_tensor.get_nnz();
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    int num_blocks = sparse_tensor.get_num_blocks();

    // Allocate device pointers
    MTTKRP_Device_Resources<T> res = allocate_mttkrp_resources(sparse_tensor);

    bool is_hierarchical = (get_compute_units() > dims[mode - 1]);
    std::pair<int,int> grid = determine_dimensions_no_smem(nnz);

    bool collect_times = false;
    if(times.size() == 0) collect_times = true;

    for(int i = 0; i < iter; i++) {
        if(!is_hierarchical) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(mttkrp_5D_kernel_1<T>, dim3(grid.first), dim3(grid.second), 0, 0,
                mode, res.d_tensor, nnz, masks[0], masks[1], masks[2], masks[3], masks[4],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2], res.d_fmats[3], res.d_fmats[4],
                dims[0], dims[1], dims[2], dims[3], dims[4], num_blocks, rank);

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
        } else {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            size_t shared_mem = get_max_shared_memory();
            hipLaunchKernelGGL(mttkrp_5D_kernel_2<T>, dim3(grid.first), dim3(grid.second), shared_mem, 0,
                mode, res.d_tensor, nnz, masks[0], masks[1], masks[2], masks[3], masks[4],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2], res.d_fmats[3], res.d_fmats[4],
                dims[0], dims[1], dims[2], dims[3], dims[4], num_blocks, rank, (int)(shared_mem/sizeof(T)));

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

    deallocate_mttkrp_resources(res, num_blocks);

    return result;
}