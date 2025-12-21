#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

//======================================================================
// Kernel 1: Non-Hierarchical MTTKRP
//======================================================================
// Each thread computes contributions for one nonzero and directly
// atomically updates the factor matrices. Uses wavefront shuffles
// to reduce redundant memory accesses. 
// Steps:
// 1. Determine which NNZ this thread handles
// 2. Decode (i,j,k) coordinates from BLCO index
// 3. Use wavefront collectives (__shfl) to share peer indices/values
// 4. Compute contributions to target factor matrix
// 5. atomicAdd() to avoid race conditions
//======================================================================
template<typename T>
__global__ void mttkrp_3D_kernel_1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask,
T* m1_fmat, T* m2_fmat, T* m3_fmat, int rows, int cols, int depth, int num_blocks, int rank, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;
    int wavefront_idx = threadIdx.x % 64;

    bool active = true;
    if (global_idx >= nnz) active = false; 
    
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor,num_blocks,block);
    uint64_t lin_index = entry.index;
    T value = entry.value;
    int m1_bits = ceiling_log2(rows), m2_bits = ceiling_log2(cols), m3_bits = ceiling_log2(depth);
    int bit_widths[3] = {m1_bits,m2_bits,m3_bits};
    uint64_t masks[3] = {m1_mask,m2_mask,m3_mask};

    int m1_index; int m2_index; int m3_index;
    m1_index = extract_mode(lin_index, 1, masks, bit_widths, 3, block);
    m2_index = extract_mode(lin_index, 2, masks, bit_widths, 3, block);
    m3_index = extract_mode(lin_index, 3, masks, bit_widths, 3, block);
    int idx_array[3] = {m1_index, m2_index,m3_index};
    int target_index = idx_array[mode - 1] - !active;

    int mode_num; //Thread represents the ith mode within with a given index the wavefront
    int total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index,value,total_modes,mode_num);
    __builtin_amdgcn_s_barrier();

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0; // ceil(rank/total_modes)
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    // Lookup tables for mode → nt_mode1, nt_mode2
    const int nt_mode1_list[3] = {2, 1, 1};
    const int nt_mode2_list[3] = {3, 3, 2};
    int nt_mode1 = nt_mode1_list[mode - 1];
    int nt_mode2 = nt_mode2_list[mode - 1];

    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};
    int fmat_sizes[3] = {rows * rank, cols * rank, depth * rank};
    int indices[3] = {m1_index, m2_index, m3_index};

    int peer_indices[64];  // max wavefront size
    int count = 0;

    unsigned long long mask = wavefront_mask;
    while(mask) {
        int peer_lane = __ffsll(mask) - 1;  // extract lowest set bit
        mask &= (mask - 1);
        peer_indices[count++] = peer_lane;
    }

    // Precompute all values from __shfl into temporary arrays
    int temp_nt1[64], temp_nt2[64];
    T temp_val[64];
    for(int i = 0; i < count; i++) {
        int peer = peer_indices[i];

        // Pick source values based on mode
        int src1 = (mode == 1 ? m2_index : (mode == 2 ? m1_index : m1_index));
        int src2 = (mode == 1 ? m3_index : (mode == 2 ? m3_index : m2_index));

        temp_nt1[i] = __shfl(src1, peer, wavefront_size) * rank;
        temp_nt2[i] = __shfl(src2, peer, wavefront_size) * rank;
        temp_val[i] = __shfl(value, peer, wavefront_size);
    }

    target_index += !active;
    int target_base = target_index * rank;
    for (int j = s1; j < e1; ++j) {
        T sum = (T)0;
        for (int i = 0; i < count; ++i) {
            int idx2 = temp_nt1[i] + j;
            int idx3 = temp_nt2[i] + j;
            // optional bounds check removed for speed if you validated elsewhere
            sum += fmat_list[nt_mode1 - 1][idx2] * fmat_list[nt_mode2 - 1][idx3] * temp_val[i];
        }
        // one atomic per j instead of one per (i,j)
        atomicAdd(&fmat_list[mode - 1][target_base + j], sum);
    }          
}

//======================================================================
// Kernel 2: Hierarchical MTTKRP
//======================================================================
// Threads within a wavefront accumulate partial results into shared
// memory, then only a leader thread performs the atomicAdd to global
// memory. Reduces contention and improves scalability.
// Steps:
    // 1. Zero shared memory buffer
    // 2. Threads compute contributions and add into shared memory
    // 3. Barrier sync
    // 4. Leader threads write results back to global factor matrices
//======================================================================
template<typename T>
__global__ void mttkrp_3D_kernel_2(
    int mode,
    BLCO_BLOCK_GPU<T>* tensor,
    uint64_t nnz,
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask,
    T* m1_fmat, T* m2_fmat, T* m3_fmat,
    int rows, int cols, int depth,
    int num_blocks, int rank, int smem_size,
    int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx  = threadIdx.x;
    int wavefront_idx = threadIdx.x % 64;

    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    // initialize the partial_fmat
    for (int i = threadIdx.x; i < smem_size; i += blockDim.x) {
        smem[i] = 0;
    }

    bool active = true;
    if (global_idx >= nnz) active = false; 
    
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor,num_blocks,block);
    uint64_t lin_index = entry.index;
    T value = entry.value;
    int m1_bits = ceiling_log2(rows), m2_bits = ceiling_log2(cols), m3_bits = ceiling_log2(depth);
    int bit_widths[3] = {m1_bits,m2_bits,m3_bits};
    uint64_t masks[3] = {m1_mask,m2_mask,m3_mask};

    int m1_index = extract_mode(lin_index, 1, masks, bit_widths, 3, block);
    int m2_index = extract_mode(lin_index, 2, masks, bit_widths, 3, block);
    int m3_index = extract_mode(lin_index, 3, masks, bit_widths, 3, block);
    int idx_array[3] = {m1_index, m2_index,m3_index};
    int target_index = idx_array[mode - 1] - !active; //All inactive modes have target indices of -1

    int mode_num; //Thread represents the ith mode within with a given index the wavefront
    int total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index,value,total_modes,mode_num);
    __builtin_amdgcn_s_barrier();

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0; // ceil(rank/total_modes)
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    // Lookup tables for mode → nt_mode1, nt_mode2
    const int nt_mode1_list[3] = {2, 1, 1};
    const int nt_mode2_list[3] = {3, 3, 2};
    int nt_mode1 = nt_mode1_list[mode - 1];
    int nt_mode2 = nt_mode2_list[mode - 1];

    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};
    int fmat_sizes[3] = {rows * rank, cols * rank, depth * rank};
    int indices[3] = {m1_index, m2_index, m3_index};

    int peer_indices[64];  // max wavefront size
    int count = 0;

    unsigned long long mask = wavefront_mask;
    while(mask) {
        int peer_lane = __ffsll(mask) - 1;  // extract lowest set bit
        mask &= (mask - 1);
        peer_indices[count++] = peer_lane;
    }

    // Precompute all values from __shfl into temporary arrays
    int temp_nt1[64], temp_nt2[64];
    T temp_val[64];
    for(int i = 0; i < count; i++) {
        int peer = peer_indices[i];

        // Pick source values based on mode
        int src1 = (mode == 1 ? m2_index : (mode == 2 ? m1_index : m1_index));
        int src2 = (mode == 1 ? m3_index : (mode == 2 ? m3_index : m2_index));

        temp_nt1[i] = __shfl(src1, peer, wavefront_size) * rank;
        temp_nt2[i] = __shfl(src2, peer, wavefront_size) * rank;
        temp_val[i] = __shfl(value, peer, wavefront_size);
    }

    target_index += !active; //Set non target indices back to 0
    int target_base = target_index * rank;
    int output_idx;
    for (int j = s1; j < e1; ++j) {
        T sum = (T)0;
        for (int i = 0; i < count; ++i) {
            int idx2 = temp_nt1[i] + j;
            int idx3 = temp_nt2[i] + j;
            // optional bounds check removed for speed if you validated elsewhere
            sum += fmat_list[nt_mode1 - 1][idx2] * fmat_list[nt_mode2 - 1][idx3] * temp_val[i] * active;
        }
        // one atomic per j instead of one per (i,j)
        output_idx = target_base + j;
        if (output_idx < smem_size && output_idx >= 0) atomicAdd(&(smem[output_idx]), sum); 
        else if (output_idx < fmat_sizes[mode - 1]) atomicAdd(&(fmat_list[mode - 1][output_idx]), sum); 
    }

    __builtin_amdgcn_s_barrier();

    T val = (T)0;
    for (int i = block_idx; i < fmat_sizes[mode - 1]; i += blockDim.x) {
        val = smem[i];
        atomicAdd(&(fmat_list[mode - 1][i]), val);
    }
    
}

//======================================================================
// Host Wrapper: MTTKRP_BLCO
//======================================================================
// Orchestrates full GPU run of MTTKRP using BLCO tensor
// Steps:
// 1. Extract tensor dimensions, masks, factor matrices
// 2. Copy BLCO tensor blocks to GPU
// 3. Copy factor matrices to GPU (flattened format)
// 4. Decide kernel strategy:
//    - Non-hierarchical if target dimension < COMPUTE_UNITS
//    - Hierarchical otherwise
// 5. Launch kernel and time execution
// 6. Copy updated factor matrix back to host
// 7. Free GPU memory and return results
//======================================================================
template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_3D(int mode, const Blco_Tensor<T,S>& sparse_tensor, int iter = 1, std::vector<int> times = {0})
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    //Rows, cols, rank, and non zeros
    const std::vector<int> dims = sparse_tensor.get_dims();
    int rows = dims[0];
    int cols = dims[1];
    int depth = dims[2];
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t non_zeros = sparse_tensor.get_nnz();

    //Masks
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    uint64_t m1_mask = masks[0];
    uint64_t m2_mask = masks[1];
    uint64_t m3_mask = masks[2];

    //Fmats
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];

    //Number of blocks
    int num_blocks = blco_tensor.size();
    bool blocked = num_blocks > 1;

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_tensor;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3;

    // Allocate
    HIP_CHECK(hipMalloc(&d_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * rows * rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * cols * rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * depth * rank));

    // Copy host data to GPU
    blocks_to_gpu(d_tensor, blco_tensor, num_blocks);
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * rows * rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * cols * rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * depth * rank, hipMemcpyHostToDevice));

    int compute_units = get_compute_units();
    bool is_hierarchical;
    switch (mode) {
        case 1: is_hierarchical = (compute_units > rows); break;
        case 2: is_hierarchical = (compute_units > cols); break;
        case 3: is_hierarchical = (compute_units > depth); break;
    }

    bool collect_times = false;
    if(times.size() == 0) collect_times = true;
    std::pair<int,int> dimensions;

    if(!is_hierarchical){
        dimensions = determine_dimensions_no_smem(non_zeros); //Determine Dimensions

        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_3D_kernel_1<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, d_tensor, sparse_tensor.get_nnz(),
                m1_mask, m2_mask, m3_mask,
                d_fmat_1, d_fmat_2, d_fmat_3,
                rows, cols, depth,
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
    else{
        dimensions = determine_dimensions_no_smem(non_zeros);
        size_t shared_mem = get_max_shared_memory();
        int smem_entries = shared_mem / sizeof(T);

        for(int i = 0; i < iter; i++){
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_3D_kernel_2<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), shared_mem, 0,
                mode, d_tensor, sparse_tensor.get_nnz(),
                m1_mask, m2_mask, m3_mask,
                d_fmat_1, d_fmat_2, d_fmat_3,
                rows, cols, depth,
                num_blocks, rank, smem_entries
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

    // Determine the size and device pointer based on mode
    size_t vec_size = 0;
    T* d_ptr = nullptr;

    switch (mode) {
        case 1: vec_size = static_cast<size_t>(rows) * rank; d_ptr = d_fmat_1; break;
        case 2: vec_size = static_cast<size_t>(cols) * rank; d_ptr = d_fmat_2; break;
        case 3: vec_size = static_cast<size_t>(depth) * rank; d_ptr = d_fmat_3; break;
        default:
            throw std::runtime_error("Invalid mode in MTTKRP copy.");
    }

    // Allocate host-side vector
    std::vector<T> fmat_vec(vec_size);

    // Copy from device to host
    HIP_CHECK(hipMemcpy(fmat_vec.data(), d_ptr, vec_size * sizeof(T), hipMemcpyDeviceToHost));

    // Free device memory
    free_blocks_from_gpu(d_tensor,num_blocks);
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));

    return fmat_vec;
}