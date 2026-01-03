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
__global__ void mttkrp_3D_kernel_1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, 
    T* m1_fmat, T* m2_fmat, T* m3_fmat, 
    int d1, int d2, int d3, int num_blocks, int rank, int wavefront_size = 64)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (global_idx < nnz);
    
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    uint64_t lin_index = entry.index;
    T value = entry.value;

    // Pre-calculate bit widths to avoid repeated logs
    int bit_widths[3] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3)};
    uint64_t masks[3] = {m1_mask, m2_mask, m3_mask};

    // Extract coordinates
    int coords[3];
    #pragma unroll
    for(int i = 0; i < 3; i++) {
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 3, block);
    }

    // Safety: If inactive, point to index 0 to avoid OOB during shuffles, 
    // but we will mask the atomicAdd later.
    int target_index = active ? coords[mode - 1] : -1;

    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, value, total_modes, mode_num);
    
    // Calculate which part of the rank-vector this thread is responsible for
    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    // Identify non-target modes (nt1, nt2)
    const int nt1_list[3] = {2, 1, 1};
    const int nt2_list[3] = {3, 3, 2};
    int nt1 = nt1_list[mode - 1];
    int nt2 = nt2_list[mode - 1];

    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};

    // We still need to know which lanes are in our group, but we only store the lane IDs.
    // 64 ints = 256 bytes. This is much better than storing 4-5 arrays of 64 elements.
    int peer_lanes[64];
    int count = 0;
    unsigned long long mask_copy = wavefront_mask;
    while(mask_copy) {
        int peer = __ffsll(mask_copy) - 1;
        mask_copy &= (mask_copy - 1);
        peer_lanes[count++] = peer;
    }

    target_index += !active;
    int target_base = target_index * rank;

    // Loop over the rank (dimension of the factor matrix)
    for (int j = 0; j < rank; ++j) {
        T sum = (T)0;
        // Only threads assigned to this rank segment actually contribute to the sum
        bool responsible_for_j = (j >= s1) && (j < e1);
        
        // Recompute/Shuffle values from peers instead of using large static arrays
        for (int i = 0; i < count; ++i) {
            int peer = peer_lanes[i];
            
            // Shared values fetched via shuffle (register-to-register)
            T peer_val = __shfl(value, peer, wavefront_size);
            int peer_idx1 = __shfl(coords[nt1-1], peer, wavefront_size);
            int peer_idx2 = __shfl(coords[nt2-1], peer, wavefront_size);

            // Compute contribution
            // Sum = Val * MatrixB[coordB][j] * MatrixC[coordC][j]
            sum += peer_val * fmat_list[nt1-1][peer_idx1 * rank + j] * fmat_list[nt2-1][peer_idx2 * rank + j] * responsible_for_j;
        }

        // Final atomic update: Only if thread is active AND responsible for this part of the rank
        if (active && responsible_for_j) {
            atomicAdd(&fmat_list[mode-1][target_base + j], sum);
        }
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

    // 1. Initialize Shared Memory
    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    for (int i = threadIdx.x; i < smem_size; i += blockDim.x) {
        smem[i] = (T)0;
    }
    __syncthreads(); // Ensure SMEM is zeroed before threads proceed

    // 2. Data Extraction
    bool active = (global_idx < nnz);
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    
    // Bit widths and masks
    int bit_widths[3] = {ceiling_log2(rows), ceiling_log2(cols), ceiling_log2(depth)};
    uint64_t masks[3] = {m1_mask, m2_mask, m3_mask};

    // Extract coordinates into local registers
    int m1_idx = extract_mode(entry.index, 1, masks, bit_widths, 3, block);
    int m2_idx = extract_mode(entry.index, 2, masks, bit_widths, 3, block);
    int m3_idx = extract_mode(entry.index, 3, masks, bit_widths, 3, block);
    
    int coords[3] = {m1_idx, m2_idx, m3_idx};
    
    // Safety: point inactive threads to 0 to avoid -1 indexing
    int target_index = active ? coords[mode - 1] : -1;

    // 3. Wavefront Grouping
    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, entry.value, total_modes, mode_num);

    // Identify non-target modes
    const int nt1_list[3] = {2, 1, 1};
    const int nt2_list[3] = {3, 3, 2};
    int nt1 = nt1_list[mode - 1];
    int nt2 = nt2_list[mode - 1];
    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};

    // Calculate rank-segment for this thread
    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    // Pre-calculate which lanes are in our group (small 64-int array is okay, usually stays in registers)
    int peer_lanes[64];
    int count = 0;
    unsigned long long mask_copy = wavefront_mask;
    while(mask_copy) {
        int peer = __ffsll(mask_copy) - 1;
        mask_copy &= (mask_copy - 1);
        peer_lanes[count++] = peer;
    }

    // 4. Computation Loop
    int target_base = target_index * rank;

    for (int j = 0; j < rank; ++j) {
        T sum = (T)0;
        bool responsible_for_j = (j >= s1 && j < e1);

        for (int i = 0; i < count; ++i) {
            int peer = peer_lanes[i];
            
            // SHUFFLE: Fetch values from peers on the fly
            T p_val = __shfl(entry.value, peer, wavefront_size);
            int p_idx1 = __shfl(coords[nt1-1], peer, wavefront_size);
            int p_idx2 = __shfl(coords[nt2-1], peer, wavefront_size);

            sum += p_val * fmat_list[nt1-1][p_idx1 * rank + j] * fmat_list[nt2-1][p_idx2 * rank + j] * responsible_for_j;
        }

        // 5. Update Shared Memory (Cache) or Global Memory
        if (active && responsible_for_j) {
            int out_idx = target_base + j;
            if (out_idx < smem_size) {
                atomicAdd(&smem[out_idx], sum);
            } else {
                atomicAdd(&fmat_list[mode - 1][out_idx], sum);
            }
        }
    }

    // 6. Final Synchronization and Flush SMEM to Global Factor Matrix
    __syncthreads();

    int fmat_total_size = (mode == 1 ? rows : (mode == 2 ? cols : depth)) * rank;
    for (int i = block_idx; i < smem_size && i < fmat_total_size; i += blockDim.x) {
        T final_val = smem[i];
        if (final_val != (T)0) {
            atomicAdd(&fmat_list[mode - 1][i], final_val);
        }
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
std::vector<T> MTTKRP_BLCO_3D(int mode, const Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    //Rows, cols, rank, and non zeros
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

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_3D_kernel_1<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, sparse_tensor.get_nnz(),
                m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
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

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(
                mttkrp_3D_kernel_2<T>,  // specify template args
                dim3(dimensions.first), dim3(dimensions.second), shared_mem, 0,
                mode, res.d_tensor, sparse_tensor.get_nnz(),
                m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
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

    size_t out_size = dims[mode-1] * rank;
    std::vector<T> result(out_size);
    HIP_CHECK(hipMemcpy(result.data(), res.d_fmats[mode-1], sizeof(T) * out_size, hipMemcpyDeviceToHost));

    deallocate_mttkrp_resources(res, num_blocks);

    return result;
}