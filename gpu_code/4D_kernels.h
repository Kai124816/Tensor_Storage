#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

//--------------------Blocked Kernels--------------------

//======================================================================
// Kernel 1: Non-Hierarchical 4D MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_4D_kernel_1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, 
T* m4_fmat, int d1, int d2, int d3, int d4, int num_blocks, int rank, int wavefront_size = 64)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (global_idx < nnz);
    
    int block = 0;
    BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
    uint64_t lin_index = entry.index;
    T value = entry.value;

    int bit_widths[4] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4)};
    uint64_t masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};

    int coords[4];
    for(int i = 0; i < 4; i++) {
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 4, block);
    }

    int target_index = coords[mode - 1] - !active;

    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, value, total_modes, mode_num);

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    // Non-target modes for 4D: if target is mode n, multiply the other 3
    const int nt1_list[4] = {2, 1, 1, 1};
    const int nt2_list[4] = {3, 3, 2, 2};
    const int nt3_list[4] = {4, 4, 4, 3};
    
    int nt1 = nt1_list[mode - 1], nt2 = nt2_list[mode - 1], nt3 = nt3_list[mode - 1];
    T* fmat_list[4] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat};

    int peer_lanes[64];
    int count = 0;
    unsigned long long mask_copy = wavefront_mask;
    while(mask_copy) {
        int peer = __ffsll(mask_copy) - 1;
        mask_copy &= (mask_copy - 1);
        peer_lanes[count++] = peer;
    }

    // Process rank segments
    bool acc;
    target_index += !active; //Set non target indices back to 0
    int target_base = (coords[mode - 1]) * rank;
    for (int j = 0; j < rank; ++j) {
        T sum = (T)0;
        acc = (j >= s1) && (j < e1);
        for (int i = 0; i < count; ++i) {
            int peer = peer_lanes[i];
            T val = __shfl(value, peer, wavefront_size);
            int idx1 = __shfl(coords[nt1-1], peer, wavefront_size) * rank + j;
            int idx2 = __shfl(coords[nt2-1], peer, wavefront_size) * rank + j;
            int idx3 = __shfl(coords[nt3-1], peer, wavefront_size) * rank + j;
            sum += fmat_list[nt1-1][idx1] * fmat_list[nt2-1][idx2] * fmat_list[nt3-1][idx3] * val * acc;
        }
        if(active) atomicAdd(&fmat_list[mode-1][target_base + j], sum);
    }
}

//======================================================================
// Kernel 2: Hierarchical 4D MTTKRP (Shared Memory)
//======================================================================
template<typename T>
__global__ void mttkrp_4D_kernel_2(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, 
T* m4_fmat, int d1, int d2, int d3, int d4, int num_blocks, int rank, int smem_size, int wavefront_size = 64)
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
    T val = entry.value;
    
    int bit_widths[4] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4)};
    uint64_t masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};
    int coords[4];
    for(int i = 0; i < 4; i++) coords[i] = extract_mode(entry.index, i + 1, masks, bit_widths, 4, block);

    int target_index = coords[mode-1] - !active;
    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, entry.value, total_modes, mode_num);

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    const int nt_modes[3][4] = {{2,1,1,1}, {3,3,2,2}, {4,4,4,3}};
    T* fmat_list[4] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat};
    int fmat_sizes[4] = {d1*rank, d2*rank, d3*rank, d4*rank};

    unsigned long long mask_copy = wavefront_mask;
    int peer_lanes[64], count = 0;
    while(mask_copy) { peer_lanes[count++] = __ffsll(mask_copy) - 1; mask_copy &= (mask_copy - 1); }

    int target_base = coords[mode-1] * rank;
    bool acc;
    target_index += !active; //Set non target indices back to 0
    int nt_1 = coords[nt_modes[0][mode-1]-1], nt_2 = coords[nt_modes[1][mode-1]-1], nt_3 = coords[nt_modes[2][mode-1]-1];
    for (int j = 0; j < rank; ++j) {
        T sum = 0;
        acc = (j >= s1) && (j < e1);
        for (int i = 0; i < count; ++i) {
            int p = peer_lanes[i];
            int idx1 = __shfl(nt_1, p, wavefront_size) * rank + j;
            int idx2 = __shfl(nt_2, p, wavefront_size) * rank + j;
            int idx3 = __shfl(nt_3, p, wavefront_size) * rank + j;
            sum += fmat_list[nt_modes[0][mode-1]-1][idx1] * fmat_list[nt_modes[1][mode-1]-1][idx2] 
            * fmat_list[nt_modes[2][mode-1]-1][idx3] * __shfl(val, p, wavefront_size) * acc;
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
// Host Wrapper: MTTKRP_BLCO_4D
//======================================================================
template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_4D(int mode, const Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_cpu = sparse_tensor.get_blco();
    const std::vector<int> dims = sparse_tensor.get_dims();
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t nnz = sparse_tensor.get_nnz();
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    int num_blocks = sparse_tensor.get_num_blocks();

    // Allocate device pointers
    MTTKRP_Device_Resources<T> res = allocate_mttkrp_resources(sparse_tensor);

    bool is_hierarchical = (get_compute_units() > dims[mode - 1]);
    std::pair<int,int> grid = determine_dimensions_no_smem(nnz);
    size_t shared_mem = get_max_shared_memory();

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

            hipLaunchKernelGGL(mttkrp_4D_kernel_1<T>, dim3(grid.first), dim3(grid.second), 0, 0,
                mode, res.d_tensor, nnz, masks[0], masks[1], masks[2], masks[3],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2], res.d_fmats[3],
                dims[0], dims[1], dims[2], dims[3], num_blocks, rank);
            
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

            hipLaunchKernelGGL(mttkrp_4D_kernel_2<T>, dim3(grid.first), dim3(grid.second), shared_mem, 0,
                mode, res.d_tensor, nnz, masks[0], masks[1], masks[2], masks[3],
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2], res.d_fmats[3],
                dims[0], dims[1], dims[2], dims[3], num_blocks, rank, (int)(shared_mem/sizeof(T)));
            
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

//--------------------CSR Kernels--------------------

//======================================================================
// Kernel 1: Non-Hierarchical 4D MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_4D_kernel_csr_1(int mode, BLCO_ENTRY<T>* entries, uint64_t* block_ptr, int block_offset, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, 
T* m4_fmat, int d1, int d2, int d3, int d4, int num_blocks, int rank, int wavefront_size = 64)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (global_idx < nnz);
    int safe_idx = active ? global_idx : 0;
    
    uint64_t lin_index = entries[safe_idx].index;
    T value = entries[safe_idx].value;
    int block = find_block_csr(block_ptr, block_offset, safe_idx, num_blocks);

    int bit_widths[4] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4)};
    uint64_t masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};
    int dims[4] = {d1, d2, d3, d4};

    int coords[4];
    for(int i = 0; i < 4; i++) {
        coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 4, block);
    }

    int target_index = active ? coords[mode - 1] : -1;

    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, value, total_modes, mode_num);

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    const int nt1_list[4] = {2, 1, 1, 1};
    const int nt2_list[4] = {3, 3, 2, 2};
    const int nt3_list[4] = {4, 4, 4, 3};
    
    int nt1 = nt1_list[mode - 1], nt2 = nt2_list[mode - 1], nt3 = nt3_list[mode - 1];
    T* fmat_list[4] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat};
    int nt_dims[3] = {dims[nt1-1], dims[nt2-1], dims[nt3-1]};

    int peer_lanes[64];
    int count = 0;
    unsigned long long mask_copy = wavefront_mask;
    while(mask_copy) {
        int peer = __ffsll(mask_copy) - 1;
        mask_copy &= (mask_copy - 1);
        peer_lanes[count++] = peer;
    }

    int target_base = coords[mode - 1] * rank;

    for (int j = 0; j < rank; ++j) {
        T sum = (T)0;
        bool acc = (j >= s1) && (j < e1);
        for (int i = 0; i < count; ++i) {
            int peer = peer_lanes[i];
            
            T val = __shfl(value, peer, wavefront_size);
            int c1 = __shfl(coords[nt1-1], peer, wavefront_size);
            int c2 = __shfl(coords[nt2-1], peer, wavefront_size);
            int c3 = __shfl(coords[nt3-1], peer, wavefront_size);

            int idx1 = c1 * rank + j;
            int idx2 = c2 * rank + j;
            int idx3 = c3 * rank + j;

            sum += fmat_list[nt1-1][idx1] * fmat_list[nt2-1][idx2] * fmat_list[nt3-1][idx3] * val * acc;
        }

        if(active && acc) {
            int final_idx = target_base + j;
            atomicAdd(&fmat_list[mode-1][final_idx], sum);
        }
    }
}

//======================================================================
// Kernel 2: Hierarchical 4D MTTKRP (Shared Memory)
//======================================================================
template<typename T>
__global__ void mttkrp_4D_kernel_csr_2(int mode, BLCO_ENTRY<T>* entries, uint64_t* block_ptr, int block_offset, uint64_t nnz, 
uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask, uint64_t m4_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, 
T* m4_fmat, int d1, int d2, int d3, int d4, int num_blocks, int rank, int smem_size, int wavefront_size = 64)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    for (int i = block_idx; i < smem_size; i += blockDim.x) smem[i] = 0;
    __syncthreads();

    bool active = (global_idx < nnz);
    uint64_t lin_index = entries[nnz * active].index * active;
    T value = entries[nnz * active].value * active;
    int block = find_block_csr(block_ptr, block_offset, global_idx, num_blocks) * active;

    
    int bit_widths[4] = {ceiling_log2(d1), ceiling_log2(d2), ceiling_log2(d3), ceiling_log2(d4)};
    uint64_t masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};
    int coords[4];
    for(int i = 0; i < 4; i++) coords[i] = extract_mode(lin_index, i + 1, masks, bit_widths, 4, block);

    int target_index = coords[mode-1] - !active;
    int mode_num, total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index, value, total_modes, mode_num);

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    int s1 = mode_num * rank_offset;
    int e1 = min((mode_num + 1) * rank_offset, rank);

    const int nt_modes[3][4] = {{2,1,1,1}, {3,3,2,2}, {4,4,4,3}};
    T* fmat_list[4] = {m1_fmat, m2_fmat, m3_fmat, m4_fmat};
    int fmat_sizes[4] = {d1*rank, d2*rank, d3*rank, d4*rank};

    unsigned long long mask_copy = wavefront_mask;
    int peer_lanes[64], count = 0;
    while(mask_copy) { peer_lanes[count++] = __ffsll(mask_copy) - 1; mask_copy &= (mask_copy - 1); }

    int target_base = coords[mode-1] * rank;
    bool acc;
    target_index += !active; //Set non target indices back to 0
    int nt_1 = coords[nt_modes[0][mode-1]-1], nt_2 = coords[nt_modes[1][mode-1]-1], nt_3 = coords[nt_modes[2][mode-1]-1];
    for (int j = 0; j < rank; ++j) {
        T sum = 0;
        acc = (j >= s1) && (j < e1);
        for (int i = 0; i < count; ++i) {
            int p = peer_lanes[i];
            int idx1 = __shfl(nt_1, p, wavefront_size) * rank + j;
            int idx2 = __shfl(nt_2, p, wavefront_size) * rank + j;
            int idx3 = __shfl(nt_3, p, wavefront_size) * rank + j;
            sum += fmat_list[nt_modes[0][mode-1]-1][idx1] * fmat_list[nt_modes[1][mode-1]-1][idx2] 
            * fmat_list[nt_modes[2][mode-1]-1][idx3] * __shfl(value, p, wavefront_size) * acc;
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
// Host Wrapper
//======================================================================
template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_CSR_4D(int mode, const Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_cpu = sparse_tensor.get_blco();
    const std::vector<int> dims = sparse_tensor.get_dims();
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t nnz = sparse_tensor.get_nnz();
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    int num_blocks = blco_cpu[blco_cpu.size() - 1].block;
    const std::vector<T*> h_fmats = sparse_tensor.get_fmats();

    // Allocate Entry Array
    BLCO_CSR_GPU<T> blco_device;
    allocate_and_copy_BLCO_to_GPU(sparse_tensor, blco_device);

    //Allocate Fmats
    std::vector<T*> d_fmats;
    d_fmats.resize(dims.size());
    for (int i = 0; i < dims.size(); ++i) {
        size_t fmat_size = static_cast<size_t>(dims[i]) * rank;
        
        HIP_CHECK(hipMalloc(&(d_fmats[i]), sizeof(T) * fmat_size));
        
        // Copy the initial factor matrix data from host to device
        HIP_CHECK(hipMemcpy(d_fmats[i], h_fmats[i], 
                            sizeof(T) * fmat_size, hipMemcpyHostToDevice));
    }

    bool is_hierarchical = (get_compute_units() > dims[mode - 1]);
    std::pair<int,int> grid = determine_dimensions_no_smem(nnz);
    size_t shared_mem = get_max_shared_memory();

    bool collect_times = false;
    if(times.size() == 0) collect_times = true;

    if(!is_hierarchical){
        for(int i = 0; i < iter; i++) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(mttkrp_4D_kernel_csr_1<T>, dim3(grid.first), dim3(grid.second), 0, 0,
                mode, blco_device.tensor_entries, blco_device.block_ptr, blco_device.offset, nnz, masks[0], masks[1], masks[2], masks[3],
                d_fmats[0], d_fmats[1], d_fmats[2], d_fmats[3],
                dims[0], dims[1], dims[2], dims[3], num_blocks, rank);

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
        for(int i = 0; i < iter; i++) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(mttkrp_4D_kernel_csr_2<T>, dim3(grid.first), dim3(grid.second), shared_mem, 0,
                mode, blco_device.tensor_entries, blco_device.block_ptr, blco_device.offset, nnz, masks[0], masks[1], masks[2], masks[3],
                d_fmats[0], d_fmats[1], d_fmats[2], d_fmats[3],
                dims[0], dims[1], dims[2], dims[3], num_blocks, rank, (int)(shared_mem/sizeof(T)));

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
    HIP_CHECK(hipMemcpy(result.data(), d_fmats[mode-1], sizeof(T) * out_size, hipMemcpyDeviceToHost));

    for(int i = 0; i < dims.size(); i++){
        HIP_CHECK(hipFree(d_fmats[i]));
    }
    HIP_CHECK(hipFree(blco_device.tensor_entries));
    HIP_CHECK(hipFree(blco_device.block_ptr));

    return result;
}