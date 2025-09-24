#include <hip/hip_runtime.h>
#include "blco_impl.h"

#define HIP_CHECK(cmd) do { hipError_t e = (cmd); if (e != hipSuccess) { \
    fprintf(stderr,"HIP ERROR %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

#define MAX_SHARED_MEM 65536

#define COMPUTE_UNITS 104

//Store Block array on GPU
// Helper: check hip errors concisely
inline void checkHIP(const hipError_t e, const char* where) {
    if (e != hipSuccess) {
        std::cerr << where << " failed: " << hipGetErrorString(e) << "\n";
    }
}

template<typename T>
void blocks_to_gpu(BLCO_BLOCK_GPU<T>*& d_block_arr,
                const std::vector<BLCO_BLOCK_CPU<T>>& tensor,
                int num_blocks)
{
    if (num_blocks <= 0) {
        d_block_arr = nullptr;
        return;
    }

    // Allocate device array of structs here (safer: single ownership location)
    hipError_t err = hipMalloc(&d_block_arr, static_cast<size_t>(num_blocks) * sizeof(BLCO_BLOCK_GPU<T>));
    checkHIP(err, "hipMalloc(d_block_arr)");
    if (err != hipSuccess) return;

    // host staging vector, zero-initialized to avoid stray padding values
    std::vector<BLCO_BLOCK_GPU<T>> h_arr_for_gpu(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        // ensure a fresh, zeroed struct (covers padding)
        h_arr_for_gpu[i].block = 0;
        h_arr_for_gpu[i].size = 0;
        h_arr_for_gpu[i].indexes = nullptr;
        h_arr_for_gpu[i].values = nullptr;

        int num_elements = tensor[i].size;
        h_arr_for_gpu[i].size = num_elements;
        h_arr_for_gpu[i].block = tensor[i].block;

        if (num_elements <= 0) {
            // explicitly set null pointers for empty blocks
            h_arr_for_gpu[i].values = nullptr;
            h_arr_for_gpu[i].indexes = nullptr;
            continue;
        }

        // Allocate device arrays for block internals
        T* d_values = nullptr;
        uint64_t* d_indexes = nullptr;

        err = hipMalloc(&d_values, static_cast<size_t>(num_elements) * sizeof(T));
        if (err != hipSuccess) {
            std::cerr << "hipMalloc(d_values) failed for block " << i << ": " << hipGetErrorString(err) << "\n";
            // decide: free previous allocations and abort; for now we bail
            // (caller could handle partial allocations)
            // free previously allocated block internals:
            for (int j = 0; j < i; ++j) {
                if (h_arr_for_gpu[j].values) hipFree(h_arr_for_gpu[j].values);
                if (h_arr_for_gpu[j].indexes) hipFree(h_arr_for_gpu[j].indexes);
            }
            hipFree(d_block_arr);
            d_block_arr = nullptr;
            return;
        }
        err = hipMemcpy(d_values, tensor[i].values.data(),
                        static_cast<size_t>(num_elements) * sizeof(T),
                        hipMemcpyHostToDevice);
        checkHIP(err, "hipMemcpy(values)");

        err = hipMalloc(&d_indexes, static_cast<size_t>(num_elements) * sizeof(uint64_t));
        if (err != hipSuccess) {
            std::cerr << "hipMalloc(d_indexes) failed for block " << i << ": " << hipGetErrorString(err) << "\n";
            hipFree(d_values);
            for (int j = 0; j < i; ++j) {
                if (h_arr_for_gpu[j].values) hipFree(h_arr_for_gpu[j].values);
                if (h_arr_for_gpu[j].indexes) hipFree(h_arr_for_gpu[j].indexes);
            }
            hipFree(d_block_arr);
            d_block_arr = nullptr;
            return;
        }
        err = hipMemcpy(d_indexes, tensor[i].indexes.data(),
                        static_cast<size_t>(num_elements) * sizeof(uint64_t),
                        hipMemcpyHostToDevice);
        checkHIP(err, "hipMemcpy(indexes)");

        // store device pointers into the host-side struct
        h_arr_for_gpu[i].values = d_values;
        h_arr_for_gpu[i].indexes = d_indexes;
    }

    // Copy the array of GPU structs to device
    err = hipMemcpy(d_block_arr, h_arr_for_gpu.data(),
                    static_cast<size_t>(num_blocks) * sizeof(BLCO_BLOCK_GPU<T>),
                    hipMemcpyHostToDevice);
    checkHIP(err, "hipMemcpy(h_arr_for_gpu -> d_block_arr)");
}

template<typename T>
void free_blocks_from_gpu(BLCO_BLOCK_GPU<T>* gpu_block_arr, int num_blocks)
{
    if (gpu_block_arr == nullptr || num_blocks <= 0) return;

    // Copy struct array from device to host so we can see the pointers
    std::unique_ptr<BLCO_BLOCK_GPU<T>[]> h_blocks(new BLCO_BLOCK_GPU<T>[num_blocks]);
    hipError_t err = hipMemcpy(h_blocks.get(), gpu_block_arr,
                                static_cast<size_t>(num_blocks) * sizeof(BLCO_BLOCK_GPU<T>),
                                hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed in free_blocks_from_gpu: " << hipGetErrorString(err) << "\n";
        // Attempt to get device error state
        hipDeviceSynchronize();
        checkHIP(hipGetLastError(), "hipGetLastError after failed memcpy");
        return;
    }

    for (int b = 0; b < num_blocks; ++b) {
        // Only free non-null device pointers
        if (h_blocks[b].values) {
            err = hipFree(h_blocks[b].values);
            if (err != hipSuccess) {
                std::cerr << "hipFree(values) failed for block " << b << ": " << hipGetErrorString(err) << "\n";
            }
        }
        if (h_blocks[b].indexes) {
            err = hipFree(h_blocks[b].indexes);
            if (err != hipSuccess) {
                std::cerr << "hipFree(indexes) failed for block " << b << ": " << hipGetErrorString(err) << "\n";
            }
        }
    }

    // Free the array of structs
    err = hipFree(gpu_block_arr);
    if (err != hipSuccess) {
        std::cerr << "hipFree(gpu_block_arr) failed: " << hipGetErrorString(err) << "\n";
    }
}

//Determine ideal number of blocks and threads per block assuming you don't use smem
std::pair<int,int> determine_dimensions_no_smem(uint64_t non_zeros, int wf_sz = 64)
{
    std::pair<int,int> dims;

    if(non_zeros <= 1024){
        dims.first = 1;
        int mul = non_zeros/wf_sz;
        if(mul * wf_sz < non_zeros) mul++;
        dims.second = wf_sz * mul;
        
    }
    else{
        dims.first = non_zeros/1024;
        if(non_zeros % 1024 != 0) dims.first++;
        dims.second = 1024;
    }

    return dims;
}

//Determine ideal number of blocks and threads per block assuming you use smem
template<typename T>
std::pair<int,int> determine_dimensions_smem(uint64_t non_zeros, int ui, int tensor_rank, int wf_sz = 64)
{
    std::pair<int,int> dims;

    if(non_zeros <= 1024){
        dims.first = 1;
        int mul = non_zeros/wf_sz;
        if(mul * wf_sz < non_zeros) mul++;
        dims.second = wf_sz * mul;
        
    }
    else{
        dims.first = non_zeros/1024;
        if(non_zeros % 1024 != 0) dims.first++;
        dims.second = 1024;
    }

    int wf_per_block = dims.second / wf_sz;
    bool enough_mem = ui * tensor_rank * wf_per_block * sizeof(T) <= MAX_SHARED_MEM;

    if(!enough_mem){
        int c1 =  ui * tensor_rank * sizeof(T);
        for(int i = 0; i < wf_per_block; i++){
            if(--wf_per_block * c1 < MAX_SHARED_MEM) break;
        }

        dims.second = wf_per_block * wf_sz;
        dims.first = non_zeros/dims.second;
        if(non_zeros % dims.second != 0) dims.first++;
    }

    return dims;
}

__device__ int ceiling_log2(int x) 
{
    if (x == 1) return 0;
    int res = 0;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}

//Find Block index
template<typename T>
__device__ int find_block_index(BLCO_BLOCK_GPU<T>* tensor, int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prefix_sum = 0;

    for(int i = 0; i < num_blocks; i++){
        if(idx < prefix_sum + tensor[i].size){
            return i;
        }
        prefix_sum += tensor[i].size;
    }

    return -1;
}

//Find linear index with with corresponding thread
template<typename T>
__device__ uint64_t extract_linear_index(BLCO_BLOCK_GPU<T>* tensor, int thread_id, int block_idx)
{
    int prefix_sum = 0;

    for(int i = 0; i < block_idx; i++){
        prefix_sum += tensor[i].size;
    }

    return tensor[block_idx].indexes[thread_id-prefix_sum];
}

//Find value with corresponding thread
template<typename T>
__device__ uint64_t extract_value(BLCO_BLOCK_GPU<T>* tensor, int block_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prefix_sum = 0;

    for(int i = 0; i < block_idx; i++){
        prefix_sum += tensor[i].size;
    }

    return tensor[block_idx].values[idx-prefix_sum];
}

//Find mode idx based on linear index
__device__ int extract_mode(uint64_t linear_idx, uint64_t mode_mask, int mode, int row_bits, int col_bits, int depth_bits, int block)
{
    uint64_t output = linear_idx & mode_mask;

    int shift;
    switch (mode) {
        case 1: shift = 0; break;
        case 2: shift = row_bits; break;
        case 3: shift = row_bits + col_bits; break;
    }

    output >>= shift;
    static_cast<int>(output);

    if(mode == 3 && row_bits + col_bits + depth_bits > 64){
        
        output |= block << (-1 * (row_bits + col_bits - 64));
    }

    return output;
}

//Bitonic sort based on target mode index
__device__ void sort_by_mode(int& mode, int& block, uint64_t& lin_idx, int warp_size = 64) 
{
    const int lane = threadIdx.x & (warp_size - 1);

    for (int k = 2; k <= warp_size; k <<= 1){
        // Outer loop: size of subsequence being merged
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner_mode = __shfl_xor(mode, j, warp_size);
            uint64_t partner_lin_idx = __shfl_xor(lin_idx, j, warp_size);
            int partner_block = __shfl_xor(block, j, warp_size);

            // Determine sort direction based on current subsequence
            bool dir = ((lane & (k)) == 0);

            // Perform compare–swap
            int new_mode;
            uint64_t new_idx;
            int new_block;
            if ((mode > partner_mode) == dir){
                new_mode = partner_mode;
                new_idx = partner_lin_idx;
                new_block = partner_block;
            }
            else{
                new_mode = mode;
                new_idx = lin_idx;
                new_block = block;
            }

            mode = new_mode;
            lin_idx = new_idx;
            block = new_block;
        }
    }
}

__device__ void group_mask_shfl_only(int target_val, unsigned long long &output_mask, int wf_size = 64) 
{
    int wavefront_idx = threadIdx.x % wf_size;   

    for (int k = 0; k < wf_size; ++k) {
        int val_k = __shfl(target_val, k, wf_size);
        if (val_k == target_val) {
            output_mask |= (1ULL << k);
        }
    }
}

template <typename T>
__device__ unsigned long long wavefront_group_reduce_1(int target_index, T value, int &total_modes, int &mode_num, int wavefront_size = 64)
{
    int wavefront_idx = threadIdx.x % wavefront_size;

    unsigned long long mask = 0ULL;
    group_mask_shfl_only(target_index,mask);

    unsigned long long mn_mask = (1ULL << wavefront_idx) - 1; 
    mn_mask &= mask;

    // 2. Find leader lane = first active thread in group
    int leader_idx = __ffsll(mask) - 1; // __ffsll returns 1-based index
    bool is_leader = (wavefront_idx == leader_idx);
    total_modes = __popcll(mask); // number of set bits
    mode_num = __popcll(mn_mask);

    return mask;
}

template <typename T>
__device__ unsigned long long wavefront_group_reduce_2(int target_index, T value, int &total_modes, int &mode_num, int &is_leader, int wavefront_size = 64)
{
    int wavefront_idx = threadIdx.x % wavefront_size;

    unsigned long long mask = 0ULL;
    group_mask_shfl_only(target_index,mask);

    unsigned long long mn_mask = (1ULL << wavefront_idx) - 1; 
    mn_mask &= mask;

    // 2. Find leader lane = first active thread in group
    int leader_idx = __ffsll(mask) - 1; // __ffsll returns 1-based index
    is_leader = (wavefront_idx == leader_idx);
    total_modes = __popcll(mask); // number of set bits
    mode_num = __popcll(mn_mask);

    return mask;
}

template<typename T>
__device__ void print_vector(T* vec, int size)
{
    for(int i = 0; i < size; i++){
        printf("%d ", vec[i]);
    }
    printf("\n");
}

__device__ int find_offset(int row, int* offsets, int size)
{
    for(int i = 0; i < size; i++){
        if(row == offsets[i]) return i;
    }
    return -1;
}

//MTTKRP function on GPU (Non Hierarchical)
template<typename T>
__global__ void mttkrp_kernel_1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask,
T* m1_fmat, T* m2_fmat, T* m3_fmat, int rows, int cols, int depth, int num_blocks, int rank, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;
    int wavefront_idx = threadIdx.x % 64;

    bool active = true;
    if (global_idx >= nnz) active = false; 
    int target_index = -1; //Default values if thread is inactive
    T value = 0; //Default if thread is inactive

    int m1_index; int m2_index; int m3_index;
    if(active){
        int bl_index = find_block_index(tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(tensor,global_idx,bl_index);
        value = extract_value(tensor,bl_index);
        int block = tensor[bl_index].block;
        int m1_bits = ceiling_log2(rows), m2_bits = ceiling_log2(cols), m3_bits = ceiling_log2(depth);
        m1_index = extract_mode(lin_index, m1_mask, 1, m1_bits, m2_bits, m3_bits, block);
        m2_index = extract_mode(lin_index, m2_mask, 2, m1_bits, m2_bits, m3_bits, block);
        m3_index = extract_mode(lin_index, m3_mask, 3, m1_bits, m2_bits, m3_bits, block);

        int idx_array[3] = {m1_index, m2_index,m3_index};
        target_index = idx_array[mode - 1];
    }

    int mode_num; //Thread represents the ith mode within with a given index the wavefront
    int total_modes;
    unsigned long long wavefront_mask = wavefront_group_reduce_1(target_index,value,total_modes,mode_num);
    __builtin_amdgcn_s_barrier();

    if(!active) return;

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0; // ceil(rank/total_modes)
    if (rank_offset == 0) {
        printf("Error calculating total modes\n");
        return;
    }
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
        
        if(count > 64){
            printf("Error with masking\n");
            break;
        }
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

    for(int i = 0; i < count; i++) {
        for(int j = s1; j < e1; j++){
            int index_1 = target_index * rank + j;
            int index_2 = temp_nt1[i] + j;
            int index_3 = temp_nt2[i] + j;

            if((index_1 < 0 || index_1 >= fmat_sizes[mode - 1] || index_2 < 0 || index_2 >= fmat_sizes[nt_mode1 - 1] 
                || index_3 < 0 || index_3 >= fmat_sizes[nt_mode2 - 1])){
                    printf("indexing error index_1: %d, index_2: %d, index_3: %d\n", index_1, index_2, index_3);
            }
    
            T mat_val = fmat_list[nt_mode1 - 1][index_2] *
                        fmat_list[nt_mode2 - 1][index_3] *
                        temp_val[i];
            
            atomicAdd(&(fmat_list[mode - 1][index_1]), mat_val);
        }
    }
            
}

//MTTKRP function on GPU (Hierarchical)
template<typename T>
__global__ void mttkrp_kernel_2(
    int mode,
    BLCO_BLOCK_GPU<T>* tensor,
    uint64_t nnz,
    uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask,
    T* m1_fmat, T* m2_fmat, T* m3_fmat,
    int rows, int cols, int depth,
    int num_blocks, int rank, int wf_indices,
    int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx  = threadIdx.x;
    int wf_per_block = blockDim.x / 64;
    int wavefront_num = threadIdx.x / 64; 
    int wavefront_idx = threadIdx.x % 64;

    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    int smem_size = wf_indices * rank * wf_per_block;
    int smem_offset = wf_indices * rank * wavefront_num;

    // initialize the partial_fmat
    for (int i = threadIdx.x; i < smem_size; i += blockDim.x) {
        smem[i] = 0;
    }

    bool active = (global_idx < nnz);
    int target_index = -1;
    T value = 0;

    int m1_index=-1, m2_index=-1, m3_index=-1;
    if (active) {
        int bl_index = find_block_index(tensor, num_blocks);
        uint64_t lin_index = extract_linear_index(tensor, global_idx, bl_index);
        value = extract_value(tensor, bl_index);
        int block = tensor[bl_index].block;
        int m1_bits = ceiling_log2(rows),
        m2_bits = ceiling_log2(cols),
        m3_bits = ceiling_log2(depth);
        m1_index = extract_mode(lin_index, m1_mask, 1, m1_bits, m2_bits, m3_bits, block);
        m2_index = extract_mode(lin_index, m2_mask, 2, m1_bits, m2_bits, m3_bits, block);
        m3_index = extract_mode(lin_index, m3_mask, 3, m1_bits, m2_bits, m3_bits, block);

        int idx_array[3] = {m1_index, m2_index, m3_index};
        target_index = idx_array[mode - 1];
    }

    int mode_num, total_modes;
    int is_leader;
    unsigned long long wavefront_mask = wavefront_group_reduce_2(target_index, value, 
                                        total_modes, mode_num, is_leader);
    __builtin_amdgcn_s_barrier();

    unsigned long long leader_mask = __ballot(is_leader);  // Keep only lanes < my lane
    unsigned long long earlier = leader_mask & ((1ULL << wavefront_idx) - 1ULL); // Count leaders before me
    int idx_num = __popcll(earlier);
    __builtin_amdgcn_s_barrier();

    int leader_idx = __ffsll(wavefront_mask) - 1;
    idx_num = __shfl(idx_num, leader_idx, wavefront_size);

    int rank_offset = (total_modes > 0) ? (rank + total_modes - 1) / total_modes : 0;
    if (rank_offset == 0) {
        printf("[thread %d] Error calculating total_modes=%d\n", global_idx, total_modes);
        return;
    }

    int s1 = mode_num * rank_offset * active;
    int e1 = min((mode_num + 1) * rank_offset * active, rank * active);

    const int nt_mode1_list[3] = {2, 1, 1};
    const int nt_mode2_list[3] = {3, 3, 2};
    int nt_mode1 = nt_mode1_list[mode - 1];
    int nt_mode2 = nt_mode2_list[mode - 1];

    T* fmat_list[3] = {m1_fmat, m2_fmat, m3_fmat};
    int fmat_sizes[3] = {rows * rank, cols * rank, depth * rank};

    // Collect peers
    int peer_indices[64];
    int count = 0;
    unsigned long long mask = wavefront_mask;
    while (mask) {
        int peer_lane = __ffsll(mask) - 1;
        mask &= (mask - 1);
        peer_indices[count++] = peer_lane;
    }

    // Precompute shuffle values
    int temp_nt1[64], temp_nt2[64];
    T temp_val[64];
    for (int i = 0; i < count; i++) {
        int peer = peer_indices[i];
        int src1 = (mode == 1 ? m2_index : (mode == 2 ? m1_index : m1_index));
        int src2 = (mode == 1 ? m3_index : (mode == 2 ? m3_index : m2_index));

        temp_nt1[i] = __shfl(src1, peer, wavefront_size);
        temp_nt2[i] = __shfl(src2, peer, wavefront_size);
        temp_val[i] = __shfl(value, peer, wavefront_size);
    }

    int pfm_offset = smem_offset + idx_num * rank; //Offset into partial fmat contained in shared memory
    for (int i = 0; i < count; i++) {
        int fm1_offset = temp_nt1[i] * rank; //Offset into first non target fmat
        int fm2_offset = temp_nt2[i] * rank; //Offset into second non target fmat
        for (int j = s1; j < e1; j++) {
            int index_1 = pfm_offset + j;
            int index_2 = fm1_offset + j;
            int index_3 = fm2_offset + j;

            if ((index_1 < 0 || index_1 >= smem_size ||
                index_2 < 0 || index_2 >= fmat_sizes[nt_mode1 - 1] ||
                index_3 < 0 || index_3 >= fmat_sizes[nt_mode2 - 1]) && global_idx % 10 == 0) {
                printf("[thread %d] INDEX ERROR: i=%d j=%d index_1=%d index_2=%d index_3=%d\n",
                    global_idx, i, j, index_1, index_2, index_3);
            }

            T mat_val = fmat_list[nt_mode1 - 1][index_2] *
                        fmat_list[nt_mode2 - 1][index_3] *
                        temp_val[i];

            smem[index_1] += mat_val;
        }
    }

    __builtin_amdgcn_s_barrier();

    if (is_leader) {
        int row_offset = smem_offset + idx_num * rank;
        int fmat_offset = target_index * rank;
        for (int i = 0; i < rank * active; i++) {
            T add_val = smem[row_offset + i];
            atomicAdd(&(fmat_list[mode - 1][fmat_offset + i]), add_val);
        }
    }
}


//Paralell MTTKRP on GPU using blco tensor
template<typename T, typename S>
T** MTTKRP_BLCO(int mode, const BLCO_Tensor_3D<T,S>& sparse_tensor)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    //Rows, cols, rank, and non zeros
    const std::vector<int> dims = sparse_tensor.get_dims();
    int rows = dims[0];
    int cols = dims[1];
    int depth = dims[2];
    const int rank = sparse_tensor.get_rank();
    uint64_t non_zeros = sparse_tensor.get_nnz();

    //Masks
    const std::vector<uint64_t> masks = sparse_tensor.get_blco_masks();
    uint64_t m1_mask = masks[0];
    uint64_t m2_mask = masks[1];
    uint64_t m3_mask = masks[2];

    //Fmats
    std::vector<T**> fmats = sparse_tensor.get_fmats();
    T** mode_1_fmat = fmats[0];
    T** mode_2_fmat = fmats[1];
    T** mode_3_fmat = fmats[2];

    //Number of blocks
    int num_blocks = blco_tensor.size();
    bool blocked = num_blocks > 1;

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_tensor;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3;

    // Allocate
    hipMalloc(&d_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks);
    hipMalloc(&d_fmat_1, sizeof(T) * rows * rank);
    hipMalloc(&d_fmat_2, sizeof(T) * cols * rank);
    hipMalloc(&d_fmat_3, sizeof(T) * depth * rank);

    // Vectorize factor matrices
    T* h_m1_vector = vectorize_matrix(mode_1_fmat,rows,rank);
    T* h_m2_vector = vectorize_matrix(mode_2_fmat,cols,rank);
    T* h_m3_vector = vectorize_matrix(mode_3_fmat,depth,rank);

    // Copy host data to GPU
    blocks_to_gpu(d_tensor, blco_tensor, num_blocks);
    hipMemcpy(d_fmat_1, h_m1_vector, sizeof(T) * rows * rank, hipMemcpyHostToDevice);
    hipMemcpy(d_fmat_2, h_m2_vector, sizeof(T) * cols * rank, hipMemcpyHostToDevice);
    hipMemcpy(d_fmat_3, h_m3_vector, sizeof(T) * depth * rank, hipMemcpyHostToDevice);

    int is_hierarchical;
    switch (mode) {
        case 1: is_hierarchical = (COMPUTE_UNITS > rows); break;
        case 2: is_hierarchical = (COMPUTE_UNITS > cols); break;
        case 3: is_hierarchical = (COMPUTE_UNITS > depth); break;
    }


    if(!is_hierarchical){
        std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros); //Determine Dimensions

        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            mttkrp_kernel_1<T>,  // specify template args
            dim3(dimensions.first), dim3(dimensions.second), 0, 0,
            mode, d_tensor, sparse_tensor.get_nnz(),
            m1_mask, m2_mask, m3_mask,
            d_fmat_1, d_fmat_2, d_fmat_3,
            rows, cols, depth,
            num_blocks, rank
        );        
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Kernel Duration: " << duration << " ms\n";
    }
    else{
        //Specify number of unique indices per wavefront
        int ui_per_wf;
        if(blocked) ui_per_wf = sparse_tensor.determine_indexes_per_wavefront_128_bit(mode);
        else ui_per_wf = sparse_tensor.determine_indexes_per_wavefront_64_bit(mode);

        //Specify shared memory size
        std::pair<int,int> dimensions = determine_dimensions_smem<T>(non_zeros,ui_per_wf,rank);
        int wavefronts_per_block = dimensions.second/64;  //Maximum number of unique target indices per wavefront
        size_t shared_mem = ui_per_wf * rank * wavefronts_per_block * sizeof(T); 

        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(
            mttkrp_kernel_2<T>,  // specify template args
            dim3(dimensions.first), dim3(dimensions.second), shared_mem, 0,
            mode, d_tensor, sparse_tensor.get_nnz(),
            m1_mask, m2_mask, m3_mask,
            d_fmat_1, d_fmat_2, d_fmat_3,
            rows, cols, depth,
            num_blocks, rank, ui_per_wf
        );        
        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Kernel Duration: " << duration << " ms\n";
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
    hipError_t err = hipMemcpy(fmat_vec.data(), d_ptr, vec_size * sizeof(T), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "hipMemcpy failed: " << hipGetErrorString(err) << std::endl;
        hipDeviceSynchronize();
    }

    sparse_tensor.copy_vector_to_fmat(fmat_vec.data(), mode);

    // Free device memory
    free_blocks_from_gpu(d_tensor,num_blocks);
    hipFree(d_fmat_1);
    hipFree(d_fmat_2);
    hipFree(d_fmat_3);

    // Free host-side buffers
    delete[] h_m1_vector;
    delete[] h_m2_vector;
    delete[] h_m3_vector;

    return fmats[mode - 1];
}