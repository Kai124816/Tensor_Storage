#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

//----------------------------Version 1---------------------------------

//======================================================================
// Version 1 Kernel: Non-Hierarchical MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_kernel_1_v1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask,
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
        int block = 0;
        BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
        uint64_t lin_index = entry.index;
        value = entry.value;
        int m1_bits = ceiling_log2(rows), m2_bits = ceiling_log2(cols), m3_bits = ceiling_log2(depth);
        int bit_widths[3] = {m1_bits, m2_bits, m3_bits};
        uint64_t masks[3] = {m1_mask, m2_mask, m3_mask};
        m1_index = extract_mode(lin_index, 1, masks, bit_widths, 3, block);
        m2_index = extract_mode(lin_index, 2, masks, bit_widths, 3, block);
        m3_index = extract_mode(lin_index, 3, masks, bit_widths, 3, block);

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

//======================================================================
// Version 1 Kernel: Hierarchical MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_kernel_2_v1(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, uint64_t m1_mask, uint64_t m2_mask, 
uint64_t m3_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, int rows, int cols, int depth, int num_blocks, int rank, int wf_indices,
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
        int block = 0;
        BLCO_ENTRY<T> entry = extract_entry(tensor, num_blocks, block);
        uint64_t lin_index = entry.index;
        value = entry.value;
        int m1_bits = ceiling_log2(rows),
        m2_bits = ceiling_log2(cols),
        m3_bits = ceiling_log2(depth);
        int bit_widths[3] = {m1_bits, m2_bits, m3_bits};
        uint64_t masks[3] = {m1_mask, m2_mask, m3_mask};
        m1_index = extract_mode(lin_index, 1, masks, bit_widths, 3, block);
        m2_index = extract_mode(lin_index, 2, masks, bit_widths, 3, block);
        m3_index = extract_mode(lin_index, 3, masks, bit_widths, 3, block);

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


//======================================================================
// Version 1: Host Wrapper
//======================================================================
template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_v1(int mode, const Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    const std::vector<int> dims = sparse_tensor.get_dims();
    int rows = dims[0], cols = dims[1], depth = dims[2];
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t non_zeros = sparse_tensor.get_nnz();
    int num_blocks = sparse_tensor.get_num_blocks();

    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    uint64_t m1_mask = masks[0], m2_mask = masks[1], m3_mask = masks[2];

    MTTKRP_Device_Resources<T> res = allocate_mttkrp_resources(sparse_tensor);

    int compute_units = get_compute_units();
    bool is_hierarchical = false;
    if (mode == 1) is_hierarchical = (compute_units > rows);
    else if (mode == 2) is_hierarchical = (compute_units > cols);
    else is_hierarchical = (compute_units > depth);

    bool collect_times = (times.size() == 0);
    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);

    for(int i = 0; i < iter; i++){
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start, 0));

        if(!is_hierarchical){
            hipLaunchKernelGGL(
                mttkrp_kernel_1_v1<T>, 
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, non_zeros, m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
                rows, cols, depth, num_blocks, rank
            );
        } else {
            int ui_per_wf = sparse_tensor.determine_indexes_per_wavefront(mode);
            int wf_per_block = dimensions.second / 64;
            // smem_size in entries: wf_indices * rank * wf_per_block
            size_t smem_bytes = ui_per_wf * rank * wf_per_block * sizeof(T);
            
            hipLaunchKernelGGL(
                mttkrp_kernel_2_v1<T>, 
                dim3(dimensions.first), dim3(dimensions.second), smem_bytes, 0,
                mode, res.d_tensor, non_zeros, m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
                rows, cols, depth, num_blocks, rank, ui_per_wf
            );
        }

        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        if(collect_times) times.push_back(ms);
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    size_t out_size = dims[mode-1] * rank;
    std::vector<T> result(out_size);
    HIP_CHECK(hipMemcpy(result.data(), res.d_fmats[mode-1], sizeof(T) * out_size, hipMemcpyDeviceToHost));
    deallocate_mttkrp_resources(res, num_blocks);
    return result;
}

//----------------------------Version 2---------------------------------

//======================================================================
// Version 2 Kernel: Non-Hierarchical MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_kernel_1_v2(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, uint64_t m1_mask, uint64_t m2_mask, uint64_t m3_mask,
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
// Version 2 Kernel: Hierarchical MTTKRP
//======================================================================
template<typename T>
__global__ void mttkrp_kernel_2_v2(int mode, BLCO_BLOCK_GPU<T>* tensor, uint64_t nnz, uint64_t m1_mask, uint64_t m2_mask, 
uint64_t m3_mask, T* m1_fmat, T* m2_fmat, T* m3_fmat, int rows, int cols, int depth, int num_blocks, int rank, int smem_size,
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
// Version 2: Host Wrapper
//======================================================================
template<typename T, typename S>
std::vector<T> MTTKRP_BLCO_v2(int mode, const Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    const std::vector<int> dims = sparse_tensor.get_dims();
    int rows = dims[0], cols = dims[1], depth = dims[2];
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t non_zeros = sparse_tensor.get_nnz();
    int num_blocks = sparse_tensor.get_num_blocks();

    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    uint64_t m1_mask = masks[0], m2_mask = masks[1], m3_mask = masks[2];

    MTTKRP_Device_Resources<T> res = allocate_mttkrp_resources(sparse_tensor);
    
    // Logic for hierarchy selection
    int compute_units = get_compute_units();
    bool is_hierarchical = (compute_units > dims[mode-1]);

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);

    for(int i = 0; i < iter; i++){
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start, 0));

        if(!is_hierarchical){
            hipLaunchKernelGGL(
                mttkrp_kernel_1_v2<T>, 
                dim3(dimensions.first), dim3(dimensions.second), 0, 0,
                mode, res.d_tensor, non_zeros, m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
                rows, cols, depth, num_blocks, rank
            );
        } else {
            // Kernel 2 V2 uses smem_size as an int representing total entries
            size_t max_smem_bytes = get_max_shared_memory();
            int smem_entries = max_smem_bytes / sizeof(T);

            hipLaunchKernelGGL(
                mttkrp_kernel_2_v2<T>, 
                dim3(dimensions.first), dim3(dimensions.second), max_smem_bytes, 0,
                mode, res.d_tensor, non_zeros, m1_mask, m2_mask, m3_mask,
                res.d_fmats[0], res.d_fmats[1], res.d_fmats[2],
                rows, cols, depth, num_blocks, rank, smem_entries
            );
        }

        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        if(times.size() < iter) times.push_back(ms);
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
    }

    size_t out_size = dims[mode-1] * rank;
    std::vector<T> result(out_size);
    HIP_CHECK(hipMemcpy(result.data(), res.d_fmats[mode-1], sizeof(T) * out_size, hipMemcpyDeviceToHost));
    deallocate_mttkrp_resources(res, num_blocks);
    return result;
}