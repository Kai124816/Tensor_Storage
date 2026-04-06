#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "../tensor_implementations/blco_impl.h"
#include "kernel_utils.h"

namespace cg = cooperative_groups;
#define TILE_SIZE 64

//================================================================================
// Cooperative group kernels in the same style as the original BLCO paper
//================================================================================

//Three dimensional kernel
template <typename T>
__global__ void mttkrp_cg_3d_kernel(BLCO_ENTRY<T>* entries, uint64_t* block_ptr, int block_offset, 
uint64_t* masks, T* f0, T* f1, T* f2, T* output, int* dims, const int mode, 
const int rank, const int nnz, const int num_blocks) 
{
    // Cooperative group for entire thread block
    auto block = cg::this_thread_block();

    // Cooperative group tile of size TILE_SIZE for warp-level primitives
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank(); // Thread index within block
    const int wfid = tid / TILE_SIZE; // Wavefront's index within block

    // Store dims locally for OOB checks
    int dim0 = dims[0], dim1 = dims[1], dim2 = dims[2];

    // Allocate dynamic shared memory
    extern __shared__ int count[]; // Holds scan counts and temporary scratch space

    // Offsets into shared memory for each array type
    int* nnz_idx = (int*) (count + block.size());       // Holds delinearized indices (x,y,z) for each nnz
    int* nnz_out = (int*) (nnz_idx + 3 * block.size()); // Holds output row IDs
    T* nnz_val = (T*) (nnz_out + block.size());     // Holds values for nnz

    // Initialize all of the values in the shared scratchpad memory to 0
    nnz_idx[tid * 3 + 0] = 0;
    nnz_idx[tid * 3 + 1] = 0;
    nnz_idx[tid * 3 + 2] = 0;
    nnz_out[tid] = 0;
    nnz_val[tid] = 0;
    block.sync();

    // Compute the range of nnz elements this block will work on
    int curr_elem = block.group_index().x * block.size(); // Start index for this block
    int end_elem = min(nnz, curr_elem + block.size());     // End index (exclusive)

    // Read BLCO masks
    uint64_t m1_mask = masks[0]; uint64_t m2_mask = masks[1];
    uint64_t m3_mask = masks[2];

    while (curr_elem < end_elem) {
        count[tid] = 0; // Reset count for each thread

        uint64_t lin_idx;
        T value;
        int blco_block;
        int x, y, z, output_row;

        int test = curr_elem + tid;
        // Bounds check and delinearize if valid
        if (curr_elem + tid < end_elem) { 
            int safe_idx = curr_elem + tid;
            lin_idx = entries[safe_idx].index;
            value = entries[safe_idx].value;
            blco_block = find_block_csr(block_ptr, block_offset, safe_idx, num_blocks);
            int bit_widths[3] = {ceiling_log2(dims[0]), ceiling_log2(dims[1]), ceiling_log2(dims[2])};
            uint64_t local_masks[3] = {m1_mask, m2_mask, m3_mask};

            x = extract_mode(lin_idx, 1, local_masks, bit_widths, 3, blco_block);
            y = extract_mode(lin_idx, 2, local_masks, bit_widths, 3, blco_block);
            z = extract_mode(lin_idx, 3, local_masks, bit_widths, 3, blco_block);

            // Determine which row of output this nonzero maps to
            if (mode == 1) output_row = x;
            else if (mode == 2) output_row = y;
            else output_row = z;
        } 
        else{     
            // Pad invalid values
            x = y = z = output_row = -1;
        }

        block.sync();

        // Find subgroup of threads with same output_row value
        // 1. Get the group mask (Use uint64_t to safely support AMD wavefronts of size 64)
        unsigned long long sg_mask = tile.match_any(output_row);
        // 2. Recreate sg_rank = sg.thread_rank()
        // Count how many threads in our subgroup have a lower lane ID than us
        int lane_id = tile.thread_rank();
        uint64_t lower_threads_mask = (1ULL << lane_id) - 1;
        int sg_rank = __popcll(sg_mask & lower_threads_mask);
        // 3. Recreate sg_id = sg.meta_group_rank()
        // A subgroup's ID is determined by how many other subgroups started before it.
        // We can find this by counting the "leaders" of the other subgroups.
        bool is_leader = (sg_rank == 0);
        uint64_t leader_mask = __ballot(is_leader); // Mask of all leaders in the warp
        // Find the lane ID of our own subgroup's leader (__ffsll is 1-indexed, so we subtract 1)
        int leader_lane_id = __ffsll(sg_mask) - 1; 
        uint64_t lower_leaders_mask = (1ULL << leader_lane_id) - 1;
        // Our sub-group ID is the number of leaders that exist before our leader
        int sg_id = __popcll(leader_mask & lower_leaders_mask);

        // Count threads in each subgroup (coalesced prefix sum style)
        if (sg_rank == 0 && sg_id < TILE_SIZE - 1) count[wfid * TILE_SIZE + sg_id + 1] = __popcll(sg_mask);
        block.sync();

        // Perform exclusive scan on count to generate subgroup offsets
        sg_mask = count[tid];
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (lane_id >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();

        // Sorted rank = local rank + prefix offset
        sg_rank += count[wfid * TILE_SIZE + sg_id];

        // Store delinearized coords and values in sorted order
        int local_rank = wfid * TILE_SIZE + sg_rank;
        nnz_idx[local_rank * 3 + 0]  = x;
        nnz_idx[local_rank * 3 + 1]  = y;
        nnz_idx[local_rank * 3 + 2]  = z;
        nnz_out[local_rank] = output_row;
        if (curr_elem + tid < end_elem) nnz_val[local_rank] = value;

        // Build scan mask for segmented scan to know where each segment ends
        if (is_leader) sg_mask = 1ULL << sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);

        // Loop over all grouped elements for this warp
        int n = 0;
        int max_n = max(0, min((int)TILE_SIZE, end_elem - curr_elem - wfid * TILE_SIZE));
        while (n < max_n) {
            int smem_idx = wfid * TILE_SIZE + n;
            const int current_output_row = nnz_out[smem_idx];
            const int next_n = n;

            // Each thread computes a slice of rank
            for (int i = lane_id; i < rank; i += TILE_SIZE) {               
                T val_acc = 0.0;
                n = next_n;
                do {
                    int n_idx = wfid * TILE_SIZE + n;
                    T val = nnz_val[n_idx];
                    x = nnz_idx[n_idx * 3];
                    y = nnz_idx[n_idx * 3 + 1];
                    z = nnz_idx[n_idx * 3 + 2];

                    // Multiply factor matrices depending on mode
                    if (mode == 1) val *= f1[rank * y + i] * f2[rank * z + i];
                    else if (mode == 2) val *= f0[rank * x + i] * f2[rank * z + i];
                    else val *= f0[rank * x + i] * f1[rank * y + i];                 

                    val_acc += val;
                    ++n;
                } while (n < max_n && !(sg_mask & (1ULL << n)));
    
                // Atomically accumulate result into output
                atomicAdd(output + current_output_row * rank + i, val_acc);    
            }
            n = tile.shfl(n, 0); // Broadcast n to all threads in warp
        }
        // Move to next batch of elements for this block
        curr_elem += block.size();
    }
}

//Four dimensional kernel
template <typename T>
__global__ void mttkrp_cg_4d_kernel(BLCO_ENTRY<T>* entries, uint64_t* block_ptr, int block_offset, 
uint64_t* masks, T* f0, T* f1, T* f2, T* f3, T* output, int* dims, const int mode, 
const int rank, const int nnz, const int num_blocks) 
{
    // Cooperative group for entire thread block
    auto block = cg::this_thread_block();

    // Cooperative group tile of size TILE_SIZE for warp-level primitives
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank(); // Thread index within block
    const int wfid = tid / TILE_SIZE; // Wavefront's index within block

    // Store dims locally for OOB checks
    int dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim3 = dims[3];

    // Allocate dynamic shared memory
    extern __shared__ int count[]; // Holds scan counts and temporary scratch space

    // Offsets into shared memory for each array type
    int* nnz_idx = (int*) (count + block.size());       // Holds delinearized indices (x,y,z,w) for each nnz
    int* nnz_out = (int*) (nnz_idx + 4 * block.size()); // Holds output row IDs
    T* nnz_val = (T*) (nnz_out + block.size());     // Holds values for nnz

    // Initialize all of the values in the shared scratchpad memory to 0
    nnz_idx[tid * 4 + 0] = 0;
    nnz_idx[tid * 4 + 1] = 0;
    nnz_idx[tid * 4 + 2] = 0;
    nnz_idx[tid * 4 + 3] = 0;
    nnz_out[tid] = 0;
    nnz_val[tid] = 0;
    block.sync();

    // Compute the range of nnz elements this block will work on
    int curr_elem = block.group_index().x * block.size(); // Start index for this block
    int end_elem = min(nnz, curr_elem + block.size());     // End index (exclusive)

    // Read BLCO masks
    uint64_t m1_mask = masks[0]; uint64_t m2_mask = masks[1];
    uint64_t m3_mask = masks[2]; uint64_t m4_mask = masks[3];

    while (curr_elem < end_elem) {
        count[tid] = 0; // Reset count for each thread

        uint64_t lin_idx;
        T value;
        int blco_block;
        int x, y, z, w, output_row;

        int test = curr_elem + tid;
        // Bounds check and delinearize if valid
        if (curr_elem + tid < end_elem) { 
            int safe_idx = curr_elem + tid;
            lin_idx = entries[safe_idx].index;
            value = entries[safe_idx].value;
            blco_block = find_block_csr(block_ptr, block_offset, safe_idx, num_blocks);
            int bit_widths[4] = {ceiling_log2(dims[0]), ceiling_log2(dims[1]), ceiling_log2(dims[2]), ceiling_log2(dims[3])};
            uint64_t local_masks[4] = {m1_mask, m2_mask, m3_mask, m4_mask};

            x = extract_mode(lin_idx, 1, local_masks, bit_widths, 4, blco_block);
            y = extract_mode(lin_idx, 2, local_masks, bit_widths, 4, blco_block);
            z = extract_mode(lin_idx, 3, local_masks, bit_widths, 4, blco_block);
            w = extract_mode(lin_idx, 4, local_masks, bit_widths, 4, blco_block);

            // Determine which row of output this nonzero maps to
            if (mode == 1) output_row = x;
            else if (mode == 2) output_row = y;
            else if (mode == 3) output_row = z;
            else output_row = w;
        } 
        else{     
            // Pad invalid values
            x = y = z = w = output_row = -1;
        }

        block.sync();

        // Find subgroup of threads with same output_row value
        // 1. Get the group mask (Use uint64_t to safely support AMD wavefronts of size 64)
        unsigned long long sg_mask = tile.match_any(output_row);
        // 2. Recreate sg_rank = sg.thread_rank()
        // Count how many threads in our subgroup have a lower lane ID than us
        int lane_id = tile.thread_rank();
        uint64_t lower_threads_mask = (1ULL << lane_id) - 1;
        int sg_rank = __popcll(sg_mask & lower_threads_mask);
        // 3. Recreate sg_id = sg.meta_group_rank()
        // A subgroup's ID is determined by how many other subgroups started before it.
        // We can find this by counting the "leaders" of the other subgroups.
        bool is_leader = (sg_rank == 0);
        uint64_t leader_mask = __ballot(is_leader); // Mask of all leaders in the warp
        // Find the lane ID of our own subgroup's leader (__ffsll is 1-indexed, so we subtract 1)
        int leader_lane_id = __ffsll(sg_mask) - 1; 
        uint64_t lower_leaders_mask = (1ULL << leader_lane_id) - 1;
        // Our sub-group ID is the number of leaders that exist before our leader
        int sg_id = __popcll(leader_mask & lower_leaders_mask);

        // Count threads in each subgroup (coalesced prefix sum style)
        if (sg_rank == 0 && sg_id < TILE_SIZE - 1) count[wfid * TILE_SIZE + sg_id + 1] = __popcll(sg_mask);
        block.sync();

        // Perform exclusive scan on count to generate subgroup offsets
        sg_mask = count[tid];
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (lane_id >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();

        // Sorted rank = local rank + prefix offset
        sg_rank += count[wfid * TILE_SIZE + sg_id];

        // Store delinearized coords and values in sorted order
        int local_rank = wfid * TILE_SIZE + sg_rank;
        nnz_idx[local_rank * 4 + 0]  = x;
        nnz_idx[local_rank * 4 + 1]  = y;
        nnz_idx[local_rank * 4 + 2]  = z;
        nnz_idx[local_rank * 4 + 3]  = w;
        nnz_out[local_rank] = output_row;
        if (curr_elem + tid < end_elem) nnz_val[local_rank] = value;

        // Build scan mask for segmented scan to know where each segment ends
        if (is_leader) sg_mask = 1ULL << sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);

        // Loop over all grouped elements for this warp
        int n = 0;
        int max_n = max(0, min((int)TILE_SIZE, end_elem - curr_elem - wfid * TILE_SIZE));
        while (n < max_n) {
            int smem_idx = wfid * TILE_SIZE + n;
            const int current_output_row = nnz_out[smem_idx];
            const int next_n = n;

            // Each thread computes a slice of rank
            for (int i = lane_id; i < rank; i += TILE_SIZE) {               
                T val_acc = 0.0;
                n = next_n;
                do {
                    int n_idx = wfid * TILE_SIZE + n;
                    T val = nnz_val[n_idx];
                    x = nnz_idx[n_idx * 4];
                    y = nnz_idx[n_idx * 4 + 1];
                    z = nnz_idx[n_idx * 4 + 2];
                    w = nnz_idx[n_idx * 4 + 3];

                    // Multiply factor matrices depending on mode
                    if (mode == 1) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (mode == 2) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i];
                    else if (mode == 3) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i];                 
                    else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i];

                    val_acc += val;
                    ++n;
                } while (n < max_n && !(sg_mask & (1ULL << n)));
    
                // Atomically accumulate result into output
                atomicAdd(output + current_output_row * rank + i, val_acc);    
            }
            n = tile.shfl(n, 0); // Broadcast n to all threads in warp
        }
        // Move to next batch of elements for this block
        curr_elem += block.size();
    }
}

//Five dimensional kernel
template <typename T>
__global__ void mttkrp_cg_5d_kernel(BLCO_ENTRY<T>* entries, uint64_t* block_ptr, int block_offset, 
uint64_t* masks, T* f0, T* f1, T* f2, T* f3, T* f4, T* output, int* dims, const int mode, 
const int rank, const int nnz, const int num_blocks) 
{
    // Cooperative group for entire thread block
    auto block = cg::this_thread_block();

    // Cooperative group tile of size TILE_SIZE for warp-level primitives
    auto tile = cg::tiled_partition<TILE_SIZE>(block);
    const int tid = block.thread_rank(); // Thread index within block
    const int wfid = tid / TILE_SIZE; // Wavefront's index within block

    // Store dims locally for OOB checks
    int dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim3 = dims[3], dim4 = dims[4];

    // Allocate dynamic shared memory
    extern __shared__ int count[]; // Holds scan counts and temporary scratch space

    // Offsets into shared memory for each array type
    int* nnz_idx = (int*) (count + block.size());       // Holds delinearized indices (x,y,z,w,v) for each nnz
    int* nnz_out = (int*) (nnz_idx + 5 * block.size()); // Holds output row IDs
    T* nnz_val = (T*) (nnz_out + block.size());     // Holds values for nnz

    // Initialize all of the values in the shared scratchpad memory to 0
    nnz_idx[tid * 5 + 0] = 0;
    nnz_idx[tid * 5 + 1] = 0;
    nnz_idx[tid * 5 + 2] = 0;
    nnz_idx[tid * 5 + 3] = 0;
    nnz_idx[tid * 5 + 4] = 0;
    nnz_out[tid] = 0;
    nnz_val[tid] = 0;
    block.sync();

    // Compute the range of nnz elements this block will work on
    int curr_elem = block.group_index().x * block.size(); // Start index for this block
    int end_elem = min(nnz, curr_elem + block.size());     // End index (exclusive)

    // Read BLCO masks
    uint64_t m1_mask = masks[0]; uint64_t m2_mask = masks[1];
    uint64_t m3_mask = masks[2]; uint64_t m4_mask = masks[3];
    uint64_t m5_mask = masks[4];

    while (curr_elem < end_elem) {
        count[tid] = 0; // Reset count for each thread

        uint64_t lin_idx;
        T value;
        int blco_block;
        int x, y, z, w, v, output_row;

        // Bounds check and delinearize if valid
        if (curr_elem + tid < end_elem) { 
            int safe_idx = curr_elem + tid;
            lin_idx = entries[safe_idx].index;
            value = entries[safe_idx].value;
            blco_block = find_block_csr(block_ptr, block_offset, safe_idx, num_blocks);
            int bit_widths[5] = {ceiling_log2(dims[0]), ceiling_log2(dims[1]), ceiling_log2(dims[2]), ceiling_log2(dims[3]), ceiling_log2(dims[4])};
            uint64_t local_masks[5] = {m1_mask, m2_mask, m3_mask, m4_mask, m5_mask};

            x = extract_mode(lin_idx, 1, local_masks, bit_widths, 5, blco_block);
            y = extract_mode(lin_idx, 2, local_masks, bit_widths, 5, blco_block);
            z = extract_mode(lin_idx, 3, local_masks, bit_widths, 5, blco_block);
            w = extract_mode(lin_idx, 4, local_masks, bit_widths, 5, blco_block);
            v = extract_mode(lin_idx, 5, local_masks, bit_widths, 5, blco_block);

            // Determine which row of output this nonzero maps to
            if (mode == 1) output_row = x;
            else if (mode == 2) output_row = y;
            else if (mode == 3) output_row = z;
            else if (mode == 4) output_row = w;
            else output_row = v;
        } 
        else{     
            // Pad invalid values
            x = y = z = w = v = output_row = -1;
        }

        block.sync();

        // Find subgroup of threads with same output_row value
        // 1. Get the group mask (Use uint64_t to safely support AMD wavefronts of size 64)
        unsigned long long sg_mask = tile.match_any(output_row);
        // 2. Recreate sg_rank = sg.thread_rank()
        // Count how many threads in our subgroup have a lower lane ID than us
        int lane_id = tile.thread_rank();
        uint64_t lower_threads_mask = (1ULL << lane_id) - 1;
        int sg_rank = __popcll(sg_mask & lower_threads_mask);
        // 3. Recreate sg_id = sg.meta_group_rank()
        // A subgroup's ID is determined by how many other subgroups started before it.
        // We can find this by counting the "leaders" of the other subgroups.
        bool is_leader = (sg_rank == 0);
        uint64_t leader_mask = __ballot(is_leader); // Mask of all leaders in the warp
        // Find the lane ID of our own subgroup's leader (__ffsll is 1-indexed, so we subtract 1)
        int leader_lane_id = __ffsll(sg_mask) - 1; 
        uint64_t lower_leaders_mask = (1ULL << leader_lane_id) - 1;
        // Our sub-group ID is the number of leaders that exist before our leader
        int sg_id = __popcll(leader_mask & lower_leaders_mask);

        // Count threads in each subgroup (coalesced prefix sum style)
        if (sg_rank == 0 && sg_id < TILE_SIZE - 1) count[wfid * TILE_SIZE + sg_id + 1] = __popcll(sg_mask);
        block.sync();

        // Perform exclusive scan on count to generate subgroup offsets
        sg_mask = count[tid];
        #pragma unroll
        for (int j = 1; j < tile.size(); j <<= 1) {
            int temp = tile.shfl_up(sg_mask, j);
            if (lane_id >= j) sg_mask += temp;
        }
        count[tid] = sg_mask;
        block.sync();

        // Sorted rank = local rank + prefix offset
        sg_rank += count[wfid * TILE_SIZE + sg_id];

        // Store delinearized coords and values in sorted order
        int local_rank = wfid * TILE_SIZE + sg_rank;
        nnz_idx[local_rank * 5 + 0]  = x;
        nnz_idx[local_rank * 5 + 1]  = y;
        nnz_idx[local_rank * 5 + 2]  = z;
        nnz_idx[local_rank * 5 + 3]  = w;
        nnz_idx[local_rank * 5 + 4]  = v;
        nnz_out[local_rank] = output_row;
        if (curr_elem + tid < end_elem) nnz_val[local_rank] = value;

        // Build scan mask for segmented scan to know where each segment ends
        if (is_leader) sg_mask = 1ULL << sg_rank;
        else sg_mask = 0;
        #pragma unroll
        for (int j = tile.size()/2; j > 0; j >>= 1) {
            sg_mask |= tile.shfl_down(sg_mask, j);
        }
        sg_mask = tile.shfl(sg_mask, 0);

        // Loop over all grouped elements for this warp
        int n = 0;
        int max_n = max(0, min((int)TILE_SIZE, end_elem - curr_elem - wfid * TILE_SIZE));
        while (n < max_n) {
            int smem_idx = wfid * TILE_SIZE + n;
            const int current_output_row = nnz_out[smem_idx];
            const int next_n = n;

            // Each thread computes a slice of rank
            for (int i = lane_id; i < rank; i += TILE_SIZE) {               
                T val_acc = 0.0;
                n = next_n;
                do {
                    int n_idx = wfid * TILE_SIZE + n;
                    T val = nnz_val[n_idx];
                    x = nnz_idx[n_idx * 5];
                    y = nnz_idx[n_idx * 5 + 1];
                    z = nnz_idx[n_idx * 5 + 2];
                    w = nnz_idx[n_idx * 5 + 3];
                    v = nnz_idx[n_idx * 5 + 4];

                    // Multiply factor matrices depending on mode
                    if (mode == 1) val *= f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i] * f4[rank * v + i];
                    else if (mode == 2) val *= f0[rank * x + i] * f2[rank * z + i] * f3[rank * w + i] * f4[rank * v + i];
                    else if (mode == 3) val *= f0[rank * x + i] * f1[rank * y + i] * f3[rank * w + i] * f4[rank * v + i];                 
                    else if (mode == 4) val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i] * f4[rank * v + i];
                    else val *= f0[rank * x + i] * f1[rank * y + i] * f2[rank * z + i] * f3[rank * w + i];

                    val_acc += val;
                    ++n;
                } while (n < max_n && !(sg_mask & (1ULL << n)));
    
                // Atomically accumulate result into output
                atomicAdd(output + current_output_row * rank + i, val_acc);    
            }
            n = tile.shfl(n, 0); // Broadcast n to all threads in warp
        }
        // Move to next batch of elements for this block
        curr_elem += block.size();
    }
}

template<typename T, typename S>
void MTTKRP_BLCO_in_progress(int mode, Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1)
{
    //Dimensions
    const std::vector<BLCO_BLOCK_CPU<T>> blco_cpu = sparse_tensor.get_blco();
    const std::vector<int> dims = sparse_tensor.get_dims();
    const int rank = sparse_tensor.get_factor_rank();
    uint64_t nnz = sparse_tensor.get_nnz();
    const std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    int num_blocks = blco_cpu[blco_cpu.size() - 1].block;
    const std::vector<std::vector<T>> h_fmats = sparse_tensor.get_fmats();

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
        HIP_CHECK(hipMemcpy(d_fmats[i], h_fmats[i].data(), 
                            sizeof(T) * fmat_size, hipMemcpyHostToDevice));
    }
    
    //Allocate dimension arrays
    int* d_dims;
    HIP_CHECK(hipMalloc(&(d_dims), sizeof(int) * dims.size()));
    HIP_CHECK(hipMemcpy(d_dims, dims.data(), 
    sizeof(int) * dims.size(), hipMemcpyHostToDevice));

    // Allocate mask arrays
    uint64_t* d_masks;
    HIP_CHECK(hipMalloc(&(d_masks), sizeof(uint64_t) * masks.size()));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), 
    sizeof(uint64_t) * masks.size(), hipMemcpyHostToDevice));

    size_t shared_mem = get_max_shared_memory();
    std::pair<int,int> dimensions = determine_dimensions_coalesced_group<T>(nnz, shared_mem, rank);
    if(dimensions.first == -1){
        std::cerr << "Not enough shared memory for implementation" << std::endl;
        return;
    }

    bool collect_times = false;
    if(times.size() == 0) collect_times = true;

    if(dims.size() == 3){
        for(int i = 0; i < iter; i++) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(mttkrp_cg_3d_kernel<T>, dim3(dimensions.first), dim3(dimensions.second), shared_mem, 0,
                blco_device.tensor_entries, blco_device.block_ptr, blco_device.offset,
                d_masks, d_fmats[0], d_fmats[1], d_fmats[2], d_fmats[mode-1], d_dims, mode, rank, nnz, num_blocks);

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
        for(int i = 0; i < iter; i++) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(mttkrp_cg_4d_kernel<T>, dim3(dimensions.first), dim3(dimensions.second), shared_mem, 0,
                blco_device.tensor_entries, blco_device.block_ptr, blco_device.offset,
                d_masks, d_fmats[0], d_fmats[1], d_fmats[2], d_fmats[3], d_fmats[mode-1], d_dims, mode, rank, nnz, num_blocks);

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
        for(int i = 0; i < iter; i++) {
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipDeviceSynchronize());
            // Record start
            HIP_CHECK(hipEventRecord(start, 0));

            hipLaunchKernelGGL(mttkrp_cg_5d_kernel<T>, dim3(dimensions.first), dim3(dimensions.second), shared_mem, 0,
                blco_device.tensor_entries, blco_device.block_ptr, blco_device.offset,
                d_masks, d_fmats[0], d_fmats[1], d_fmats[2], d_fmats[3], d_fmats[4], d_fmats[mode-1], d_dims, mode, rank, nnz, num_blocks);

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
    sparse_tensor.reassign_fmat(mode, result);

    for(int i = 0; i < dims.size(); i++){
        HIP_CHECK(hipFree(d_fmats[i]));
    }
    HIP_CHECK(hipFree(d_dims));
    HIP_CHECK(hipFree(d_masks));
    HIP_CHECK(hipFree(blco_device.tensor_entries));
    HIP_CHECK(hipFree(blco_device.block_ptr));
}
