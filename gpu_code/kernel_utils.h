#pragma once
#include <hip/hip_runtime.h>
#include "../tensor_implementations/blco_impl.h"

//See HIP_instructions.txt for an explanation of the HIP functions in this file

// Error-checking macro for HIP API calls
#define HIP_CHECK(cmd) do { hipError_t e = (cmd); if (e != hipSuccess) { \
    fprintf(stderr,"HIP ERROR %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

//======================================================================
// Functions that get information about the GPU
//======================================================================
//Get maximum shared memory
inline size_t get_max_shared_memory() {
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // The key property for kernel launch limitation:
    return props.sharedMemPerBlock;
}

int get_compute_units() {
    int deviceId = 0; // Use the first GPU
    hipDeviceProp_t props;
    
    // Query device properties
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
    
    return props.multiProcessorCount;
}

inline double get_gpu_memory_capacity() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No HIP-compatible devices found.\n";
        return -1.0;
    }

    // Assuming we check device 0
    int device = 0;
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    // props.totalGlobalMem gives the memory size in bytes
    unsigned long long total_bytes = props.totalGlobalMem;

    // Convert bytes to GB for readability
    double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

    // You can also check available memory at runtime
    size_t free, total;
    HIP_CHECK(hipMemGetInfo(&free, &total));
    double free_gb = static_cast<double>(free) / (1024.0 * 1024.0 * 1024.0);
    return free_gb;
}

inline void print_amd_gpu_model() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "No HIP-compatible AMD devices found.\n";
        return;
    }

    // Loop through all found devices
    for (int i = 0; i < 1; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));

        // The name of the GPU is stored in props.name
        std::cout << "Device: " << props.name << "\n";
        std::cout << "Arch: " << props.gcnArchName << "\n";
        std::cout << "Shared Memory: " << props.sharedMemPerBlock << "bytes\n";
        std::cout << "Currently Free VRAM: " << get_gpu_memory_capacity() << " GB\n";
    }
}

//-----------------------------------------------
// Helper function for reporting HIP errors
//-----------------------------------------------
inline void checkHIP(const hipError_t e, const char* where) {
    if (e != hipSuccess) {
        std::cerr << where << " failed: " << hipGetErrorString(e) << "\n";
    }
}

//======================================================================
// Move BLCO tensor blocks from CPU to GPU memory
//======================================================================
template<typename T>
inline void blocks_to_gpu(BLCO_BLOCK_GPU<T>*& d_block_arr,
                   const std::vector<BLCO_BLOCK_CPU<T>>& tensor,
                   int num_blocks)
{
    if (num_blocks <= 0) {
        d_block_arr = nullptr;
        return;
    }

    // Temporary host-side struct array (with GPU pointers inside)
    std::vector<BLCO_BLOCK_GPU<T>> h_arr_for_gpu(num_blocks);

    // Copy each CPU block into GPU memory
    for (int i = 0; i < num_blocks; ++i) {
        int num_elements = tensor[i].size;
        h_arr_for_gpu[i].block = tensor[i].block;
        h_arr_for_gpu[i].size  = num_elements;

        if (num_elements > 0) {
            // Allocate device array for this block
            BLCO_ENTRY<T>* d_entries;
            HIP_CHECK(hipMalloc(&d_entries, num_elements * sizeof(BLCO_ENTRY<T>)));

            // Copy block contents from CPU → GPU
            HIP_CHECK(hipMemcpy(d_entries,  tensor[i].entries.data(),  num_elements * sizeof(BLCO_ENTRY<T>), hipMemcpyHostToDevice));

            h_arr_for_gpu[i].entries  = d_entries;
        } else {
            // Empty block → null pointers
            h_arr_for_gpu[i].entries = nullptr;
        }
    }

    // Copy the struct array (with device pointers) into GPU memory
    HIP_CHECK(hipMemcpy(d_block_arr, h_arr_for_gpu.data(),
              num_blocks * sizeof(BLCO_BLOCK_GPU<T>),
              hipMemcpyHostToDevice));
}

//======================================================================
// Free BLCO tensor blocks from GPU memory
//======================================================================
template<typename T>
inline void free_blocks_from_gpu(BLCO_BLOCK_GPU<T>* gpu_block_arr, int num_blocks)
{
    if (!gpu_block_arr || num_blocks <= 0) return;

    // Copy GPU struct array back to CPU so we can access the device pointers
    std::vector<BLCO_BLOCK_GPU<T>> h_blocks(num_blocks);
    HIP_CHECK(hipMemcpy(h_blocks.data(), gpu_block_arr,
              num_blocks * sizeof(BLCO_BLOCK_GPU<T>),
              hipMemcpyDeviceToHost));

    // Free device memory for each block’s values/indexes
    for (int b = 0; b < num_blocks; ++b) {
        if (h_blocks[b].entries) HIP_CHECK(hipFree(h_blocks[b].entries));
    }

    // Free the array of structs itself
    HIP_CHECK(hipFree(gpu_block_arr));
}

// ----------------------------
// Grid/Block Dimension Helpers
// ----------------------------

// Compute grid/block dimensions when NOT using shared memory (smem).
// Decides how many thread blocks (dims.first) and how many threads per block (dims.second)
// should be launched based on the number of nonzeros and a default wavefront size.
//
// Strategy: 
//  - If the tensor is small (≤ 320 nonzeros), launch just 1 block with enough threads
//    to cover all nonzeros, rounded up to a multiple of wf_sz.
//  - Otherwise, assign 320 threads per block and compute how many blocks are needed.
inline std::pair<int,int> determine_dimensions_no_smem(uint64_t non_zeros, int wf_sz = 64)
{
    std::pair<int,int> dims;

    if(non_zeros <= 320){
        dims.first = 1;  // only one block
        int mul = non_zeros / wf_sz;
        if(mul * wf_sz < non_zeros) mul++;  // round up to next multiple of wf_sz
        dims.second = wf_sz * mul;          // threads per block
    }
    else{
        dims.first = non_zeros / 320;       // number of blocks
        if(non_zeros % 320 != 0) dims.first++;
        dims.second = 320;                  // threads per block fixed at 320
    }

    return dims;
}

// Compute grid/block dimensions when using shared memory (smem).
// Adds an extra check to ensure the shared memory requirement fits within MAX_SHARED_MEM.
//
// Parameters:
//   - non_zeros: number of nonzero entries in tensor
//   - ui: number of unique target indices per wavefront
//   - tensor_rank: rank (width) of the factor matrices
//   - wf_sz: wavefront size (default 64 on AMD GPUs)
//
// Logic:
//  - Similar to above, but with a max threads-per-block of 1024.
//  - Check whether smem usage (ui * rank * wf_per_block * sizeof(T)) fits within GPU limits.
//  - If not, reduce wf_per_block until memory fits.
template<typename T>
inline std::pair<int,int> determine_dimensions_smem(uint64_t non_zeros, int ui, int tensor_rank, int wf_sz = 64)
{
    std::pair<int,int> dims;

    if(non_zeros <= 1024){
        dims.first = 1;
        int mul = non_zeros / wf_sz;
        if(mul * wf_sz < non_zeros) mul++;
        dims.second = wf_sz * mul;
    }
    else{
        dims.first = non_zeros / 1024;
        if(non_zeros % 1024 != 0) dims.first++;
        dims.second = 1024;
    }

    size_t max_shared_mem = get_max_shared_memory();

    // Check shared memory usage
    int wf_per_block = dims.second / wf_sz;
    bool enough_mem = ui * tensor_rank * wf_per_block * sizeof(T) <= max_shared_mem;

    if(!enough_mem){
        // Reduce wf_per_block until memory fits
        int c1 = ui * tensor_rank * sizeof(T);
        for(int i = 0; i < wf_per_block; i++){
            if(--wf_per_block * c1 < max_shared_mem) break;
        }

        // Adjust block dimensions
        dims.second = wf_per_block * wf_sz;
        dims.first = non_zeros / dims.second;
        if(non_zeros % dims.second != 0) dims.first++;
    }

    return dims;
}

// ----------------------------
// Device-Side Utilities
// ----------------------------

// Compute ceil(log2(x)) at runtime on device.
// Used to determine how many bits are required to represent dimensions.
__device__ inline int ceiling_log2(int x) 
{
    if (x == 1) return 0;
    int res = 0;
    while (x) {
        x >>= 1;   // divide by 2 each step
        ++res;
    }
    return res;
}

// Find which block of the BLCO tensor the current thread's index falls into.
// Returns the entry for the corresponding thread
template<typename T>
__device__ inline BLCO_ENTRY<T> extract_entry(BLCO_BLOCK_GPU<T>* tensor, int num_blocks, int& block)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prefix_sum = 0;

    for(int i = 0; i < num_blocks; i++){
        if(idx < prefix_sum + tensor[i].size){
            block = tensor[i].block;
            return tensor[i].entries[idx - prefix_sum];
        }
        prefix_sum += tensor[i].size;
    }
    return BLCO_ENTRY<T>{0,0}; // invalid just return empty entry
}

// Extract mode-specific index from a linearized index.
// Uses masks and shifts to recover the appropriate mode coordinate.
// Handles overflow cases when row+col+depth bits exceed 64 (needs block info).
__device__ inline int extract_mode(uint64_t linear_idx, int mode, 
    const uint64_t* bitmasks, const int* bit_widths, int rank, int block)
{
    // Compute total shift by summing bit widths of all previous modes
    int shift = 0;
    for (int m = 0; m < mode - 1; m++) {
        shift += bit_widths[m];
    }

    // Extract using mask and shift
    uint64_t mask = bitmasks[mode - 1];
    uint64_t output = (linear_idx >> shift) & mask;

    // Handle 64-bit overflow case (rare but possible for very large tensors)
    int total_bits = 0;
    for (int m = 0; m < rank; m++) total_bits += bit_widths[m];

    if (total_bits > 64) {
        // If this mode extends beyond 64 bits, use block info to reconstruct high bits
        int extra_bits = total_bits - 64;

        // Compute how many bits this mode contributes beyond 64
        int mode_start = shift;
        int mode_end = shift + bit_widths[mode - 1];

        if (mode_end > 64) {
            int overlap_bits = mode_end - 64;
            output |= static_cast<uint64_t>(block) << (bit_widths[mode - 1] - overlap_bits);
        }
    }

    return static_cast<int>(output);
}

// ----------------------------
// Wavefront Reduction Helpers
// ----------------------------

// Construct a mask of which threads in a wavefront have the same target_val.
// Uses shuffles (__shfl) to compare values across lanes in a wavefront.
__device__ inline void group_mask_shfl_only(int target_val, unsigned long long &output_mask, int wf_size = 64) 
{
    int wavefront_idx = threadIdx.x % wf_size;   

    for (int k = 0; k < wf_size; ++k) {
        int val_k = __shfl(target_val, k, wf_size);
        if (val_k == target_val) {
            output_mask |= (1ULL << k);
        }
    }
}

// Reduce within a wavefront: figure out how many threads share the same target index,
// and which number (mode_num) this thread is within its subgroup.
// Returns: mask of participating lanes.
template <typename T>
__device__ inline unsigned long long wavefront_group_reduce_1(int target_index, T value,
                                                       int &total_modes, int &mode_num,
                                                       int wavefront_size = 64)
{
    int wavefront_idx = threadIdx.x % wavefront_size;

    unsigned long long mask = 0ULL;
    group_mask_shfl_only(target_index, mask);

    unsigned long long mn_mask = (1ULL << wavefront_idx) - 1; 
    mn_mask &= mask;

    int leader_idx = __ffsll(mask) - 1;   // first set bit
    bool is_leader = (wavefront_idx == leader_idx);
    total_modes = __popcll(mask);         // population count = group size
    mode_num = __popcll(mn_mask);         // rank of this thread in group

    return mask;
}

// Same as above but also outputs whether this thread is the group leader.
template <typename T>
__device__ inline unsigned long long wavefront_group_reduce_2(int target_index, T value,
                                                       int &total_modes, int &mode_num,
                                                       int &is_leader,
                                                       int wavefront_size = 64)
{
    int wavefront_idx = threadIdx.x % wavefront_size;

    unsigned long long mask = 0ULL;
    group_mask_shfl_only(target_index, mask);

    unsigned long long mn_mask = (1ULL << wavefront_idx) - 1; 
    mn_mask &= mask;

    int leader_idx = __ffsll(mask) - 1;
    is_leader = (wavefront_idx == leader_idx);
    total_modes = __popcll(mask);
    mode_num = __popcll(mn_mask);

    return mask;
}

// ----------------------------
// Debugging/Utility Helpers
// ----------------------------

// Print a vector (for debugging inside GPU kernel).
template<typename T>
__device__ inline void print_vector(T* vec, int size)
{
    for(int i = 0; i < size; i++){
        printf("%d ", vec[i]);
    }
    printf("\n");
}

// Find the position of a row in an offsets array.
// Returns -1 if row is not found.
__device__ inline int find_offset(int row, int* offsets, int size)
{
    for(int i = 0; i < size; i++){
        if(row == offsets[i]) return i;
    }
    return -1;
}