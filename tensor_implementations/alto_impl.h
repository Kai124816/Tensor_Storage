#ifndef ALTO_H
#define ALTO_H

#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include "tensor_impl.h"
#include "../utility/utils.h"

//======================================================================
// ALTOEntry: Represents one nonzero entry in ALTO format
//======================================================================
// linear_index = compact ALTO-encoded coordinate (bitmask-based)
// value        = stored tensor value at that location
template<typename T, typename S>
struct ALTOEntry {
    S linear_index;  
    T value;            
};

//======================================================================
// Alto_Tensor_3D
//======================================================================
// Inherits from Tensor_3D<T,S> and extends it with ALTO storage.
//
// Key features:
//   - Encodes (i,j,k, ...) coordinates into a *linearized ALTO index*
//     using adaptive bitmask assignment.
//   - Stores nonzeros in a sorted vector (by ALTO index).
//   - Partitions NNZs for parallel algorithms.
//======================================================================
template<typename T, typename S>
class Alto_Tensor : public Tensor<T, S>
{
protected:
    int num_threads;                            // Number of threads for parallel MTTKRP                        
    std::vector<S> bitmasks;                    // Stores bitmasks for each mode
    std::vector<int> partitions;                // Partition boundaries for NNZ distribution
    std::vector<ALTOEntry<T,S>> alto_tensor;    // Compact ALTO representation of the tensor

    //------------------------------------------------------------------
    // Read a binary tensor file directly into alto_tensor, bypassing
    // any intermediate NNZ_Entry representation.
    // File format (mirrors read_tensor_file_binary from utils.h):
    //   - No file header
    //   - Each entry: rank x int32_t coords (1-indexed), then 1 x T value
    //------------------------------------------------------------------
    void read_binary_to_alto(const std::string& filename, int64_t nnz)
    {
        if (filename.empty())
            throw std::invalid_argument("read_binary_to_alto: filename must not be empty");
        if (nnz <= 0)
            throw std::invalid_argument("read_binary_to_alto: nnz must be positive, got " +
                                       std::to_string(nnz));

        int fd = open(filename.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("read_binary_to_alto: could not open file \"" + filename + "\"");

        const int    file_rank    = this->rank;
        const size_t coords_bytes = file_rank * sizeof(int32_t);
        const size_t value_bytes  = sizeof(T);
        const size_t entry_bytes  = coords_bytes + value_bytes;

        alto_tensor.clear();
        alto_tensor.resize(nnz);

        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();
            
            // Calculate thread partition
            int64_t entries_per_thread = nnz / num_threads;
            int64_t start_entry = thread_id * entries_per_thread;
            int64_t end_entry = (thread_id == num_threads - 1) ? nnz : start_entry + entries_per_thread;
            int64_t thread_entries = end_entry - start_entry;
            
            if (thread_entries > 0) {
                // Read chunks of 1 million entries (~20-40 MB) to maintain bounded memory overhead
                const int64_t chunk_size = 1048576;
                std::vector<char> local_buffer;
                std::vector<int> coords(file_rank);

                int64_t entries_processed = 0;
                while (entries_processed < thread_entries) {
                    int64_t current_chunk_entries = std::min(chunk_size, thread_entries - entries_processed);
                    size_t bytes_to_read = current_chunk_entries * entry_bytes;
                    
                    local_buffer.resize(bytes_to_read);
                    ssize_t bytes_read = pread(fd, local_buffer.data(), bytes_to_read, 
                                            (start_entry + entries_processed) * entry_bytes);
                    
                    if (bytes_read != static_cast<ssize_t>(bytes_to_read)) {
                        #pragma omp critical
                        {
                            std::cerr << "Thread " << thread_id << " failed to read full chunk (expected " 
                                      << bytes_to_read << " bytes, got " << bytes_read << ")" << std::endl;
                        }
                        break; 
                    }

                    const char* ptr = local_buffer.data();
                    for (int64_t i = 0; i < current_chunk_entries; ++i) {
                        // Parse coordinates
                        std::copy(ptr, ptr + coords_bytes, reinterpret_cast<char*>(coords.data()));
                        ptr += coords_bytes;

                        // Parse value
                        T value;
                        std::copy(ptr, ptr + value_bytes, reinterpret_cast<char*>(&value));
                        ptr += value_bytes;

                        // Convert 1-indexed → 0-indexed
                        for (auto& c : coords) c -= 1;

                        alto_tensor[start_entry + entries_processed + i].linear_index = translate_idx(coords);
                        alto_tensor[start_entry + entries_processed + i].value = value;
                    }

                    entries_processed += current_chunk_entries;
                }
            }
        }
        
        close(fd);

        std::sort(alto_tensor.begin(), alto_tensor.end(),
            [](const ALTOEntry<T,S>& a, const ALTOEntry<T,S>& b) {
                return a.linear_index < b.linear_index;
            });
    }

    //------------------------------------------------------------------
    // Choose "ideal" number of threads for parallel algorithm based on NNZ size
    //------------------------------------------------------------------
    int ideal_threads()
    {
        const int thresholds[] = {250, 500, 750, 1000, 2000, 20000};
        const int max_threads = 8;
    
        for (int t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); ++t) {
            if (this->nnz_entries < thresholds[t]) return t + 2;
        }
        return max_threads;
    }

    //------------------------------------------------------------------
    // Partition NNZs evenly across threads
    //------------------------------------------------------------------
    void set_partitions()
    {
        int partition_size = this->nnz_entries / num_threads;
        if(this->nnz_entries % num_threads != 0) partition_size++;

        int sum = 0;
        for(int i = 0; i < num_threads; i++){
            sum += partition_size;
            if(sum > this->nnz_entries) sum = this->nnz_entries;
            partitions.push_back(sum);
        }
    }

    //------------------------------------------------------------------
    // For a given mode and block (thread partition), find minimum index
    //------------------------------------------------------------------
    int determine_block_offset(int mode, int block_index)
    {
        if (block_index < 0 || block_index >= static_cast<int>(partitions.size()))
            throw std::out_of_range("determine_block_offset: block_index " + std::to_string(block_index) +
                                   " is out of range [0, " + std::to_string(partitions.size() - 1) + "]");
        int start = (block_index > 0) ? partitions[block_index - 1] : 0;
        int end = partitions[block_index];
        int min = *(std::max_element(this->dims.begin(), this->dims.end()));

        for(int i = start; i < end; i++){
            int idx = get_mode_idx_alto(alto_tensor[i].linear_index, mode);
            if(idx < min) min = idx;
        }
        return min;
    }

    //------------------------------------------------------------------
    // For a given mode and block (thread partition), find maximum index
    //------------------------------------------------------------------
    int determine_block_limit(int mode, int block_index)
    {
        if (block_index < 0 || block_index >= static_cast<int>(partitions.size()))
            throw std::out_of_range("determine_block_limit: block_index " + std::to_string(block_index) +
                                   " is out of range [0, " + std::to_string(partitions.size() - 1) + "]");
        int start = (block_index > 0) ? partitions[block_index - 1] : 0;
        int end = partitions[block_index];
        int max = 0;

        for(int i = start; i < end; i++){
            int idx = get_mode_idx_alto(alto_tensor[i].linear_index, mode);
            if(idx > max) max = idx;
        }
        return max;
    }

    //------------------------------------------------------------------
    // Find which partition/block a given NNZ index belongs to
    //------------------------------------------------------------------
    int determine_block(int index)
    {
        int previous = 0;
        for(int i = 0; i < partitions.size(); i++){
            if(previous < index && index < partitions[i]) return i;
            previous = partitions[i];
        }
        return -1;
    }

    //------------------------------------------------------------------
    // Mark entries that lie on "fiber boundaries" spanning multiple blocks
    // These require atomics to prevent race conditions.
    //------------------------------------------------------------------
    void set_boundaries(int mode)
    {
        std::unordered_map<int, std::unordered_set<int>> fiber_blocks; 
        S boundary_mask = S(1) << this->num_bits;

        // Map: fiber index → set of blocks that contain it
        for (int i = 0; i < alto_tensor.size(); ++i) {
            S lin_idx = alto_tensor[i].linear_index;
            int idx = get_mode_idx_alto(lin_idx, mode);
            int block = determine_block(i);
            fiber_blocks[idx].insert(block);
        }

        // Mark boundary entries
        for (int i = 0; i < alto_tensor.size(); ++i) {
            S lin_idx = alto_tensor[i].linear_index;
            int idx = get_mode_idx_alto(lin_idx, mode);
            int block = determine_block(i);

            if (fiber_blocks[idx].size() > 1) {
                alto_tensor[i].linear_index |= boundary_mask;
            }
        }
    }

    //------------------------------------------------------------------
    // Reset boundary flag bit in linear index
    //------------------------------------------------------------------
    void reset_boundaries()
    {
        S mask = ~(S(1) << this->num_bits);
        for(int i = 0; i < alto_tensor.size(); i++){
            alto_tensor[i].linear_index &= mask;
        }
    }

    //------------------------------------------------------------------
    // Helper: Pick largest mode (bit allocation priority)
    //------------------------------------------------------------------
    int largest_mode(std::vector<int> bits)
    {
        int max_mode = 1; //Find the mode with the largest number of bits left
        bool all_zeros = bits[0] == 0;
        for(int i = 1; i < bits.size(); i++){
            if(all_zeros && bits[i] > 0) all_zeros = false;
            if((bits[i] > bits[max_mode - 1]) || (bits[i] == bits[max_mode - 1] && this->dims[i] > this->dims[max_mode - 1])) max_mode = i + 1;
        }
        return !all_zeros * max_mode;
    }

    //------------------------------------------------------------------
    // Create ALTO bitmasks for each mode (rows, cols, depth)
    // Assigns bits in interleaved order based on largest_mode()
    //------------------------------------------------------------------
    void create_masks()
    {
        std::vector<int> bits;
        for(int i = 0; i < this->rank; i++){
            bits.push_back(ceiling_log2(this->dims[i]));
        }

        for(int i = 0; i < this->rank; i++) bitmasks.push_back(S(0));

        S mask = S(1) << (this->num_bits - 1);
        int sum_of_bits = std::accumulate(bits.begin(), bits.end(), 0);
        while (sum_of_bits > 0 && mask != 0) {
            int l1 = largest_mode(bits);
            bits[l1 - 1]--;
            bitmasks[l1 - 1] |= mask;
            mask >>= 1;
            sum_of_bits--;
        }
    }

    //------------------------------------------------------------------
    // Translate (i,j,k,...) → ALTO linearized index using masks
    //------------------------------------------------------------------
    S translate_idx(const std::vector<int>& indices) 
    {
        if (static_cast<int>(indices.size()) != this->rank)
            throw std::invalid_argument(
                "translate_idx: indices size " + std::to_string(indices.size()) +
                " does not match tensor rank " + std::to_string(this->rank));
        if (std::accumulate(this->dims.begin(), this->dims.end(), 0) == 0) return 0;
    
        S val = 0;
        int local_indices[32]; // Max rank fallback
        for (int m = 0; m < this->rank; ++m) local_indices[m] = indices[m];

        for (int i = 0; i < this->num_bits; ++i) {
            S mask = static_cast<S>(1) << i;
            for(int m = 0; m < this->rank; m++){
                if(mask & bitmasks[m]) { 
                    if(local_indices[m] & 1ULL) val |= mask; 
                    local_indices[m] >>= 1;
                }
            }
        }
        return val;
    }

    //------------------------------------------------------------------
    // Build ALTO tensor from vector of NNZ entries
    //------------------------------------------------------------------
    void create_alto_vector(const std::vector<NNZ_Entry<T>>& tensor_vec)
    {
        alto_tensor.clear();
        alto_tensor.resize(tensor_vec.size());

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(tensor_vec.size()); i++) {
            ALTOEntry<T,S> entry;
            entry.linear_index = translate_idx(tensor_vec[i].coords);
            entry.value = tensor_vec[i].value;
            alto_tensor[i] = entry;
        }

        std::sort(alto_tensor.begin(), alto_tensor.end(),
            [](const ALTOEntry<T,S>& a, const ALTOEntry<T,S>& b) {
                return a.linear_index < b.linear_index;
            });
    }

public:
    //------------------------------------------------------------------
    // Default Constructor
    //------------------------------------------------------------------
    Alto_Tensor() : Tensor<T,S>()
    {
        bitmasks = {};
        alto_tensor = {};
        partitions = {};
        num_threads = 0;
    }

    //------------------------------------------------------------------
    // Constructor 1: from a pre-built vector of sparse nonzero entries
    //------------------------------------------------------------------
    Alto_Tensor(const std::vector<NNZ_Entry<T>>& entry_vec, std::vector<int> dims, int decomp_rank = 10) 
    : Tensor<T,S>(entry_vec, dims, decomp_rank)
    {
        // Ensure S is wide enough to hold all ALTO bits plus the boundary flag bit
        if (static_cast<int>(sizeof(S) * 8) <= this->num_bits)
            throw std::invalid_argument(
                "Alto_Tensor: index type S (" + std::to_string(sizeof(S) * 8) +
                " bits) is too narrow for " + std::to_string(this->num_bits) +
                " ALTO bits; use a wider type such as uint64_t or __uint128_t");

        create_masks();
        create_alto_vector(entry_vec);
        num_threads = ideal_threads();
        set_partitions();
    }

    //------------------------------------------------------------------
    // Constructor 2: read NNZ entries directly from a binary file,
    // writing straight into alto_tensor with no intermediate NNZ_Entry.
    // The binary format mirrors read_tensor_file_binary from utils.h:
    //   - No file header
    //   - Each entry: rank x int32_t coords (1-indexed), then 1 x T value
    //------------------------------------------------------------------
    Alto_Tensor(const std::string& filename, int64_t nnz,
                std::vector<int> dims, std::vector<std::vector<T>>& fmats, int decomp_rank = 10)
    : Tensor<T,S>({}, dims, fmats, decomp_rank)   // empty entry vec: factor matrices init'd, nnz_entries=0
    {
        // Fix nnz_entries — the empty entry vec left it at 0
        if (nnz <= 0)
            throw std::invalid_argument(
                "Alto_Tensor: nnz must be positive, got " + std::to_string(nnz));
        this->nnz_entries = static_cast<uint64_t>(nnz);

        // Ensure S is wide enough to hold all ALTO bits plus the boundary flag bit
        if (static_cast<int>(sizeof(S) * 8) <= this->num_bits)
            throw std::invalid_argument(
                "Alto_Tensor: index type S (" + std::to_string(sizeof(S) * 8) +
                " bits) is too narrow for " + std::to_string(this->num_bits) +
                " ALTO bits; use a wider type such as uint64_t or __uint128_t");

        create_masks();  // must come before read_binary_to_alto
        read_binary_to_alto(filename, nnz);  // read + encode + sort directly
        num_threads = ideal_threads();
        set_partitions();
        this->nnz_entries = nnz;
    }

    //------------------------------------------------------------------
    // Copy Constructor
    //------------------------------------------------------------------
    Alto_Tensor(const Alto_Tensor& other) : Tensor<T, S>(other) 
    {
        num_threads = other.num_threads;
        bitmasks = other.bitmasks;
        partitions = other.partitions;
        alto_tensor = other.alto_tensor;
    }

    //------------------------------------------------------------------
    // Copy Assignment Operator
    //------------------------------------------------------------------
    Alto_Tensor& operator=(const Alto_Tensor& other) 
    {
        if (this != &other) {
            Tensor<T, S>::operator=(other);
            num_threads = other.num_threads;
            bitmasks = other.bitmasks;
            partitions = other.partitions;
            alto_tensor = other.alto_tensor;
        }
        return *this;
    }

    //------------------------------------------------------------------
    // Extract coordinate from ALTO index
    //------------------------------------------------------------------
    int get_mode_idx_alto(S alto_idx, int mode) const
    {
        if (mode < 1 || mode > this->rank)
            throw std::out_of_range("get_mode_idx_alto: mode " + std::to_string(mode) +
                                   " is out of range [1, " + std::to_string(this->rank) + "]");

        S mask = bitmasks[mode - 1];

        int coord = 0, bit_pos = 0;
        int length = sizeof(S) * 8;

        for (int i = 0; i < length; ++i) {
            if ((mask >> i) & static_cast<S>(1)) {
                coord |= ((alto_idx >> i) & static_cast<S>(1)) << bit_pos;
                ++bit_pos;
            }
        }
        return coord;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    const std::vector<ALTOEntry<T,S>>& get_alto() const { return alto_tensor; }
    const std::vector<S> get_modemasks() const { return bitmasks;}
    
    //------------------------------------------------------------------
    // Debug utility: Print ALTO indices and decoded coordinates
    //------------------------------------------------------------------
    void debug_linear_indices(){
        for (auto& e : alto_tensor) {
            std::string sci = uint128_to_sci_string(e.linear_index,10);
            std::cout << "Index: " << sci;
            for(int i = 0; i < this->rank; i++){
                std::cout << ", mode = " << get_mode_idx_alto(e.linear_index, i + 1) << " ";
            }
            std::cout << ", val=" << e.value << "\n";
        }
    }

    //------------------------------------------------------------------
    // Create an Entry Vec using the Alto Tensor
    //------------------------------------------------------------------
    std::vector<NNZ_Entry<T>> create_entry_vec() const
    {
        std::vector<NNZ_Entry<T>> entries;
        entries.resize(this->nnz_entries);
        for(int i = 0; i < this->nnz_entries; i++){
            std::vector<int> indices = {};
            S lin_idx = alto_tensor[i].linear_index;
            for(int j = 0; j < this->rank; j++){
                indices.push_back(get_mode_idx_alto(lin_idx, j + 1));
            }
            entries[i].coords = indices;
            entries[i].value = alto_tensor[i].value;
        }

        return entries;
    }

    //------------------------------------------------------------------
    // Parallel MTTKRP (N-dimensional)
    //------------------------------------------------------------------
    std::vector<T> MTTKRP_Parallel(int target_mode) 
    {
        if (target_mode < 1 || target_mode > this->rank)
            throw std::out_of_range("MTTKRP_Alto_Parallel: target_mode " + std::to_string(target_mode) +
                                   " is out of range [1, " + std::to_string(this->rank) + "]");
        if (alto_tensor.empty())
            throw std::runtime_error("MTTKRP_Alto_Parallel: tensor has no nonzero entries");

        omp_set_num_threads(num_threads);
        const int num_modes = this->rank;
        const int R = this->factor_rank;
        
        // Total bits for bit-masking
        int total_shift = 0;
        for(int d = 0; d < num_modes; ++d) total_shift += ceiling_log2(this->dims[d]);
    
        // Heuristic for fiber reuse
        long long total_dim_prod = 1;
        for(int d = 0; d < num_modes; ++d) {
            if(d != (target_mode - 1)) total_dim_prod *= this->dims[d];
        }
        
        // Strategy selection
        if(this->nnz_entries / total_dim_prod < 4) {
            // --- Strategy 1: Atomic Accumulation ---
            set_boundaries(target_mode);
            S mask = S(1) << this->num_bits;
    
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = (thread_id > 0) ? partitions[thread_id-1] : 0;
                int end   = partitions[thread_id];
                
                // Local indices array to avoid repeated re-allocation
                int* local_indices = new int[num_modes]; 
    
                for (int m = start; m < end; ++m) {
                    S idx = alto_tensor[m].linear_index;
                    T val = alto_tensor[m].value;
                    bool boundary = (idx >> total_shift) & (S(1) != 0);
                    idx &= ~mask;
    
                    for(int d = 1; d <= num_modes; ++d) {
                        local_indices[d-1] = get_mode_idx_alto(idx, d);
                    }
    
                    int target_row = local_indices[target_mode - 1];
                    T* target_fmat = this->fmats[target_mode - 1].data();
    
                    for (int r = 0; r < R; ++r) {
                        T product = val;
                        for(int d = 1; d <= num_modes; ++d) {
                            if(d == target_mode) continue;
                            // Flattened access: matrix[row * rank + r]
                            product *= this->fmats[d-1][local_indices[d-1] * R + r];
                        }
    
                        if (boundary) {
                            #pragma omp atomic
                            target_fmat[target_row * R + r] += product;
                        } else {
                            target_fmat[target_row * R + r] += product;
                        }
                    }
                }
                delete[] local_indices;
            }
            reset_boundaries();
        } 
        else {
            // --- Strategy 2: Private Accumulation ---
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = (thread_id > 0) ? partitions[thread_id-1] : 0;
                int end   = partitions[thread_id];
    
                int block_offset = determine_block_offset(target_mode, thread_id);
                int block_limit  = determine_block_limit(target_mode, thread_id);
                int mode_range   = block_limit - block_offset + 1;
    
                // Thread-local accumulation buffer (Flattened)
                T* temp_buffer = new T[mode_range * R](); 
                int* local_indices = new int[num_modes];
    
                for (int m = start; m < end; ++m) {
                    S idx = alto_tensor[m].linear_index;
                    T val = alto_tensor[m].value;
    
                    for(int d = 1; d <= num_modes; ++d) {
                        local_indices[d-1] = get_mode_idx_alto(idx, d);
                    }
    
                    int target_row = local_indices[target_mode - 1];
    
                    for (int r = 0; r < R; ++r) {
                        T product = val;
                        for(int d = 1; d <= num_modes; ++d) {
                            if(d == target_mode) continue;
                            product *= this->fmats[d-1][local_indices[d-1] * R + r];
                        }
                        temp_buffer[(target_row - block_offset) * R + r] += product;
                    }
                }
    
                // Merge back using flattened indexing
                T* global_target_fmat = this->fmats[target_mode - 1].data();
                for (int i = 0; i < mode_range; ++i) {
                    for (int r = 0; r < R; ++r) {
                        #pragma omp atomic
                        global_target_fmat[(i + block_offset) * R + r] += temp_buffer[i * R + r];
                    }
                }
    
                delete[] temp_buffer;
                delete[] local_indices;
            }
        }
        return this->fmats[target_mode - 1];
    }
};

#endif



















