#ifndef ALTO_H
#define ALTO_H

#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <omp.h>
#include "tensor_impl.h"

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
//   - Encodes (i,j,k) coordinates into a *linearized ALTO index*
//     using adaptive bitmask assignment.
//   - Stores nonzeros in a sorted vector (by ALTO index).
//   - Partitions NNZs for parallel algorithms.
//   - Implements serial and parallel MTTKRP (Matricized Tensor × Khatri-Rao Product).
//======================================================================
template<typename T, typename S>
class Alto_Tensor_3D : public Tensor_3D<T, S>{
protected:
    int num_threads;                        // Number of threads for parallel MTTKRP
    int num_bits;                           // Total bits needed for encoding all modes
    S mode1_mask = 0;                       // Bitmask for mode 1 (rows)
    S mode2_mask = 0;                       // Bitmask for mode 2 (cols)
    S mode3_mask = 0;                       // Bitmask for mode 3 (depth)
    std::vector<int> partitions;            // Partition boundaries for NNZ distribution
    std::vector<ALTOEntry<T,S>> alto_tensor;// Compact ALTO representation of the tensor

    //------------------------------------------------------------------
    // Choose "ideal" number of threads based on NNZ size
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
        int start = (block_index > 0) ? partitions[block_index - 1] : 0;
        int end = partitions[block_index];
        int min = std::max({this->rows, this->cols, this->depth});

        for(int i = start; i < end; i++){
            int idx = get_mode_idx(alto_tensor[i].linear_index, mode);
            if(idx < min) min = idx;
        }
        return min;
    }

    //------------------------------------------------------------------
    // For a given mode and block (thread partition), find maximum index
    //------------------------------------------------------------------
    int determine_block_limit(int mode, int block_index)
    {
        int start = (block_index > 0) ? partitions[block_index - 1] : 0;
        int end = partitions[block_index];
        int max = 0;

        for(int i = start; i < end; i++){
            int idx = get_mode_idx(alto_tensor[i].linear_index, mode);
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
        S boundary_mask = S(1) << num_bits;

        // Map: fiber index → set of blocks that contain it
        for (int i = 0; i < alto_tensor.size(); ++i) {
            S lin_idx = alto_tensor[i].linear_index;
            int idx = get_mode_idx(lin_idx, mode);
            int block = determine_block(i);
            fiber_blocks[idx].insert(block);
        }

        // Mark boundary entries
        for (int i = 0; i < alto_tensor.size(); ++i) {
            S lin_idx = alto_tensor[i].linear_index;
            int idx = get_mode_idx(lin_idx, mode);
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
        S mask = ~(S(1) << num_bits);
        for(int i = 0; i < alto_tensor.size(); i++){
            alto_tensor[i].linear_index &= mask;
        }
    }

    //------------------------------------------------------------------
    // Helper: Pick largest mode (bit allocation priority)
    //------------------------------------------------------------------
    int largest_mode(int mode_1_bits, int mode_2_bits, int mode_3_bits)
    {
        if(mode_1_bits == 0 && mode_2_bits == 0 && mode_3_bits == 0) return 0;

        if((mode_1_bits > mode_2_bits || (mode_1_bits == mode_2_bits && this->rows >= this->cols))
        && (mode_1_bits > mode_3_bits || (mode_1_bits == mode_3_bits && this->rows >= this->depth))) return 1;

        if(mode_2_bits > mode_3_bits || (mode_2_bits == mode_3_bits && this->cols >= this->depth)) return 2;

        return 3;
    }

    //------------------------------------------------------------------
    // Create ALTO bitmasks for each mode (rows, cols, depth)
    // Assigns bits in interleaved order based on largest_mode()
    //------------------------------------------------------------------
    void create_masks()
    {
        if (this->rows == 0 || this->cols == 0 || this->depth == 0) return;

        int m1_bits = ceiling_log2(this->rows);
        int m2_bits = ceiling_log2(this->cols);
        int m3_bits = ceiling_log2(this->depth); 

        int shift = m1_bits + m2_bits + m3_bits;
        S mask = S(1) << (shift-1);

        while ((m1_bits || m2_bits || m3_bits) && mask != 0) {
            int l1 = largest_mode(m1_bits,m2_bits,m3_bits);

            if (l1 == 0) break; 
            if (l1 == 1) { mode1_mask |= mask; m1_bits--; }
            else if (l1 == 2) { mode2_mask |= mask; m2_bits--; }
            else              { mode3_mask |= mask; m3_bits--; }

            mask >>= 1;
        }
    }

    //------------------------------------------------------------------
    // Translate (i,j,k) → ALTO linearized index using masks
    //------------------------------------------------------------------
    S translate_idx(int r, int c, int d) {
        if (this->rows == 0 || this->cols == 0 || this->depth == 0) return 0;
    
        S val = 0;
        for (int i = 0; i < num_bits; ++i) {
            S mask = static_cast<S>(1) << i;
            if (mask & mode1_mask) { if(r & 1ULL) val |= mask; r >>= 1; }
            else if (mask & mode2_mask) { if(c & 1ULL) val |= mask; c >>= 1; }
            else if (mask & mode3_mask) { if(d & 1ULL) val |= mask; d >>= 1; }
        }
        return val;
    }

    //------------------------------------------------------------------
    // Build ALTO tensor from dense array
    //------------------------------------------------------------------
    void create_alto_array(T*** tensor_arr)
    {
        alto_tensor.clear();

        for (int i = 0; i < this->rows; ++i)
            for (int j = 0; j < this->cols; ++j)
                for (int k = 0; k < this->depth; ++k) {
                    T val = tensor_arr[i][j][k];
                    if (val != 0) {
                        this->nnz_entries++;
                        ALTOEntry<T,S> entry;
                        entry.linear_index = translate_idx(i,j,k);
                        entry.value = val;
                        alto_tensor.push_back(entry);
                    }
                }

        // Sort entries by ALTO index for compact storage
        std::sort(alto_tensor.begin(), alto_tensor.end(),
            [](const ALTOEntry<T,S>& a, const ALTOEntry<T,S>& b) {
                return a.linear_index < b.linear_index;
            });
    }

    //------------------------------------------------------------------
    // Build ALTO tensor from vector of NNZ entries
    //------------------------------------------------------------------
    void create_alto_vector(const std::vector<NNZ_Entry<T>>& tensor_vec)
    {
        alto_tensor.clear();
        alto_tensor.resize(tensor_vec.size());

        #pragma omp parallel for
        for (int s = 0; s < static_cast<int>(tensor_vec.size()); s++) {
            ALTOEntry<T,S> entry;
            entry.linear_index = translate_idx(tensor_vec[s].i, tensor_vec[s].j, tensor_vec[s].k);
            entry.value = tensor_vec[s].value;
            alto_tensor[s] = entry;
        }

        std::sort(alto_tensor.begin(), alto_tensor.end(),
            [](const ALTOEntry<T,S>& a, const ALTOEntry<T,S>& b) {
                return a.linear_index < b.linear_index;
            });
    }

public:
    //------------------------------------------------------------------
    // Constructors
    //------------------------------------------------------------------
    Alto_Tensor_3D(T*** array, int r, int c, int d) : Tensor_3D<T,S>(array, r, c, d)
    {
        num_bits = ceiling_log2(r) + ceiling_log2(c) + ceiling_log2(d);
        create_masks();
        create_alto_array(array);
        num_threads = ideal_threads();
        set_partitions();
    }

    Alto_Tensor_3D(const std::vector<NNZ_Entry<T>>& entry_vec, int r, int c, int d) 
        : Tensor_3D<T,S>(entry_vec, r, c, d)
    {
        num_bits = ceiling_log2(r) + ceiling_log2(c) + ceiling_log2(d);
        create_masks();
        create_alto_vector(entry_vec);
        num_threads = ideal_threads();
        set_partitions();
    }

    //------------------------------------------------------------------
    // Extract coordinate from ALTO index
    //------------------------------------------------------------------
    int get_mode_idx(S alto_idx, int mode) 
    {
        S mask;
        switch (mode) {
            case 1: mask = mode1_mask; break;
            case 2: mask = mode2_mask; break;
            case 3: mask = mode3_mask; break;
            default: throw std::invalid_argument("Invalid mode (must be 1,2,3)");
        }

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
    virtual std::vector<S> get_modemasks() const { return {mode1_mask, mode2_mask, mode3_mask}; }
    
    //------------------------------------------------------------------
    // Serial MTTKRP
    //------------------------------------------------------------------
    T** MTTKRP_Alto(int mode)
    {
        if(mode != 1 && mode != 2 && mode != 3) return nullptr;

        for(auto& entry : alto_tensor){
            int i = get_mode_idx(entry.linear_index, 1);
            int j = get_mode_idx(entry.linear_index, 2);
            int k = get_mode_idx(entry.linear_index, 3);

            if(mode == 1)
                for(int r = 0; r < this->rank; r++)
                    this->mode_1_fmat[i][r] += entry.value * this->mode_2_fmat[j][r] * this->mode_3_fmat[k][r];
            else if(mode == 2)
                for(int r = 0; r < this->rank; r++)
                    this->mode_2_fmat[j][r] += entry.value * this->mode_1_fmat[i][r] * this->mode_3_fmat[k][r];
            else
                for(int r = 0; r < this->rank; r++)
                    this->mode_3_fmat[k][r] += entry.value * this->mode_1_fmat[i][r] * this->mode_2_fmat[j][r];
        }

        return (mode == 1) ? this->mode_1_fmat :
               (mode == 2) ? this->mode_2_fmat : this->mode_3_fmat;
    }

    //------------------------------------------------------------------
    // Parallel MTTKRP
    // Two strategies:
    //   - If fibers overlap across blocks → atomics
    //   - Else use private accumulation per thread then merge
    //------------------------------------------------------------------
    T** MTTKRP_Alto_Parallel(int mode) {
        omp_set_num_threads(num_threads);
        int shift = ceiling_log2(this->rows) + ceiling_log2(this->cols) + ceiling_log2(this->depth);

        int num_fibers = (mode == 1) ? this->cols * this->depth :
                        (mode == 2) ? this->rows * this->depth :
                                      this->rows * this->cols;

        if(this->nnz_entries/num_fibers < 4){
            // Fiber reuse is low → use atomics
            set_boundaries(mode);
            S mask = S(1) << num_bits;

            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = (thread_id > 0) ? partitions[thread_id-1] : 0;
                int end   = partitions[thread_id];

                for (int m = start; m < end; ++m) {
                    S idx = alto_tensor[m].linear_index;
                    T val = alto_tensor[m].value;
                    bool boundary = (idx >> shift) & S(1) != 0;
                    idx &= ~mask;

                    int i = get_mode_idx(idx, 1);
                    int j = get_mode_idx(idx, 2);
                    int k = get_mode_idx(idx, 3);

                    for (int r = 0; r < this->rank; ++r) {
                        if (mode == 1) {
                            if (boundary){ 
                                #pragma omp atomic
                                this->mode_1_fmat[i][r] += val * this->mode_2_fmat[j][r] * this->mode_3_fmat[k][r];
                            }
                            else this->mode_1_fmat[i][r] += val * this->mode_2_fmat[j][r] * this->mode_3_fmat[k][r];
                        }
                        else if (mode == 2) {
                            if (boundary){ 
                                #pragma omp atomic
                                this->mode_2_fmat[j][r] += val * this->mode_1_fmat[i][r] * this->mode_3_fmat[k][r]; 
                            }
                            else this->mode_2_fmat[j][r] += val * this->mode_1_fmat[i][r] * this->mode_3_fmat[k][r];
                        }
                        else {
                            if (boundary){ 
                                #pragma omp atomic
                                this->mode_3_fmat[k][r] += val * this->mode_1_fmat[i][r] * this->mode_2_fmat[j][r]; 
                            }
                            else this->mode_3_fmat[k][r] += val * this->mode_1_fmat[i][r] * this->mode_2_fmat[j][r];
                        }
                    }
                }
            }
            reset_boundaries();
        }
        else{
            // Fiber reuse is high → use private accumulation then merge
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int start = (thread_id > 0) ? partitions[thread_id-1] : 0;
                int end   = partitions[thread_id];

                int block_offset = determine_block_offset(mode, thread_id);
                int block_limit  = determine_block_limit(mode, thread_id);
                int mode_range   = block_limit - block_offset + 1;

                // Thread-local accumulation buffer
                T** temp_arr = new T*[mode_range];
                for (int i = 0; i < mode_range; ++i)
                    temp_arr[i] = new T[this->rank](); // zero-initialize

                for (int m = start; m < end; ++m) {
                    S idx = alto_tensor[m].linear_index;
                    T val = alto_tensor[m].value;

                    int i = get_mode_idx(idx, 1);
                    int j = get_mode_idx(idx, 2);
                    int k = get_mode_idx(idx, 3);

                    for (int r = 0; r < this->rank; ++r) {
                        if (mode == 1) temp_arr[i - block_offset][r] += val * this->mode_2_fmat[j][r] * this->mode_3_fmat[k][r];
                        else if (mode == 2) temp_arr[j - block_offset][r] += val * this->mode_1_fmat[i][r] * this->mode_3_fmat[k][r];
                        else temp_arr[k - block_offset][r] += val * this->mode_1_fmat[i][r] * this->mode_2_fmat[j][r];
                    }
                }

                // Merge results back into global factor matrices
                for (int i = 0; i < mode_range; ++i)
                    for (int j = 0; j < this->rank; ++j)
                        if (mode == 1) this->mode_1_fmat[i + block_offset][j] += temp_arr[i][j];
                        else if (mode == 2) this->mode_2_fmat[i + block_offset][j] += temp_arr[i][j];  
                        else this->mode_3_fmat[i + block_offset][j] += temp_arr[i][j];

                this->delete_matrix(temp_arr, mode_range, this->rank); 
            }
        }
        return (mode == 1) ? this->mode_1_fmat :
               (mode == 2) ? this->mode_2_fmat : this->mode_3_fmat;
    }
    
    //------------------------------------------------------------------
    // Debug utility: Print ALTO indices and decoded coordinates
    //------------------------------------------------------------------
    virtual void debug_linear_indices(){
        for (auto& e : alto_tensor) {
            std::string sci = uint128_to_sci_string(e.linear_index,10);
            std::cout << "Index: " << sci 
                      << ", i=" << get_mode_idx(e.linear_index, 1)
                      << ", j=" << get_mode_idx(e.linear_index, 2)
                      << ", k=" << get_mode_idx(e.linear_index, 3)
                      << ", val=" << e.value << "\n";
        }
    }    
};

#endif



















