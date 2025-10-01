#ifndef BLCO_H
#define BLCO_H

#include <vector>
#include <utility>
#include <unordered_map>
#include <chrono>
#include <omp.h> 
#include <hip/hip_runtime.h>
#include "alto_impl.h"

//======================================================================
// BLCO_BLOCK_CPU: Represents one block of a BLCO tensor stored on CPU
// - block: block identifier
// - size: number of NNZ in the block
// - indexes: vector of encoded BLCO indices
// - values: vector of corresponding NNZ values
//======================================================================
template<typename T>
struct BLCO_BLOCK_CPU {
    int block;
    int size;
    std::vector<uint64_t> indexes;
    std::vector<T> values;
};

//======================================================================
// BLCO_BLOCK_GPU: Same structure but ready for GPU (device pointers)
//======================================================================
template<typename T>
struct BLCO_BLOCK_GPU {
    int block;
    int size;
    uint64_t* indexes;
    T* values;
};


//======================================================================
// BLCO_Tensor_3D
//======================================================================
// Blocked Linearized Coordinate (BLCO) tensor format for 3D tensors
// Extends ALTO_Tensor_3D<T,S> with:
//   - Index re-encoding for GPU efficiency (shift/mask rather than scatter)
//   - Blocking step: splits tensor into manageable chunks
//   - Support for both 64-bit and 128-bit indexing
//   - Functions for extracting coordinates from BLCO indices
//   - Debug utilities
//
// Notes: 
//   * Assumes dimensions < 2^31 (due to use of int)
//   * Optimized for GPU out-of-memory streaming
//======================================================================
template<typename T, typename S>
class BLCO_Tensor_3D : public Alto_Tensor_3D<T, S>
{
protected:
    // Bit-widths needed for each mode
    int row_bits;
    int col_bits;
    int depth_bits;

    // BLCO representation: vector of blocks
    std::vector<BLCO_BLOCK_CPU<T>> blco_tensor;

    // Masks defining bit placement for each mode
    uint64_t m1_blco_mask = 0; // rows
    uint64_t m2_blco_mask = 0; // cols
    uint64_t m3_blco_mask = 0; // depth


    //------------------------------------------------------------------
    // Utility: sort two parallel vectors by the first one
    //------------------------------------------------------------------
    void sort_pair_by_first(std::pair<std::vector<S>, std::vector<T>>& p) 
    {
        std::vector<size_t> indices(p.first.size());
        std::iota(indices.begin(), indices.end(), 0);  

        std::sort(indices.begin(), indices.end(),
                  [&](size_t i, size_t j) { return p.first[i] < p.first[j]; });

        std::vector<S> sorted_first;
        std::vector<T> sorted_second;
        for (size_t i : indices) {
            sorted_first.push_back(p.first[i]);
            sorted_second.push_back(p.second[i]);
        }

        p.first = std::move(sorted_first);
        p.second = std::move(sorted_second);
    }

    //------------------------------------------------------------------
    // Create BLCO masks (unlike ALTO, assigns bits sequentially)
    //------------------------------------------------------------------
    void create_blco_masks()
    {
        if (this->rows == 0 || this->cols == 0 || this->depth == 0) return;

        int m1 = this->rows, m2 = this->cols, m3 = this->depth;
        S mask = 1ULL;

        // Assign bits: row bits first, then col bits, then depth bits
        while (m1 != 0 || m2 != 0 || m3 != 0) {
            if(m1 != 0){
                m1_blco_mask |= mask;
                m1 >>= 1;
            }
            else if(m2 != 0){
                m2_blco_mask |= mask;
                m2 >>= 1;
            }
            else if(m3 != 0){
                m3_blco_mask |= mask;
                m3 >>= 1;
            }
            mask <<= 1;
        }
    }

    //------------------------------------------------------------------
    // Convert ALTO index → BLCO index (re-encode)
    // - Uses masks to place bits for row/col/depth
    // - If >64 bits are needed, overflow goes to higher bits
    //------------------------------------------------------------------
    S index_conversion(S alto_idx) 
    {
        int r = this->get_mode_idx(alto_idx,1);
        int c = this->get_mode_idx(alto_idx,2);
        int d = this->get_mode_idx(alto_idx,3);

        S val = 0;
        int extra_bits = this->num_bits - 64;

        // Encode first 64 bits
        for (int i = 0; i < 64; ++i) {
            S mask = static_cast<S>(1) << i;
            if (mask & m1_blco_mask) { if(r & 1ULL) val |= mask; r >>= 1; }
            else if (mask & m2_blco_mask) { if(c & 1ULL) val |= mask; c >>= 1; }
            else if (mask & m3_blco_mask) { if(d & 1ULL) val |= mask; d >>= 1; }
        }

        // Encode overflow bits (depth only, if needed)
        if(extra_bits > 0){
            for (int i = 0; i < extra_bits; ++i) {
                S mask = static_cast<S>(1) << (64 + i);
                if(d & 1ULL) val |= mask;
                d >>= 1;
            }
        }
        return val;
    }

    //------------------------------------------------------------------
    // Build intermediate tensor representation (BLCO indices + values)
    //------------------------------------------------------------------
    std::pair<std::vector<S>, std::vector<T>> create_intermediate_tensor()
    {
        std::pair<std::vector<S>, std::vector<T>> p1;
        size_t n = this->alto_tensor.size();
        p1.first.resize(n);
        p1.second.resize(n);

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); i++) {
            p1.first[i]  = index_conversion(this->alto_tensor[i].linear_index);
            p1.second[i] = this->alto_tensor[i].value;
        }
        return p1;
    }

    //------------------------------------------------------------------
    // Create BLCO tensor (split into blocks if >64 bits)
    //------------------------------------------------------------------
    void create_blco_tensor()
    {
        auto p1 = create_intermediate_tensor();

        int limit_bits = 0;
        if constexpr (std::is_same_v<S, __uint128_t>) limit_bits = 1;

        if (limit_bits) {
            // If >64 bits needed, split into blocks by upper bits
            std::vector<std::unordered_map<int, BLCO_BLOCK_CPU<T>>> local_blocks(omp_get_max_threads());

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto &local_map = local_blocks[tid];

                #pragma omp for nowait
                for (int i = 0; i < static_cast<int>(p1.first.size()); i++) {
                    int block_num = static_cast<int>(p1.first[i] >> 64);
                    __uint128_t blco_index = p1.first[i] & limit;
                    T val = p1.second[i];

                    auto &block = local_map[block_num];
                    block.block = block_num;
                    block.indexes.push_back(blco_index);
                    block.values.push_back(val);
                    block.size++;
                }
            }

            // Merge thread-local maps into global blco_tensor
            for (auto &local_map : local_blocks) {
                for (auto &[block_num, block] : local_map) {
                    int result = find_block(block_num);
                    if (result == -1) {
                        blco_tensor.push_back(std::move(block));
                    } else {
                        auto &dst = blco_tensor[result];
                        dst.indexes.insert(dst.indexes.end(), block.indexes.begin(), block.indexes.end());
                        dst.values.insert(dst.values.end(), block.values.begin(), block.values.end());
                        dst.size += block.size;
                    }
                }
            }
        }
        else {
            // If <=64 bits, everything fits in one block
            BLCO_BLOCK_CPU<T> b1;
            b1.block = 0;
            b1.size = p1.first.size();
            b1.indexes.resize(p1.first.size());
            b1.values.resize(p1.first.size());

            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(p1.first.size()); i++) {
                b1.indexes[i] = static_cast<uint64_t>(p1.first[i]);
                b1.values[i]  = p1.second[i];
            }
            blco_tensor.push_back(std::move(b1));
        }
    }

public:
    //------------------------------------------------------------------
    // Constructors: build from dense array or NNZ list
    //------------------------------------------------------------------
    BLCO_Tensor_3D(T*** array, int r, int c, int d) : Alto_Tensor_3D<T,S>(array, r, c, d)
    {
        row_bits = ceiling_log2(r); col_bits = ceiling_log2(c); depth_bits = ceiling_log2(d);
        create_blco_masks();
        create_blco_tensor();
    }

    BLCO_Tensor_3D(const std::vector<NNZ_Entry<T>>& entry_vec, int r, int c, int d) 
        : Alto_Tensor_3D<T,S>(entry_vec, r, c, d)
    {
        row_bits = ceiling_log2(r); col_bits = ceiling_log2(c); depth_bits = ceiling_log2(d);
        create_blco_masks();
        create_blco_tensor();
    }

    //------------------------------------------------------------------
    // Decode BLCO index → coordinate (64-bit version)
    //------------------------------------------------------------------
    int get_mode_idx_blco_64_bit(uint64_t blco_index, int mode) const
    {
        uint64_t mask = (mode == 1) ? m1_blco_mask :
                        (mode == 2) ? m2_blco_mask : m3_blco_mask;

        blco_index &= mask;
        if(mode == 2) blco_index >>= row_bits;
        else if(mode == 3) blco_index >>= row_bits + col_bits;
        return static_cast<int>(blco_index);
    }

    //------------------------------------------------------------------
    // Decode BLCO index → coordinate (128-bit, multi-block version)
    //------------------------------------------------------------------
    int get_mode_idx_blco_128_bit(uint64_t blco_index, int block, int mode) const
    {
        uint64_t mask = (mode == 1) ? m1_blco_mask :
                        (mode == 2) ? m2_blco_mask : m3_blco_mask;

        uint64_t masked_idx = blco_index & mask;
        if (mode == 2) masked_idx >>= row_bits;
        else if (mode == 3) masked_idx >>= row_bits + col_bits;

        if(mode == 3 && row_bits + col_bits + depth_bits > 64){
            masked_idx |= block << (-1 * (row_bits + col_bits - 64));
        }
        return masked_idx;
    }

    //------------------------------------------------------------------
    // Find a block by ID (returns index or -1)
    //------------------------------------------------------------------
    int find_block(int target_block)
    {
        for(int i = 0; i < blco_tensor.size(); i++)
            if(blco_tensor[i].block == target_block) return i;
        return -1;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    const std::vector<BLCO_BLOCK_CPU<T>>& get_blco() const { return blco_tensor; }
    std::vector<uint64_t> get_blco_masks() const { return {m1_blco_mask, m2_blco_mask, m3_blco_mask}; }
    std::vector<int> get_bit_offsets() const { return {row_bits, col_bits, depth_bits}; }

    //------------------------------------------------------------------
    // Copy GPU result vector back into factor matrix (for MTTKRP output)
    //------------------------------------------------------------------
    void copy_vector_to_fmat(T* v1, int mode) const
    {
        if(mode == 1) {
            for(int i = 0; i < this->rows * this->rank; i++)
                this->mode_1_fmat[i/this->rank][i % this->rank] = v1[i];
        }
        else if(mode == 2) {
            for(int i = 0; i < this->cols * this->rank; i++)
                this->mode_2_fmat[i/this->rank][i % this->rank] = v1[i];
        }
        else {
            for(int i = 0; i < this->depth * this->rank; i++)
                this->mode_3_fmat[i/this->rank][i % this->rank] = v1[i];
        }
    }

    //------------------------------------------------------------------
    // Estimate number of distinct indexes per GPU wavefront
    // Helps in tuning GPU kernels
    //------------------------------------------------------------------
    int determine_indexes_per_wavefront_64_bit(int mode) const
    {
        BLCO_BLOCK_CPU<T> b1 = blco_tensor[0];
        std::vector<int> indexes;
        int max_indexes = 0;
        for(int i = 0; i < b1.indexes.size(); i++){
            if(i != 0 && i % 64 == 0){
                if(indexes.size() > max_indexes) max_indexes = indexes.size();
                indexes.clear();
            }
            int idx = get_mode_idx_blco_64_bit(b1.indexes[i], mode);
            if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) 
                indexes.push_back(idx);
        }
        return std::max(max_indexes, (int)indexes.size());
    }

    int determine_indexes_per_wavefront_128_bit(int mode) const {
        int max_indexes = 0;
        int block_num = -1;
        for (const auto &block : blco_tensor) {
            std::vector<int> indexes;
            block_num++;
            for (size_t i = 0; i < block.indexes.size(); i++) {
                if (i != 0 && i % 64 == 0) {
                    if (indexes.size() > max_indexes) max_indexes = indexes.size();
                    indexes.clear();
                }
                int idx = get_mode_idx_blco_128_bit(block.indexes[i],block_num,mode);
                if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) {
                    indexes.push_back(idx);
                }
            }
            if (indexes.size() > max_indexes) max_indexes = indexes.size();
        }
        return max_indexes;
    }

    //------------------------------------------------------------------
    // Debug utilities: print decoded indices and values
    //------------------------------------------------------------------
    void debug_linear_indices_64_bit()
    {
        for (auto &b1 : blco_tensor) {
            for(int j = 0; j < b1.indexes.size(); j++){
                std::cout << "Block: " << b1.block
                    << ", Index: " << b1.indexes[j]
                    << ", i=" << get_mode_idx_blco_64_bit(b1.indexes[j], 1)
                    << ", j=" << get_mode_idx_blco_64_bit(b1.indexes[j], 2)
                    << ", k=" << get_mode_idx_blco_64_bit(b1.indexes[j], 3)
                    << ", val=" << b1.values[j] << "\n";
            }
        }
    }

    void debug_linear_indices_128_bit()
    {
        int block_num = -1;
        for (auto &b1 : blco_tensor) {
            block_num++;
            for(int j = 0; j < b1.indexes.size(); j++){
                std::cout << "Block: " << b1.block
                    << ", Index: " << b1.indexes[j]
                    << ", i=" << get_mode_idx_blco_128_bit(b1.indexes[j], block_num, 1)
                    << ", j=" << get_mode_idx_blco_128_bit(b1.indexes[j], block_num, 2)
                    << ", k=" << get_mode_idx_blco_128_bit(b1.indexes[j], block_num, 3)
                    << ", val=" << b1.values[j] << "\n";
            }
        }
    }
};

#endif

