#ifndef BLCO_H
#define BLCO_H

#include <vector>
#include <utility>
#include <unordered_map>
#include <chrono>
#include <omp.h> 
#include <hip/hip_runtime.h>
#include <map>
#include "alto_impl.h"

//======================================================================
// BLCO_ENTRY: Represents one BLCO entry
// - index: linearized index
// - value: value at index
//======================================================================
template<typename T>
struct BLCO_ENTRY{
    uint64_t index;
    T value;
};

//======================================================================
// BLCO_BLOCK_CPU: Represents one block of a BLCO tensor stored on CPU
// - block: block identifier
// - size: number of NNZ in the block
// - entries: vector of BLCO entry
//======================================================================
template<typename T>
struct BLCO_BLOCK_CPU {
    int block;
    int size;
    std::vector<BLCO_ENTRY<T>> entries;
};

//======================================================================
// BLCO_BLOCK_GPU: Same structure but ready for GPU (device pointers)
//======================================================================
template<typename T>
struct BLCO_BLOCK_GPU {
    int block;
    int size;
    BLCO_ENTRY<T>* entries;
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
class Blco_Tensor : public Alto_Tensor<T, S>
{
protected:   
    bool blocks_needed; //If 64 bits isn't enough to represent all of the indexes properly                 
    std::vector<int> bit_widths; // Bit-widths needed for each mode
    std::vector<int> block_modes; // Modes whose indices are represented by the block number
    std::vector<uint64_t> bitmasks; // Masks defining bit placement for each mode
    std::vector<int> populated_blocks; //Indexes of all the different blocks which are populated
    std::vector<BLCO_BLOCK_CPU<T>> blco_tensor; // BLCO representation: vector of blocks

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

    void determine_bit_widths()
    {
        int bit_sum = 0;
        for(int i = 0; i < this->rank; i++){
            int bits_needed = ceiling_log2(this->dims[i]);
            bit_widths.push_back(bits_needed);
            bit_sum += bits_needed;
        }
        if(bit_sum > 64) blocks_needed = true;
        else blocks_needed = false;
    }

    //------------------------------------------------------------------
    // Create BLCO masks (unlike ALTO, assigns bits sequentially)
    //------------------------------------------------------------------
    void create_blco_masks()
    {
        if (std::accumulate(bit_widths.begin(), bit_widths.end(), 0) == 0) return;

        int total_bits = 0;
        for (int i = 0; i < this->rank; ++i) {
            uint64_t mask;
            int w = bit_widths[i];
            if (w >= 64) {
                mask = ~uint64_t(0); // all ones if the mode uses >=64 bits
            } else {
                mask = (uint64_t(1) << w) - 1;
            }
            bitmasks.push_back(mask);

            total_bits += w;
            if (total_bits > 64) {
                // mode i (0-based) crosses into block bits.
                // store mode index in 1-based convention if you use that elsewhere.
                block_modes.push_back(i + 1);
            }
        }
    }

    //------------------------------------------------------------------
    // Convert ALTO index → BLCO index (re-encode)
    // - Uses masks to place bits for row/col/depth
    // - If >64 bits are needed, overflow goes to higher bits
    //------------------------------------------------------------------
    S index_conversion(S alto_idx) 
    {
        int shift = 0;
        S val = 0;

        for (int i = 0; i < this->rank; ++i) {
            int idx = this->get_mode_idx_alto(alto_idx, i + 1);
            // bounds & mask the coordinate for this mode
            S indice = static_cast<S>(idx) & static_cast<S>(bitmasks[i]); // single mask

            // place the bits of indice starting at 'shift'
            val |= (indice << shift);

            // advance shift by this mode's bit width
            shift += bit_widths[i];
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
        // 1. Thread-local accumulation to avoid contention
        std::vector<std::unordered_map<int, BLCO_BLOCK_CPU<T>>> local_blocks(omp_get_max_threads());

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &local_map = local_blocks[tid];

            #pragma omp for nowait
            for (int i = 0; i < static_cast<int>(p1.first.size()); i++) {
                // Extract upper 64 bits as block number
                int block_num = static_cast<int>(p1.first[i] >> 64);
                // Extract lower 64 bits as the local index
                uint64_t blco_index = static_cast<uint64_t>(p1.first[i] & 0xFFFFFFFFFFFFFFFF);
                T val = p1.second[i];

                auto &block = local_map[block_num];
                block.block = block_num;
                block.entries.push_back({blco_index, val});
                block.size++;
            }
        }

        // 2. Merge into a global std::map to handle sorting and uniqueness
        std::map<int, BLCO_BLOCK_CPU<T>> global_merged_map;
        for (auto &local_map : local_blocks) {
            for (auto &[block_num, block] : local_map) {
                auto &dst = global_merged_map[block_num];
                if (dst.entries.empty()) {
                    dst.block = block_num;
                }
                dst.entries.insert(dst.entries.end(), block.entries.begin(), block.entries.end());
                dst.size += block.size;
            }
        }

        // 3. Transfer from sorted map to the final vector and update tracking
        blco_tensor.clear();
        populated_blocks.clear();
        blco_tensor.reserve(global_merged_map.size());
        populated_blocks.reserve(global_merged_map.size());

        for (auto &[block_num, block] : global_merged_map) {
            populated_blocks.push_back(block_num);
            blco_tensor.push_back(std::move(block));
        }
    }
    else {
        // If <= 64 bits, everything fits into a single block at index 0
        blco_tensor.clear();
        populated_blocks.clear();
        
        BLCO_BLOCK_CPU<T> b1;
        b1.block = 0;
        b1.size = static_cast<int>(p1.first.size());
        b1.entries.resize(p1.first.size());

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(p1.first.size()); i++) {
            b1.entries[i] = {static_cast<uint64_t>(p1.first[i]), p1.second[i]};
        }
        
        blco_tensor.push_back(std::move(b1));
        populated_blocks.push_back(0);
    }
}

public:
    //------------------------------------------------------------------
    // Constructor
    //------------------------------------------------------------------
    Blco_Tensor(const std::vector<NNZ_Entry<T>>& entry_vec, std::vector<int> dims, int decomp_rank = 10) : 
    Alto_Tensor<T,S>(entry_vec, dims, decomp_rank)
    {
        determine_bit_widths();
        create_blco_masks();
        create_blco_tensor();
    }

    //------------------------------------------------------------------
    // Decode BLCO index → coordinate (64-bit or 128 bit version)
    //------------------------------------------------------------------
    int get_mode_idx_blco(uint64_t blco_index, int mode, int block) const
    {
        // mode is 1-based; convert to 0-based
        int m = mode - 1;
        int shift = 0;
        for (int i = 0; i < m; ++i) shift += bit_widths[i];

        uint64_t mask = bitmasks[m];

        if (shift >= 64) {
            // all bits are in the block (upper part)
            int upper_shift = shift - 64;
            // extract bits_from_block starting at upper_shift
            uint64_t val = static_cast<uint64_t>((static_cast<uint64_t>(block) >> upper_shift) & mask);
            return static_cast<int>(val);
        } else {
            int bits_in_low = std::min(64 - shift, bit_widths[m]);
            uint64_t low_part = (blco_index >> shift) & ((bits_in_low == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits_in_low) - 1));

            if (bits_in_low == bit_widths[m]) {
                return static_cast<int>(low_part);
            } else {
                // need remaining bits from block
                int bits_from_block = bit_widths[m] - bits_in_low;
                uint64_t block_mask_small = (bits_from_block == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits_from_block) - 1);
                uint64_t upper_part = static_cast<uint64_t>(block & block_mask_small);
                uint64_t combined = (upper_part << bits_in_low) | low_part;
                return static_cast<int>(combined & mask);
            }
        }
    }


    //------------------------------------------------------------------
    // Find a block by ID (returns index or -1)
    //------------------------------------------------------------------
    int find_block(int target_block) const
    {
        for(int i = 0; i < blco_tensor.size(); ++i){
            if (blco_tensor[i].block == target_block) return i;
        }
        return -1;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    const std::vector<BLCO_BLOCK_CPU<T>>& get_blco() const {return blco_tensor;}
    std::vector<uint64_t> get_bitmasks() const {return bitmasks;}
    int get_num_blocks() const {return populated_blocks.size();}

    //------------------------------------------------------------------
    // Copy GPU result vector(mathematical vector not c++ vector) 
    // back into factor matrix (for MTTKRP output)
    //------------------------------------------------------------------
    void copy_vector_to_fmat(T* v1, int mode) const
    {
        for(int i = 0; i < this->dims[mode - 1] * this->factor_rank; i++){
            this->fmats[mode - 1][i] = v1[i];
        }
    }

    //------------------------------------------------------------------
    // Estimate number of distinct indexes per GPU wavefront
    // Helps in tuning GPU kernels
    //------------------------------------------------------------------
    int determine_indexes_per_wavefront(int mode) const
    {
        int max_indexes = 0;

        if(!blocks_needed){
            BLCO_BLOCK_CPU<T> b1 = blco_tensor[0];
            std::vector<int> indexes;
            for(int i = 0; i < b1.entries.size(); i++){
                if(i != 0 && i % 64 == 0){
                    if(indexes.size() > max_indexes) max_indexes = indexes.size();
                    indexes.clear();
                }
                int idx = get_mode_idx_blco(b1.entries[i].index, mode, 0);
                if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) 
                    indexes.push_back(idx);
            }
            return std::max(max_indexes, (int)indexes.size());
        }
        else{
            int block_num = -1;
            for (const auto &block : blco_tensor) {
                std::vector<int> indexes;
                block_num++;
                for (size_t i = 0; i < block.entries.size(); i++) {
                    if (i != 0 && i % 64 == 0) {
                        if (indexes.size() > max_indexes) max_indexes = indexes.size();
                        indexes.clear();
                    }
                    int idx = get_mode_idx_blco(block.entries[i].index, mode, block_num);
                    if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) {
                        indexes.push_back(idx);
                    }
                }
                if (indexes.size() > max_indexes) max_indexes = indexes.size();
            }
            return max_indexes;
        }
    }
    //------------------------------------------------------------------
    // Debug utilities: print decoded indices and values
    //------------------------------------------------------------------
    void debug_linear_indices()
    {
        int block_num;
        for (auto &b1 : blco_tensor) {
            block_num = b1.idx;
            for(int j = 0; j < b1.entries.size(); j++){
                for(int k = 0; k < this->rank; k++){
                    std::cout << "Mode " << k + 1 << " : " << get_mode_idx_blco(b1.entries[j].index, block_num, k + 1);
                }
                std::cout << ", val=" << b1.entries[j].value << "\n";
            }
        }
    }
};

#endif

