#ifndef BLCO_H
#define BLCO_H

#include <vector>
#include <utility>
#include <unordered_map>
#include <chrono>
#include <omp.h> 
#include <hip/hip_runtime.h>
#include "alto_impl.h"

template<typename T>
struct BLCO_BLOCK_CPU {
    int block;
    int size;
    std::vector<uint64_t> indexes;
    std::vector<T> values;
};

template<typename T>
struct BLCO_BLOCK_GPU {
    int block;
    int size;
    uint64_t* indexes;
    T* values;
};


template<typename T, typename S>
class BLCO_Tensor_3D : public Alto_Tensor_3D<T, S>
{
//Note does not support tensors with a dimension larger than 2,147,483,647 or the largest value
//for a signed int
protected:
    int row_bits;
    int col_bits;
    int depth_bits;
    std::vector<BLCO_BLOCK_CPU<T>> blco_tensor;
    uint64_t m1_blco_mask = 0; //m1 means rows
    uint64_t m2_blco_mask = 0; //m2 means cols
    uint64_t m3_blco_mask = 0; //m3 means depth

    //Function used to sort two vectors
    void sort_pair_by_first(std::pair<std::vector<S>, std::vector<T>>& p) 
    {
        std::vector<size_t> indices(p.first.size());
        std::iota(indices.begin(), indices.end(), 0);  // 0, 1, 2, ...

        // Sort indices based on the values in p.first
        std::sort(indices.begin(), indices.end(),
                [&](size_t i, size_t j) { return p.first[i] < p.first[j]; });

        // Create sorted copies
        std::vector<S> sorted_first;
        std::vector<T> sorted_second;

        for (size_t i : indices) {
            sorted_first.push_back(p.first[i]);
            sorted_second.push_back(p.second[i]);
        }

        // Replace original vectors
        p.first = std::move(sorted_first);
        p.second = std::move(sorted_second);
    }

    //Function to help create masks for BLCO format
    void create_blco_masks()
    {
        if (this->rows == 0 || this->cols == 0 || this->depth == 0) return;

        int m1 = this->rows, m2 = this->cols, m3 = this->depth;

        S mask = 1ULL;

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
            else{
                break;
            }
            mask <<= 1;
        }
    }

    //Converts ALTO index to BLCO index
    S index_conversion(S alto_idx) 
    {
        int r = this->get_mode_idx(alto_idx,1);
        int c = this->get_mode_idx(alto_idx,2);
        int d = this->get_mode_idx(alto_idx,3);

        S val = 0;
        int extra_bits = this->num_bits - 64;

        for (int i = 0; i < 64; ++i) {
            S mask = static_cast<S>(1) << i;
            if (mask & m1_blco_mask) {
                if(r & 1ULL) val |= mask;
                r >>= 1;
            }
            else if (mask & m2_blco_mask) {
                if(c & 1ULL) val |= mask;
                c >>= 1;
            }
            else if (mask & m3_blco_mask) {
                if(d & 1ULL) val |= mask;
                d >>= 1;
            }
        }

        if(extra_bits > 0){
            for (int i = 0; i < extra_bits; ++i) {
                S mask = static_cast<S>(1) << (64 + i);
                if(d & 1ULL) val |= mask;
                d >>= 1;
            }
        }

        return val;
    }

    //Converts the alto tensor into an intermediate tensor representation
    std::pair<std::vector<S>, std::vector<T>> create_intermediate_tensor()
    {
        std::pair<std::vector<S>, std::vector<T>> p1;
        size_t n = this->alto_tensor.size();

        // Preallocate
        p1.first.resize(n);
        p1.second.resize(n);

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); i++) {
            p1.first[i]  = index_conversion(this->alto_tensor[i].linear_index);
            p1.second[i] = this->alto_tensor[i].value;
        }

        return p1;
    }

    //Creates the BLCO tensor and fills the block pointer array
    void create_blco_tensor()
    {
        auto p1 = create_intermediate_tensor();

        int limit_bits = 0;
        if constexpr (std::is_same_v<S, __uint128_t>) limit_bits = 1;

        if (limit_bits) {
            // Thread-local accumulation
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

            // Merge into shared blco_tensor
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
            // Everything goes into one block
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
    BLCO_Tensor_3D(T*** array, int r, int c, int d) : Alto_Tensor_3D<T,S>(array, r, c, d)
    {
        row_bits = ceiling_log2(r); col_bits = ceiling_log2(c); depth_bits = ceiling_log2(d);
        create_blco_masks();
        create_blco_tensor();
    }

    BLCO_Tensor_3D(const std::vector<NNZ_Entry<T>>& entry_vec, int r, int c, int d) : Alto_Tensor_3D<T,S>(entry_vec, r, c, d)
    {
        row_bits = ceiling_log2(r); col_bits = ceiling_log2(c); depth_bits = ceiling_log2(d);
        create_blco_masks();
        create_blco_tensor();
    }

    //Get that idx for any given mode based on the BLCO index if only 64 bits are needed
    int get_mode_idx_blco_64_bit(uint64_t blco_index, int mode) const
    {
        uint64_t mask = (mode == 1) ? m1_blco_mask :
                        (mode == 2) ? m2_blco_mask : m3_blco_mask;

        blco_index &= mask;

        if(mode == 2) blco_index >>= row_bits;
            else if(mode == 3) blco_index >>= row_bits + col_bits;
            
        return static_cast<int>(blco_index);
    }

    //Get that idx for any given mode based on the BLCO index if 128 bits are needed
    int get_mode_idx_blco_128_bit(uint64_t blco_index, int block, int mode) const
    {
        uint64_t mask = (mode == 1) ? m1_blco_mask :
                        (mode == 2) ? m2_blco_mask :
                                        m3_blco_mask;

        uint64_t masked_idx = blco_index & mask;

        if (mode == 2) {
            masked_idx >>= row_bits;
        } else if (mode == 3) {
            masked_idx >>= row_bits + col_bits;
        }

        static_cast<int>(masked_idx);

        if(mode == 3 && row_bits + col_bits + depth_bits > 64){
            
            masked_idx |= block << (-1 * (row_bits + col_bits - 64));
        }

        return masked_idx;
    }

    //Returns the index in the vector that the block is in -1 if it doesn't exist in the vector
    int find_block(int target_block)
    {
        for(int i = 0; i < blco_tensor.size(); i++){
            if(blco_tensor[i].block == target_block) return i;
        }
        return -1;
    }

    //Returns the blco tensor
    const std::vector<BLCO_BLOCK_CPU<T>>& get_blco() const
    {
        return blco_tensor;
    }

    //Returns the ModeMasks
    std::vector<uint64_t> get_blco_masks() const
    {
        return {m1_blco_mask, m2_blco_mask, m3_blco_mask};
    }

    //Returns the bit offsets
    std::vector<int> get_bit_offsets() const
    {
        return {row_bits, col_bits, depth_bits};
    }

    void copy_vector_to_fmat(T* v1, int mode) const
    {
        if(mode == 1)
        {
            for(int i = 0; i < this->rows * this->rank; i++){
                this->mode_1_fmat[i/this->rank][i % this->rank] = v1[i];
            }
        }
        else if(mode == 2)
        {
            for(int i = 0; i < this->cols * this->rank; i++){
                this->mode_2_fmat[i/this->rank][i % this->rank] = v1[i];
            }
        }
        else{
            for(int i = 0; i < this->depth * this->rank; i++){
                this->mode_3_fmat[i/this->rank][i % this->rank] = v1[i];
            }
        }
    }

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
            if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) indexes.push_back(idx);
        }

        if(indexes.size() > max_indexes) return indexes.size();
        return max_indexes;
    }

    int determine_indexes_per_wavefront_128_bit(int mode) const {
        int max_indexes = 0;
        int block_num = -1;
        for (const auto &block : blco_tensor) {
            std::vector<int> indexes;
            block_num++;
            for (size_t i = 0; i < block.indexes.size(); i++) {
                if (i != 0 && i % 64 == 0) {
                    // Update max
                    if (indexes.size() > max_indexes)
                        max_indexes = indexes.size();
                    indexes.clear();
                }

                int idx = get_mode_idx_blco_128_bit(block.indexes[i],block_num,mode);
                if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) {
                    indexes.push_back(idx);
                }
            }

            // Final check for the last partial wavefront in this block
            if (indexes.size() > max_indexes)
                max_indexes = indexes.size();
        }

        return max_indexes;
    }


    //Used to debug 
    void debug_linear_indices_64_bit()
    {
        BLCO_BLOCK_CPU<T> b1;
        int block;
        for (int i = 0; i < blco_tensor.size(); i++) {
            b1 = blco_tensor[i];
            block = b1.block;
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

    //Used to debug 
    void debug_linear_indices_128_bit()
    {
        BLCO_BLOCK_CPU<T> b1;
        int block;
        for (int i = 0; i < blco_tensor.size(); i++) {
            b1 = blco_tensor[i];
            block = b1.block;
            for(int j = 0; j < b1.indexes.size(); j++){
                std::cout << "Block: " << b1.block
                    << ", Index: " << b1.indexes[j]
                    << ", i=" << get_mode_idx_blco_128_bit(b1.indexes[j], block, 1)
                    << ", j=" << get_mode_idx_blco_128_bit(b1.indexes[j], block, 2)
                    << ", k=" << get_mode_idx_blco_128_bit(b1.indexes[j], block, 3)
                    << ", val=" << b1.values[j] << "\n";
            }
        }
    }
};

#endif
