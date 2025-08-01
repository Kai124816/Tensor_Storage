#ifndef BLCO_H
#define BLCO_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <initializer_list>
#include <random>
#include <type_traits>
#include "alto_impl.h"


template<typename T, typename S>
class BLCO_Tensor_3D : public Alto_Tensor_3D<T, S>{
protected:
    std::pair<std::vector<uint64_t>, std::vector<T>> blco_tensor;
    std::pair<std::vector<int>, std::vector<int>> block_pointer; 
    S m1_blco_mask = 0; //m1 means rows
    S m2_blco_mask = 0; //m2 means cols
    S m3_blco_mask = 0; //m3 means depth

    //Function used to sort two vectors
    void sort_pair_by_first(std::pair<std::vector<S>, std::vector<T>>& p) {
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

        int m1 = this->rows - 1, m2 = this->cols - 1, m3 = this->depth - 1;

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
        int num_bits = ceiling_log2(this->rows) + ceiling_log2(this->cols) + ceiling_log2(this->depth);

        for (int i = 0; i < num_bits; ++i) {
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

        return val;
    }

    //Converts the alto tensor into an intermediate tensor representation
    std::pair<std::vector<S>, std::vector<T>> create_intermediate_tensor()
    {
        std::pair<std::vector<S>, std::vector<T>> p1;

        for(int i = 0; i < this->alto_tensor.size(); i++){
            p1.first.push_back(index_conversion(this->alto_tensor[i].linear_index));
            p1.second.push_back(this->alto_tensor[i].value);
        }

        sort_pair_by_first(p1);
        
        return p1;
    }

    //Creates the BLCO tensor and fills the block pointer array
    void create_blco_tensor()
    {
        std::pair<std::vector<S>, std::vector<T>> p1 = create_intermediate_tensor();

        int largest_block;
        if(p1.first[p1.first.size() - 1] <= limit) largest_block = 0;
        else largest_block = static_cast<int>((p1.first[p1.first.size() - 1] >> 64) & limit);

        int limit_bits = 0;
        if constexpr (std::is_same_v<S, __uint128_t>) limit_bits = 1;;

        int block_num;

        if(limit_bits){
            __uint128_t blco_index;
            for(int i = 0; i < p1.first.size(); i++){
                __uint128_t shifted = p1.first[i] >> 64;
                block_num = static_cast<int>((p1.first[i] >> 64) & limit);
                blco_index = p1.first[i] & limit;
                static_cast<uint64_t>(blco_index);
                T val = p1.second[i];
                blco_tensor.first.push_back(blco_index);
                blco_tensor.second.push_back(val);

                auto it = std::find(block_pointer.first.begin(), block_pointer.first.end(), block_num);
                if (it != block_pointer.first.end()) {
                    auto index = std::distance(block_pointer.first.begin(), it);
                    block_pointer.second[index]++;
                } else {
                    block_pointer.first.push_back(block_num);
                    block_pointer.second.push_back(0);  // <-- Correct way to add new entry
                }
            }
        }
        else{
            uint64_t blco_index;
            uint64_t mask = 0xFFFFFFFF;
            block_pointer.first.push_back(0);
            for(int i = 0; i < p1.first.size(); i++){
                blco_index = p1.first[i] & mask;
                T val = p1.second[i];
                blco_tensor.first.push_back(blco_index);
                blco_tensor.second.push_back(val);

                block_pointer.second[0]++;
            }
        }
    }

public:
    BLCO_Tensor_3D(T*** array, int r, int c, int d) : Alto_Tensor_3D<T,S>(array, r, c, d)
    {
        create_blco_masks();
        create_blco_tensor();
    }

    BLCO_Tensor_3D(const std::vector<NNZ_Entry<T>>& entry_vec, int r, int c, int d) : Alto_Tensor_3D<T,S>(entry_vec, r, c, d)
    {
        create_blco_masks();
        create_blco_tensor();
    }

    //Get that idx for any given mode based on the BLCO index
    int get_mode_idx_blco(uint64_t blco_index, int vector_index, int mode)
    {
        S mask = (mode == 1) ? m1_blco_mask :
                        (mode == 2) ? m2_blco_mask : m3_blco_mask;
        int num_bits = ceiling_log2(this->rows) + ceiling_log2(this->cols) + ceiling_log2(this->depth); 

        S block = static_cast<S>(find_block(vector_index));
        S index = static_cast<S>(blco_index);
        S full_index = index | (block << 64);
        S masked = full_index & mask;


        for (int i = 0; i < num_bits; i++) {
            if(mask & 1ULL) break;
            masked >>= 1;
            mask >>=1;
        }
        
        return static_cast<int>(masked);
    } 

    //Finds block based on index in blco tensor
    int find_block(int index)
    {
        int i = 0;
        for (; i < block_pointer.first.size(); i++) {
            if (block_pointer.first[i] > index) break;
        }
        return block_pointer.first[i];
    }

    //Returns the blco tensor
    const std::pair<std::vector<uint64_t>, std::vector<T>>& get_blco() const
    {
        return blco_tensor;
    }

    //Returns the ModeMasks
    std::vector<S> get_modemasks() const override
    {
        return {m1_blco_mask, m2_blco_mask, m3_blco_mask};
    }

    std::pair<std::vector<int>, std::vector<int>> get_block_pointer()
    {
        return block_pointer;
    }

    //Used to debug 
    void debug_linear_indices() override{
        for (int i = 0; i < blco_tensor.first.size(); i++) {
            std::cout << "Index: " << blco_tensor.first[i]
                      << ", i=" << get_mode_idx_blco(blco_tensor.first[i], i, 1)
                      << ", j=" << get_mode_idx_blco(blco_tensor.first[i], i, 2)
                      << ", k=" << get_mode_idx_blco(blco_tensor.first[i], i, 3)
                      << ", val=" << blco_tensor.second[i] << "\n";
        }
    }
};


#endif
