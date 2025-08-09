#ifndef BLCO_H
#define BLCO_H

// #include <hip/hip_runtime.h>
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
class BLCO_Tensor_3D : public Alto_Tensor_3D<T, S>{
protected:
    int row_bits;
    int col_bits;
    int depth_bits;
    std::vector<BLCO_BLOCK_CPU<T>> blco_tensor;
    uint64_t m1_blco_mask = 0; //m1 means rows
    uint64_t m2_blco_mask = 0; //m2 means cols
    uint64_t m3_blco_mask = 0; //m3 means depth

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

    //Creates an array of BLCO blocks that can be used on the GPU
    BLCO_BLOCK_GPU<T>* create_blco_array()
    {
        BLCO_BLOCK_GPU<T>* arr = new BLCO_BLOCK_GPU[blco_tensor.size()];

        for(int i = 0; i < blco_tensor.size(); i++){
            arr[i].block = blco_tensor[i].block;
            arr[i].size = blco_tensor[i].size;
            arr[i].indexes =  new uint64_t[blco_tensor[i].size];
            arr[i].values = new T[blco_tensor[i].size];
            vector_to_array(arr[i].indexes, blco_tensor[i].indexes);
        }
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

        int limit_bits = 0;
        if constexpr (std::is_same_v<S, __uint128_t>) limit_bits = 1;;

        int block_num;
        int result;

        if(limit_bits){
            __uint128_t blco_index;
            for(int i = 0; i < p1.first.size(); i++){
                block_num = static_cast<int>(p1.first[i] >> 64);
                result = find_block(block_num);
                blco_index = p1.first[i] & limit;
                static_cast<uint64_t>(blco_index);
                T val = p1.second[i];

                if(result == -1){
                    BLCO_BLOCK_CPU<T> new_block;
                    new_block.block = block_num;
                    new_block.indexes.push_back(blco_index);
                    new_block.values.push_back(val);
                    new_block.size = 1;
                    blco_tensor.push_back(new_block);
                }
                else{
                    blco_tensor[result].indexes.push_back(blco_index);
                    blco_tensor[result].values.push_back(val);
                    blco_tensor[result].size++;
                }
            }
        }
        else{
            uint64_t blco_index;

            BLCO_BLOCK_CPU<T> b1;
            b1.block = 0;
            b1.size = 0;
            blco_tensor.push_back(b1);

            for(int i = 0; i < p1.first.size(); i++){
                blco_index = static_cast<uint64_t>(p1.first[i]);
                T val = p1.second[i];
                
                blco_tensor[0].indexes.push_back(blco_index);
                blco_tensor[0].values.push_back(val);
                blco_tensor[0].size++;
            }
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

    //Get that idx for any given mode based on the BLCO index
    int get_mode_idx_blco(uint64_t blco_index, int block, int mode)
    {
        S mask = (mode == 1) ? static_cast<S>(m1_blco_mask) :
                        (mode == 2) ? static_cast<S>(m2_blco_mask) : static_cast<S>(m3_blco_mask);

        if(mode == 3) mask |= static_cast<S>(0xFFFFFFFF) << 64;

        S casted_block = static_cast<S>(block);
        S index = static_cast<S>(blco_index);
        S full_index = index | (casted_block << 64);
        S masked = full_index & mask;

        if(mode == 2) masked >>= row_bits;
        else if(mode == 3) masked >>= row_bits + col_bits;
        
        return static_cast<int>(masked);
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
    std::vector<S> get_modemasks() const override
    {
        return {m1_blco_mask, m2_blco_mask, m3_blco_mask};
    }

    //Paralell MTTKRP on GPU
    // void MTTKRP_BLCO(int mode)
    // {
    //     int target_mode_size = (mode == 1) ? this->rows:
    //                     (mode == 2) ? this->cols : this->depth;

    //     T** target_fmat = (mode == 1) ? this->mode_1_fmat:
    //     (mode == 2) ? this->mode_2_fmat : this->mode_3_fmat;

    //     //Initialize device memory
    //     uint64_t d_nnz;
    //     uint64_t d_m1_mask, d_m2_mask, d_m3_mask;
    //     uint64_t* index_array;
    //     T* val_array;
    //     T* d_fmat_vector;
    //     int d_histogram;
    //     int d_m1_bits, d_m2_bits, d_m3_bits;
    //     hipMalloc(&d_nnz, sizeof(uint64_t));
    //     hipMalloc(&d_m1_mask, sizeof(uint64_t));
    //     hipMalloc(&d_m2_mask, sizeof(uint64_t));
    //     hipMalloc(&d_m3_mask, sizeof(uint64_t));
    //     hipMalloc(&index_array, sizeof(uint64_t) * this->nnz_entries);
    //     hipMalloc(&val_array,sizeof(T) * this->nnz_entries);
    //     hipMalloc(&d_fmat_vector,sizeof(T) * target_mode_size * this->rank);
    //     hipMalloc(&d_histogram, sizeof(int) * block_histogram.size());
    //     hipMalloc(&d_m1_bits, sizeof(int));
    //     hipMalloc(&d_m2_bits, sizeof(int));
    //     hipMalloc(&d_m3_bits, sizeof(int));

    //     T* h_fmat_vector = vectorize_matrix(target_fmat,target_mode_size,this->rank);
        
    //     //Copy host data to GPU
    //     hipMemcpy(d_nnz, this->nnz_entries, sizeof(uint64_t), hipMemcpyHostToDevice);
    //     hipMemcpy(d_m1_mask, m1_blco_mask, sizeof(uint64_t), hipMemcpyHostToDevice);
    //     hipMemcpy(d_m2_mask, m2_blco_mask, sizeof(uint64_t), hipMemcpyHostToDevice);
    //     hipMemcpy(d_m3_mask, m3_blco_mask, sizeof(uint64_t), hipMemcpyHostToDevice);
    //     hipMemcpy(index_array, blco_tensor.first().data(), sizeof(uint64_t) * this->nnz_entries, hipMemcpyHostToDevice);
    //     hipMemcpy(val_array, blco_tensor.second().data(), sizeof(T) * this->nnz_entries, hipMemcpyHostToDevice);
    //     hipMemcpy(d_fmat_vector, h_fmat_vector, sizeof(T) * target_mode_size * this->rank, hipMemcpyHostToDevice);
    //     hipMemcpy(d_histogram, block_histogram.data(), sizeof(int) * block_histogram.size(), hipMemcpyHostToDevice);
    //     hipMemcpy(d_m1_bits, row_bits, sizeof(int), hipMemcpyHostToDevice);
    //     hipMemcpy(d_m2_bits, col_bits, sizeof(int), hipMemcpyHostToDevice);
    //     hipMemcpy(d_m3_bits, depth_bits, sizeof(int), hipMemcpyHostToDevice);
    // }


    //Used to debug 
    void debug_linear_indices() override
    {
        BLCO_BLOCK_CPU<T> b1;
        int block;
        for (int i = 0; i < blco_tensor.size(); i++) {
            b1 = blco_tensor[i];
            block = b1.block;
            for(int j = 0; j < b1.indexes.size(); j++){
                std::cout << "Block: " << b1.block
                    << "Index: " << b1.indexes[j]
                    << ", i=" << get_mode_idx_blco(b1.indexes[j], block, 1)
                    << ", j=" << get_mode_idx_blco(b1.indexes[j], block, 2)
                    << ", k=" << get_mode_idx_blco(b1.indexes[j], block, 3)
                    << ", val=" << b1.values[j] << "\n";
            }
        }
    }
};


#endif
