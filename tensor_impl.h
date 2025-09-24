#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <sstream>
#include <initializer_list>
#include <random>
#include <type_traits>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "utils.h"


template<typename T, typename S>
class Tensor_3D{
protected:
    int rows; //Number of rows
    int cols; //Number of columns
    int depth; //Depth of tensor
    __uint128_t total_entries; //Number of total entries in tensor
    uint64_t nnz_entries; //Total number of non zero entries
    int rank; //Rank of Tensor
    T** mode_1_fmat; //Factor matrix of mode 1
    T** mode_2_fmat; //Factor matrix of mode 2
    T** mode_3_fmat; //Factor matrix of mode 3

    //Calculate elements per fiber for paralell alto algorithm
    int elements_per_fiber(int mode)
    {
        if(mode == 1) return nnz_entries / cols * depth;
        else if(mode == 2) return nnz_entries / rows * depth;
        else if(mode == 3) return nnz_entries / rows * cols;
    }

    //Determine typical rank of the tensor based on it's dimensions
    int estimate_typical_rank() 
    {
        // Total number of elements
        long double total_elements = static_cast<long double>(rows) *
                                     static_cast<long double>(cols) *
                                     static_cast<long double>(depth);
    
        // Heuristic 1: Logarithmic mean of tensor size (used for dense tensors)
        int log_rank = static_cast<int>(std::log10(total_elements));
    
        // Heuristic 2: Sparsity-aware estimate (if nnz is provided)
        int sparse_rank = nnz_entries > 0 ? std::min(static_cast<int>(nnz_entries / 1000000), 500) : log_rank;
    
        // Return a rank that's a balance of both
        return std::clamp((log_rank + sparse_rank) / 2, 10, 500);
    }

    //Generates random matricies for factor matricies
    void init_mode_matrix(T**& matrix, int rows, int cols) 
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type.");

        // Allocate row pointers
        matrix = new T*[rows];

        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            matrix[i] = new T[cols];
        }

        // Parallel initialization with thread-local RNGs
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() ^ omp_get_thread_num());  // unique seed per thread

            if constexpr (std::is_integral<T>::value) {
                std::uniform_int_distribution<T> dist(0, 3);
                #pragma omp for
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        matrix[i][j] = dist(gen);
            } else {
                std::uniform_real_distribution<T> dist(0.0, 1.0);
                #pragma omp for
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        matrix[i][j] = dist(gen);
            }
        }
    }

    //Function to delete matrixes
    void delete_matrix(T** matrix, int rows, int cols)
    {
        if (!matrix) return;

        for (int i = 0; i < rows; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }

    //Initialize the factor matricies
    void init_factor_matricies()
    {
        init_mode_matrix(mode_1_fmat, rows, rank);
        init_mode_matrix(mode_2_fmat, cols, rank);
        init_mode_matrix(mode_3_fmat, depth, rank);
    }

public:
    //Initializer function with pointer array
    Tensor_3D(T*** array, int r, int c, int d)
    {
        rows = r; cols = c; depth = d; 
        total_entries = r * c * d;
        rank = estimate_typical_rank();
        init_factor_matricies();
    }

    //Initializer function with list of nonzero entries
    Tensor_3D(const std::vector<NNZ_Entry<T>>& entry_vec, int r, int c, int d)
    {
        rows = r; cols = c; depth = d; 
        total_entries = r * c * d;
        nnz_entries = entry_vec.size();
        rank = estimate_typical_rank();
        init_factor_matricies(); 
    }

    //Returns the mode masks as a vector
    std::vector<T**> get_fmats() const 
    {
        return {mode_1_fmat, mode_2_fmat, mode_3_fmat};
    }

    //Returns rows,cols,depth in a vector
    std::vector<int> get_dims() const
    {
        return {rows,cols,depth}; 
    }

    int get_rank() const
    {
        return rank;
    }

    int get_nnz() const
    {
        return nnz_entries;
    }

    //Destructor
    ~Tensor_3D() 
    {
        delete_matrix(mode_1_fmat,rows,rank);
        mode_1_fmat = nullptr;
        delete_matrix(mode_2_fmat,cols,rank);
        mode_2_fmat = nullptr;
        delete_matrix(mode_3_fmat,depth,rank);
        mode_3_fmat = nullptr;
        return;
    }
};



#endif