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
#include "../utility/utils.h"


//======================================================================
// Template class for representing and factorizing a 3D sparse tensor
//======================================================================
//
// T = numeric type (e.g., float, double, int)
// S = index type (e.g., int, size_t)
//
// The tensor is factorized into 3 factor matrices, one for each mode:
//   - mode_1_fmat: rows × rank
//   - mode_2_fmat: cols × rank
//   - mode_3_fmat: depth × rank
//
// The implementation supports initialization from raw arrays or sparse
// NNZ (nonzero) entry lists, and automatically estimates a suitable rank.
//======================================================================
template<typename T, typename S>
class Tensor_3D {
protected:
    int rows;                // Number of rows in tensor
    int cols;                // Number of columns in tensor
    int depth;               // Depth (3rd dimension)
    __uint128_t total_entries; // Total possible entries (rows * cols * depth)
    uint64_t nnz_entries;    // Number of nonzero entries
    int rank;                // Factorization rank (estimated automatically)

    // Factor matrices (dense, small rank approximations)
    T** mode_1_fmat; // Rows × Rank
    T** mode_2_fmat; // Cols × Rank
    T** mode_3_fmat; // Depth × Rank


    //------------------------------------------------------------------
    // Utility: Estimate number of elements per fiber (line of tensor)
    // For use in parallel ALTO algorithms
    //------------------------------------------------------------------
    int elements_per_fiber(int mode)
    {
        if (mode == 1) return nnz_entries / cols * depth;
        else if (mode == 2) return nnz_entries / rows * depth;
        else if (mode == 3) return nnz_entries / (rows * cols);
        return 0;
    }

    //------------------------------------------------------------------
    // Estimate a "typical rank" for the tensor
    // Combines a logarithmic heuristic and a sparsity-aware heuristic
    //------------------------------------------------------------------
    int estimate_typical_rank() 
    {
        // Compute total number of elements in dense tensor
        long double total_elements = static_cast<long double>(rows) *
                                     static_cast<long double>(cols) *
                                     static_cast<long double>(depth);
    
        // Heuristic 1: log10-based estimate for dense tensors
        int log_rank = static_cast<int>(std::log10(total_elements));
    
        // Heuristic 2: limit rank based on number of nonzeros
        int sparse_rank = nnz_entries > 0 ? 
            std::min(static_cast<int>(nnz_entries / 1000000), 500) : log_rank;
    
        // Combine heuristics, clamp to [10, 500]
        return std::clamp((log_rank + sparse_rank) / 2, 10, 500);
    }

    //------------------------------------------------------------------
    // Initialize one factor matrix with random values
    // Allocates a rows×cols 2D array and fills with random ints or floats
    //------------------------------------------------------------------
    void init_mode_matrix(T**& matrix, int rows, int cols) 
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type.");

        // Allocate pointer array for rows
        matrix = new T*[rows];

        // Allocate each row in parallel
        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            matrix[i] = new T[cols];
        }

        // Parallel initialization with thread-local random engines
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() ^ omp_get_thread_num()); // unique seed per thread

            if constexpr (std::is_integral<T>::value) {
                // If T is integer → uniform [0,3]
                std::uniform_int_distribution<T> dist(0, 3);
                #pragma omp for
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        matrix[i][j] = dist(gen);
            } else {
                // If T is floating point → uniform [0,1]
                std::uniform_real_distribution<T> dist(0.0, 1.0);
                #pragma omp for
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        matrix[i][j] = dist(gen);
            }
        }
    }

    //------------------------------------------------------------------
    // Delete a dynamically allocated factor matrix
    //------------------------------------------------------------------
    void delete_matrix(T** matrix, int rows, int /*cols*/)
    {
        if (!matrix) return;

        for (int i = 0; i < rows; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }

    //------------------------------------------------------------------
    // Initialize all 3 factor matrices
    //------------------------------------------------------------------
    void init_factor_matricies()
    {
        init_mode_matrix(mode_1_fmat, rows, rank);
        init_mode_matrix(mode_2_fmat, cols, rank);
        init_mode_matrix(mode_3_fmat, depth, rank);
    }

public:
    //------------------------------------------------------------------
    // Constructor: from a dense 3D pointer array
    // (Assumes array is already allocated externally)
    //------------------------------------------------------------------
    Tensor_3D(T*** array, int r, int c, int d)
    {
        rows = r; cols = c; depth = d; 
        total_entries = static_cast<__uint128_t>(r) * c * d;
        nnz_entries = 0; // not explicitly set
        rank = estimate_typical_rank();
        init_factor_matricies();
    }

    //------------------------------------------------------------------
    // Constructor: from a list of sparse nonzero entries
    //------------------------------------------------------------------
    Tensor_3D(const std::vector<NNZ_Entry<T>>& entry_vec, int r, int c, int d)
    {
        rows = r; cols = c; depth = d; 
        total_entries = static_cast<__uint128_t>(r) * c * d;
        nnz_entries = entry_vec.size();
        rank = estimate_typical_rank();
        init_factor_matricies(); 
    }

    //------------------------------------------------------------------
    // Get factor matrices (mode-1, mode-2, mode-3)
    //------------------------------------------------------------------
    std::vector<T**> get_fmats() const 
    {
        return {mode_1_fmat, mode_2_fmat, mode_3_fmat};
    }

    //------------------------------------------------------------------
    // Get tensor dimensions [rows, cols, depth]
    //------------------------------------------------------------------
    std::vector<int> get_dims() const
    {
        return {rows, cols, depth}; 
    }

    //------------------------------------------------------------------
    // Get factorization rank
    //------------------------------------------------------------------
    int get_rank() const
    {
        return rank;
    }

    //------------------------------------------------------------------
    // Get number of nonzero entries
    //------------------------------------------------------------------
    int get_nnz() const
    {
        return nnz_entries;
    }

    //------------------------------------------------------------------
    // Destructor: free factor matrices
    //------------------------------------------------------------------
    ~Tensor_3D() 
    {
        delete_matrix(mode_1_fmat, rows, rank);
        mode_1_fmat = nullptr;

        delete_matrix(mode_2_fmat, cols, rank);
        mode_2_fmat = nullptr;

        delete_matrix(mode_3_fmat, depth, rank);
        mode_3_fmat = nullptr;
    }
};

#endif // TENSOR_IMPL_H
