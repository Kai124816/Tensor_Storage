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

//======================================================================
// Template class for representing and factorizing a sparse tensor
//======================================================================
//
// T = numeric type (e.g., float, double, int)
// S = index type (e.g., int, size_t)
//
// The tensor is factorized into n factor matrices, one for each mode:
//
// The implementation supports initialization from raw arrays or sparse
// NNZ (nonzero) entry lists, and automatically estimates a suitable rank.
//======================================================================
template<typename T, typename S>
class Tensor {
protected:
    int rank; //Rank of tensor (number of dimensions)
    int num_bits; //Total bits needed to encode tensor
    int factor_rank;  // Factorization rank (estimated automatically) or passed in
    __uint128_t total_entries; // Total possible entries (rows * cols * depth)
    uint64_t nnz_entries;    // Number of nonzero entries
    std::vector<T*> fmats; //vector of matricized factor matricies
    std::vector<int> dims; //Vector which stores mode sizes


    //------------------------------------------------------------------
    // Utility: Estimate number of elements per fiber (line of tensor)
    // For use in parallel ALTO algorithms
    //------------------------------------------------------------------
    int elements_per_fiber(int mode)
    {
        int div = 1;
        for(int i = 0; i < rank; i++){
            if(i + 1 != mode) div *= dims[i];
        }
        return div/nnz_entries;
    }

    //------------------------------------------------------------------
    // Utility: Determine the number of bits you need to represent the 
    // tensor dimensions (Useful for ALTO and BLCO representations)
    //------------------------------------------------------------------
    void determine_num_bits()
    {
        int bit_count = 0;
        for(int i = 0; i < rank; i++){
            bit_count += ceiling_log2(dims[i]);
        }
        num_bits = bit_count;
    }

    //------------------------------------------------------------------
    // Estimate a "typical rank" for the tensor
    // Combines a logarithmic heuristic and a sparsity-aware heuristic
    //------------------------------------------------------------------
    int estimate_typical_rank() 
    {
        // Compute total number of elements in dense tensor
        long double total_elements = 1;
        for(int i = 0; i < rank; i++){
            total_elements *= dims[i];
        }
    
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
    // Allocates a colsxrows vectorized and transposed matrix and fills with random ints or floats
    // Matrix is transposed for tucker decomposition
    //------------------------------------------------------------------
    void init_mode_matrix(T*& matrix, int rows, int cols) 
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type.");

        // Allocate pointer array for rows
        matrix = new T[rows * cols];

        // Parallel initialization with thread-local random engines
        #pragma omp parallel
        {
            if constexpr (std::is_integral<T>::value) {
                std::uniform_int_distribution<T> dist(0, 3);
            
                // Parallelize outer loop (rows), each thread works on independent rows
                #pragma omp parallel
                {
                    // Each thread needs its own RNG to avoid contention
                    std::mt19937 gen_local(std::random_device{}());
                    #pragma omp for schedule(static)
                    for (int i = 0; i < rows; ++i) {
                        T* row_ptr = &matrix[i * cols];  // pointer to start of row
                        for (int j = 0; j < cols; ++j) {
                            row_ptr[j] = dist(gen_local);
                        }
                    }
                }
            } 
            else {
                // Generate integers from 0 to 1024
                std::uniform_int_distribution<int> dist(0, 3);
            
                #pragma omp parallel
                {
                    std::mt19937 gen_local(std::random_device{}());
                    #pragma omp for schedule(static)
                    for (int i = 0; i < rows; ++i) {
                        T* row_ptr = &matrix[i * cols];
                        for (int j = 0; j < cols; ++j) {
                            // Dividing by 1024.0 is perfectly precise in binary
                            row_ptr[j] = static_cast<T>(dist(gen_local));
                        }
                    }
                }
            }
        }  
    }

    //------------------------------------------------------------------
    // Initialize all 3 factor matrices
    //------------------------------------------------------------------
    void init_factor_matricies()
    {
        for(int i = 0; i < rank; i++){
            T* mat;
            init_mode_matrix(mat, dims[i], factor_rank);
            fmats.push_back(mat);
        }
    }

    void copy_array(T*& dest, T* const& src, int r) {
        if (!src) {
            dest = nullptr;
            return;
        }
        dest = new T[r];
        #pragma omp parallel for
        for (int i = 0; i < r; ++i) {
            dest[i] = src[i];
        }
    }

    //------------------------------------------------------------------
    // Helper: Free all managed matrices (used by destructor and assignment)
    //------------------------------------------------------------------
    void cleanup() {
        for(int i = 0; i < rank; i++){
            delete[] fmats[i];
            fmats[i] = nullptr;
        }
    }

public:
    //------------------------------------------------------------------
    // Constructor: from a list of sparse nonzero entries
    //------------------------------------------------------------------
    Tensor(const std::vector<NNZ_Entry<T>>& entry_vec, std::vector<int> dim_list, int decomp_rank = 10)
    {
        rank = dim_list.size();
        dims = dim_list;
        nnz_entries = entry_vec.size();
        determine_num_bits();
        factor_rank = decomp_rank;
        total_entries = static_cast<__uint128_t>(dims[0]);
        for(int i = 1; i < rank; i++){
            total_entries *= dims[i];
        }
        init_factor_matricies(); 
    }

    //------------------------------------------------------------------
    // 2. COPY CONSTRUCTOR
    // Performs a deep copy so that the new object has its own memory
    //------------------------------------------------------------------
    Tensor(const Tensor& other) {
        rank = other.rank;
        num_bits = other.num_bits;
        factor_rank = other.factor_rank;
        total_entries = other.total_entries;
        nnz_entries = other.nnz_entries;
        total_entries = other.total_entries;
        nnz_entries = other.nnz_entries;
        dims = other.dims;

        for(int i = 0; i < rank; ++i){
            T* mode_fmat;
            copy_array(other.fmats[i], mode_fmat, dims[i] * factor_rank);
            fmats.push_back(mode_fmat);
        }
    }

    //------------------------------------------------------------------
    // 3. COPY ASSIGNMENT OPERATOR
    // Handles self-assignment and frees existing memory before copying
    //------------------------------------------------------------------
    Tensor& operator=(const Tensor& other) {
        if (this != &other) { // Protect against self-assignment
            // Free current resources
            cleanup();

            rank = other.rank;
            num_bits = other.num_bits;
            factor_rank = other.factor_rank;
            total_entries = other.total_entries;
            nnz_entries = other.nnz_entries;
            total_entries = other.total_entries;
            nnz_entries = other.nnz_entries;
            dims = other.dims;

            for(int i = 0; i < rank; ++i){
                T* mode_fmat;
                copy_array(other.fmats[i], mode_fmat, dims[i] * factor_rank);
                fmats.push_back(mode_fmat);
            }
        }
        return *this;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    std::vector<T*> get_fmats() const { return fmats;}
    std::vector<int> get_dims() const{ return dims;}
    int get_factor_rank() const { return factor_rank;}
    int get_nnz() const { return nnz_entries;}
    int get_total_bits_needed() const { return num_bits;}

    //------------------------------------------------------------------
    // Destructor: free factor matrices
    //------------------------------------------------------------------
    ~Tensor() 
    {
       cleanup();
    }
};

#endif // TENSOR_IMPL_H
