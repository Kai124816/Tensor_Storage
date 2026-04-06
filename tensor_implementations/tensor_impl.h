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
#include <fstream>

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
    std::vector<std::vector<T>> fmats; //vector of matricized factor matrices
    std::vector<int> dims; //Vector which stores mode sizes


    //------------------------------------------------------------------
    // Utility: Estimate number of elements per fiber (line of tensor)
    // For use in parallel ALTO algorithms
    //------------------------------------------------------------------
    int elements_per_fiber(int mode)
    {
        if (mode < 1 || mode > rank)
            throw std::out_of_range("elements_per_fiber: mode " + std::to_string(mode) +
                                   " is out of range [1, " + std::to_string(rank) + "]");
        if (nnz_entries == 0)
            throw std::domain_error("elements_per_fiber: tensor has no nonzero entries");
        int div = 1;
        for(int i = 0; i < rank; i++){
            if(i + 1 != mode) div *= dims[i];
        }
        return div / nnz_entries;
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
    void init_mode_matrix(std::vector<T>& matrix, int rows, int cols) 
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type.");
        if (rows <= 0)
            throw std::invalid_argument("init_mode_matrix: rows must be positive, got " + std::to_string(rows));
        if (cols <= 0)
            throw std::invalid_argument("init_mode_matrix: cols must be positive, got " + std::to_string(cols));

        matrix.resize(rows * cols);

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
                std::uniform_int_distribution<int> dist(0, 3);
            
                #pragma omp parallel
                {
                    std::mt19937 gen_local(std::random_device{}());
                    #pragma omp for schedule(static)
                    for (int i = 0; i < rows; ++i) {
                        T* row_ptr = &matrix[i * cols];
                        for (int j = 0; j < cols; ++j) {
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
        fmats.resize(rank);
        for(int i = 0; i < rank; i++){
            init_mode_matrix(fmats[i], dims[i], factor_rank);
        }
    }

public:
    //------------------------------------------------------------------
    // Default Constructor
    //------------------------------------------------------------------
    Tensor()
    {
        rank = 0; 
        num_bits = 0; 
        factor_rank = 0;  
        total_entries = 0; 
        nnz_entries = 0;   
        fmats = {}; 
        dims = {}; 
    }
    //------------------------------------------------------------------
    // Constructor: from a list of sparse nonzero entries
    //------------------------------------------------------------------
    Tensor(const std::vector<NNZ_Entry<T>>& entry_vec, std::vector<int> dim_list, int decomp_rank = 10)
    {
        if (dim_list.empty())
            throw std::invalid_argument("Tensor: dim_list must not be empty");
        for (int i = 0; i < static_cast<int>(dim_list.size()); ++i)
            if (dim_list[i] <= 0)
                throw std::invalid_argument("Tensor: dim_list[" + std::to_string(i) +
                                          "] must be positive, got " + std::to_string(dim_list[i]));
        if (decomp_rank <= 0)
            throw std::invalid_argument("Tensor: decomp_rank must be positive, got " + std::to_string(decomp_rank));
        for (int i = 0; i < static_cast<int>(entry_vec.size()); ++i)
            if (static_cast<int>(entry_vec[i].coords.size()) != static_cast<int>(dim_list.size()))
                throw std::invalid_argument("Tensor: entry_vec[" + std::to_string(i) +
                                          "] has " + std::to_string(entry_vec[i].coords.size()) +
                                          " coordinates but tensor has " + std::to_string(dim_list.size()) + " modes");

        rank = dim_list.size();
        dims = dim_list;
        nnz_entries = static_cast<uint64_t>(entry_vec.size());
        determine_num_bits();
        factor_rank = decomp_rank;
        total_entries = static_cast<__uint128_t>(dims[0]);
        for(int i = 1; i < rank; i++){
            total_entries *= dims[i];
        }
        init_factor_matricies(); 
    }

    //------------------------------------------------------------------
    // Constructor: from NNZ entries + pre-loaded factor matrices from files
    // fmat_files[i] must be a path to a .txt file (one value per line)
    // containing at least dims[i] * decomp_rank values.
    // Values are read in row-major order; wrap-around is applied if the
    // file is shorter than the required size (via file_to_array).
    //------------------------------------------------------------------
    Tensor(const std::vector<NNZ_Entry<T>>& entry_vec,
           std::vector<int> dim_list,
           std::vector<std::vector<T>>& default_fmats,
           int decomp_rank = 10)
    {
        if (dim_list.empty())
            throw std::invalid_argument("Tensor: dim_list must not be empty");
        for (int i = 0; i < static_cast<int>(dim_list.size()); ++i)
            if (dim_list[i] <= 0)
                throw std::invalid_argument("Tensor: dim_list[" + std::to_string(i) +
                                          "] must be positive, got " + std::to_string(dim_list[i]));
        if (decomp_rank <= 0)
            throw std::invalid_argument("Tensor: decomp_rank must be positive, got " +
                                       std::to_string(decomp_rank));

        rank = static_cast<int>(dim_list.size());
        dims = dim_list;
        nnz_entries = static_cast<uint64_t>(entry_vec.size());
        determine_num_bits();
        factor_rank = decomp_rank;
        total_entries = static_cast<__uint128_t>(dims[0]);
        for (int i = 1; i < rank; i++)
            total_entries *= dims[i];

        // Load each factor matrix from its file instead of random init
        fmats.resize(rank);
        for (int i = 0; i < rank; i++) {
            fmats[i].resize(factor_rank * dims[i]);
            fmats[i] = default_fmats[i];
        }
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
        dims = other.dims;
        fmats = other.fmats; // std::vector deep-copies automatically
    }

    //------------------------------------------------------------------
    // 3. COPY ASSIGNMENT OPERATOR
    // Handles self-assignment and frees existing memory before copying
    //------------------------------------------------------------------
    Tensor& operator=(const Tensor& other) {
        if (this != &other) { // Protect against self-assignment
            rank = other.rank;
            num_bits = other.num_bits;
            factor_rank = other.factor_rank;
            total_entries = other.total_entries;
            nnz_entries = other.nnz_entries;
            dims = other.dims;
            fmats = other.fmats; // std::vector deep-copies automatically
        }
        return *this;
    }

    //------------------------------------------------------------------
    // Reassign fmat
    //------------------------------------------------------------------
    void reassign_fmat(int mode, std::vector<T> new_fmat)
    {
        if (new_fmat.size() != fmats[mode - 1].size())
            throw std::invalid_argument("New factor matrix isn't the correct size");
        fmats[mode - 1] = new_fmat;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    std::vector<T> get_fmat(int mode) const 
    {
        if (mode < 1)
            throw std::out_of_range("get_fmat: mode must be >= 1, got " + std::to_string(mode));
        if (mode > rank)
            throw std::out_of_range("get_fmat: mode " + std::to_string(mode) +
                                   " exceeds tensor rank of " + std::to_string(rank));
        return fmats[mode - 1];
    }
    std::vector<std::vector<T>> get_fmats() const {return fmats;}
    std::vector<int> get_dims() const{ return dims;}
    int get_factor_rank() const { return factor_rank;}
    int get_nnz() const { return nnz_entries;}
    int get_total_bits_needed() const { return num_bits;}

    //------------------------------------------------------------------
    // Destructor: free factor matrices
    //------------------------------------------------------------------
    ~Tensor() = default; // std::vector members clean up automatically
};

#endif // TENSOR_IMPL_H
