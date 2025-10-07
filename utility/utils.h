#ifndef UTILS_H
#define UTILS_H

// ==========================
// Standard Library Includes
// ==========================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <omp.h>

// ==========================
// Constants
// ==========================
// Maximum 64-bit unsigned integer (as 128-bit type)
// Useful when checking tensor sizes that may exceed 64-bit range
extern __uint128_t limit = 0xFFFFFFFFFFFFFFFF;

// ==========================
// Structs
// ==========================

// Represents a single nonzero entry in a sparse tensor.
// Stores the coordinate (i, j, k) and its value.
template<typename T>
struct NNZ_Entry {
    int i;      // row index
    int j;      // column index
    int k;      // depth index
    T value;    // nonzero value at (i, j, k)
};

//Represents the header of a binary tensor file
struct TensorHeader {
    int32_t rows;
    int32_t cols;
    int32_t depth;
    int64_t nnz;
};

// ==========================
// Forward Declarations
// ==========================

// --- Bit / Printing Utilities ---
void print_lsb_bits(__uint128_t value, int x);  // print least-significant x bits of a 128-bit int
void print_uint64(uint64_t value, int x);       // print least-significant x bits of a 64-bit int
std::string uint128_to_sci_string(__uint128_t value, int precision = 15); // convert uint128_t to scientific notation string

// --- Math Utilities ---
int byte_size(int r, int c, int d);             // returns 64 or 128 depending on size of tensor
template<typename S> int floor_log2(S x);       // floor(log2(x))
template<typename S> int ceiling_log2(S x);     // ceil(log2(x))

// --- Sparse Tensor Utilities ---
template<typename T>
std::vector<NNZ_Entry<T>> generate_block_sparse_tensor(
    int rows, int cols, int depth,
    float density, T min_val, T max_val,
    int block_size = 8, int max_blocks = 20000);

template<typename T>
bool find_entry(std::vector<NNZ_Entry<T>> entry_vec, int r, int c, int d, T val); // check if specific entry exists

// --- MTTKRP (Matricized Tensor Times Khatri-Rao Product) ---
template<typename T>
T** MTTKRP(int mode, T** M, T** A, T** B, int R,
           const std::vector<NNZ_Entry<T>>& entries);

// --- Matrix Utilities ---
template<typename T> T** create_and_copy_matrix(T** basis, int rows, int cols); // deep copy a matrix
template<typename T> int compare_matricies(T** m1, T** m2, int rows, int cols); // exact equality check
template<typename T> int compare_matricies_id(T** m1, T** m2, int rows, int cols); // check and print differences
template<typename T> T* vectorize_matrix(T** m1, int rows, int cols);            // flatten matrix into 1D vector
template<typename T> void vector_to_array(T* a1, std::vector<T> v1);             // copy std::vector into raw array

// --- Printing Helpers ---
template<typename T> void print_entry_vec(const std::vector<NNZ_Entry<T>>& entry_vec); // print all tensor entries
template<typename T> void print_matrix(T** matrix, int rows, int cols, int width = 6); // print 2D matrix with formatting

// ==========================
// Template / Inline Definitions
// ==========================

// --- Math Utilities ---

// floor(log2(x)) for integer-like types
template<typename S>
int floor_log2(S x) {
    int res = -1;
    while (x) {
        x >>= 1;   // shift right until zero
        ++res;
    }
    return res;
}

// ceil(log2(x)) for integer-like types
template<typename S>
int ceiling_log2(S x) {
    if (x == 1) return 0;
    int res = 0;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}

// --- Sparse Tensor Utilities ---

// Generate a random sparse tensor with approximate block structure.
// Parameters:
//   - rows, cols, depth: tensor dimensions
//   - density: fraction of nonzeros (between 0 and 1)
//   - min_val, max_val: value range for nonzeros
//   - block_size: size of each "dense block"
//   - max_blocks: stop condition for block attempts
//
// Uses random number generators to select starting coordinates and fill small blocks.
template<typename T>
std::vector<NNZ_Entry<T>> generate_block_sparse_tensor(
    int rows, int cols, int depth,
    float density, T min_val, T max_val,
    int block_size, int max_blocks) 
{
    if (rows <= 0 || cols <= 0 || depth <= 0)
        throw std::invalid_argument("All dimensions must be positive.");
    if (density <= 0.0f || density > 1.0f)
        throw std::invalid_argument("Density must be in (0,1].");
    if (min_val > max_val)
        throw std::invalid_argument("Invalid value range.");

    // Compute target number of nonzeros
    std::cout<<"rows: "<<rows<<" cols: "<<cols<<" depth: "<<depth<<" density: "<<density<<"\n";
    uint64_t total_entries = static_cast<uint64_t>(rows) * cols * depth;
    double target_nnz = static_cast<double>(total_entries) * density;

    std::vector<NNZ_Entry<T>> entries;
    entries.reserve(target_nnz);

    std::mt19937 rng(std::random_device{}()); // random seed

    int stride = block_size * 2; // spacing between blocks
    std::uniform_int_distribution<int> i_start(0, std::max((rows - block_size) / stride, 1));
    std::uniform_int_distribution<int> j_start(0, std::max((cols - block_size) / stride, 1));
    std::uniform_int_distribution<int> k_start(0, std::max((depth - block_size) / stride, 1));

    // Value distributions depending on type T
    std::uniform_int_distribution<T> int_val_dist(min_val, max_val);
    std::uniform_real_distribution<double> real_val_dist(static_cast<double>(min_val), static_cast<double>(max_val));
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f); // skip ~50% randomly

    auto generate_value = [&]() -> T {
        if constexpr (std::is_integral<T>::value) {
            return int_val_dist(rng);
        } else if constexpr (std::is_floating_point<T>::value) {
            return static_cast<T>(real_val_dist(rng));
        } else {
            throw std::invalid_argument("Unsupported type for value generation.");
        }
    };

    uint64_t nnz_count = 0;
    int blocks_attempted = 0;

    // Keep generating random blocks until enough nonzeros are produced
    while (nnz_count < target_nnz && blocks_attempted < max_blocks) {
        blocks_attempted++;

        int i0 = i_start(rng) * stride;
        int j0 = j_start(rng) * stride;
        int k0 = k_start(rng) * stride;

        int bi_max = std::min(i0 + block_size, rows);
        int bj_max = std::min(j0 + block_size, cols);
        int bk_max = std::min(k0 + block_size, depth);

        for (int i = i0; i < bi_max && nnz_count < target_nnz; ++i) {
            for (int j = j0; j < bj_max && nnz_count < target_nnz; ++j) {
                for (int k = k0; k < bk_max && nnz_count < target_nnz; ++k) {
                    if (dropout_dist(rng) < 0.5f) continue; // skip with prob 0.5
                    entries.push_back({i, j, k, generate_value()});
                    ++nnz_count;
                }
            }
        }
    }

    return entries;
}

// Read a tensor from a file (1-indexed format).
// Each line: i j k value
// Converts to 0-indexed internally.
template<typename T>
std::vector<NNZ_Entry<T>> read_tensor_file(const std::string &filename, size_t maxLines = 0) 
{
    std::ifstream file(filename);
    std::vector<NNZ_Entry<T>> entries;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return entries;
    }

    int i, j, k;
    T value;
    size_t count = 0;

    while (file >> i >> j >> k >> value) {
        entries.push_back({i-1, j-1, k-1, value}); // shift to 0-index
        count++;
        if (maxLines > 0 && count >= maxLines) break;
    }

    return entries;
}

// Read a tensor from a binary file (1-indexed format).
// Each line: i j k value
// Converts to 0-indexed internally.
template<typename T>
std::vector<NNZ_Entry<T>> read_tensor_file_binary(const std::string &filename, size_t maxEntries = 0) 
{
    std::ifstream file(filename, std::ios::binary);
    std::vector<NNZ_Entry<T>> entries;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return entries;
    }

    // Read header
    TensorHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    size_t toRead = (maxEntries > 0) ? std::min<size_t>(header.nnz, maxEntries) 
                                     : header.nnz;

    entries.resize(toRead);

    // Read entries directly into vector
    file.read(reinterpret_cast<char*>(entries.data()), toRead * sizeof(NNZ_Entry<T>));

    if (!file) {
        std::cerr << "Error: Only read " << file.gcount() << " bytes from " << filename << "\n";
    }

    for (auto &e : entries) {
        e.i -= 1;
        e.j -= 1;
        e.k -= 1;
    }

    return entries;
}

// Return true if a given entry exists in the tensor.
template<typename T>
bool find_entry(std::vector<NNZ_Entry<T>> entry_vec, int r, int c, int d, T val) {
    for (size_t i = 0; i < entry_vec.size(); i++) {
        if (entry_vec[i].i == r && entry_vec[i].j == c &&
            entry_vec[i].k == d && entry_vec[i].value == val) {
            return true;
        }
    }
    return false;
}

// --- MTTKRP ---

// Perform matricized tensor times Khatri-Rao product (MTTKRP).
// Updates the factor matrix M depending on the mode:
//   mode 1 → accumulate into M[row][r]
//   mode 2 → accumulate into M[col][r]
//   mode 3 → accumulate into M[depth][r]
template<typename T>
T** MTTKRP(int mode, T** M, T** A, T** B, int R,
           const std::vector<NNZ_Entry<T>>& entries) 
{
    int count = 0;
    for (const auto& entry : entries) {
        count++;
        for (int r = 0; r < R; ++r) {
            T contrib = 0;
            if (mode == 1) {
                contrib = A[entry.j][r] * B[entry.k][r];
                M[entry.i][r] += entry.value * contrib;
            } else if (mode == 2) {
                contrib = A[entry.i][r] * B[entry.k][r];
                M[entry.j][r] += entry.value * contrib;
            } else if (mode == 3) {
                contrib = A[entry.i][r] * B[entry.j][r];
                M[entry.k][r] += entry.value * contrib;
            }
        }
    }
    return M;
}

// --- Matrix Utilities ---

// Deep copy of a matrix (allocate and copy all entries)
template<typename T>
T** create_and_copy_matrix(T** basis, int rows, int cols) {
    T** copy_matrix = new T*[rows];
    for (int i = 0; i < rows; ++i)
        copy_matrix[i] = new T[cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            copy_matrix[i][j] = basis[i][j];
    return copy_matrix;
}

// Return 1 if matrices are exactly equal, 0 otherwise
template<typename T>
int compare_matricies(T** m1, T** m2, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (m1[i][j] != m2[i][j]) return 0;
    return 1;
}

// Compare float matrices with tolerance epsilon (relative error).
// Uses OpenMP parallelization for speed.
bool compare_matricies_float(float** m1, float** m2, int rows, int cols, float epsilon = 1.0f) {
    bool equal = true;

    #pragma omp parallel for collapse(2) shared(equal)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (!equal) continue;  
            if (fabs(m1[i][j] - m2[i][j]) > epsilon * fabs(m1[i][j])) {
                #pragma omp atomic write
                equal = false;
            }
        }
    }
    return equal;
}

// Compare matrices and log mismatches
template<typename T>
int compare_matricies_id(T** m1, T** m2, int rows, int cols, std::ostream& out) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (m1[i][j] != m2[i][j]) 
                out << "values " << m1[i][j] << " and " << m2[i][j]
                    << " at index " << i << " " << j << " don't match\n";
    return 1;
}

// Same as above, but for floats with tolerance
int compare_matricies_id_float(float** m1, float** m2, int rows, int cols, 
                               std::ostream& out, float epsilon = 1.0) {
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            if (fabs(m1[i][j] - m2[i][j]) > epsilon * fabs(m1[i][j])) 
                out << "values " << m1[i][j] << " and " << m2[i][j]
                    << " at index " << i << " " << j << " don't match\n";
        }
    }
    return 1;
}

// Flatten a 2D matrix into a 1D vector (row-major)
template<typename T>
T* vectorize_matrix(T** m1, int rows, int cols) {
    T* ret_vector = new T[rows * cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            ret_vector[i * cols + j] = m1[i][j];
    return ret_vector;
}

// Flatten and repeat a matrix multiple times into one long vector
template<typename T>
T* vectorize_and_multiply_matrix(T** m1, int rows, int cols, int copies) {
    T* ret_vector = new T[copies * rows * cols];
    for (int copy = 0; copy < copies; copy++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ret_vector[copy * rows * cols + i * cols + j] = m1[i][j];
            }
        }
    }
    return ret_vector;
}

// Copy a flat vector back into a 2D matrix
template<typename T>
void copy_vector_to_matrix(T* &v1, T** &m1, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        m1[i/rows][i % rows] = v1[i];
    }
}

// Copy std::vector into raw C-style array
template<typename T>
void vector_to_array(T* a1, std::vector<T> v1) {
    for (int i = 0; i < v1.size(); i++)
        a1[i] = v1[i];
}

// --- Printing Helpers ---

// Print all entries in a sparse tensor
template<typename T>
void print_entry_vec(const std::vector<NNZ_Entry<T>>& entry_vec) {
    for (size_t i = 0; i < entry_vec.size(); ++i) {
        try {
            std::cout << "i:" << entry_vec[i].i
                      << " j:" << entry_vec[i].j
                      << " k:" << entry_vec[i].k
                      << " val:" << entry_vec[i].value << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error printing entry at index " << i
                      << ": " << e.what() << "\n";
        }
    }
}

// Print a dense 2D matrix with column width formatting
template<typename T>
void print_matrix(T** matrix, int rows, int cols, int width) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::cout << std::setw(width) << matrix[i][j] << " ";
        std::cout << "\n";
    }
}

// ==========================
// Non-template Implementations
// ==========================

// Print x least-significant bits of a 128-bit value
void print_lsb_bits(__uint128_t value, int x) {
    if (x < 1 || x > 128) {
        std::cerr << "Error: x must be between 1 and 128\n";
        return;
    }
    for (int i = x - 1; i >= 0; --i) {
        int bit = (value >> i) & 1;
        std::cout << bit;
    }
    std::cout << std::endl;
}

// Print x least-significant bits of a 64-bit value
void print_uint64(uint64_t value, int x) {
    if (x < 1 || x > 64) {
        std::cerr << "Error: x must be between 1 and 64\n";
        return;
    }
    for (int i = x - 1; i >= 0; --i) {
        int bit = (value >> i) & 1;
        std::cout << bit;
    }
    std::cout << std::endl;
}

// Convert 128-bit unsigned integer to scientific notation string
std::string uint128_to_sci_string(__uint128_t value, int precision) {
    if (value == 0) return "0.0e+0";

    __uint128_t temp = value;
    int exponent = 0;
    while (temp >= 10) {
        temp /= 10;
        ++exponent;
    }

    __uint128_t scale = 1;
    for (int i = 0; i < precision; ++i) scale *= 10;
    __uint128_t digits = (value * scale) / __uint128_t(std::pow(10, exponent));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision)
        << static_cast<double>(digits) / scale << "e+" << exponent;
    return oss.str();
}

int byte_size(int r, int c, int d) {
    if ((__uint128_t)r * c * d >= limit) return 128;
    return 64;
}

#endif

