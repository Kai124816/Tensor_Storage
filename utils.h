#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <type_traits>

// Maximum 64 bit unsigned integer (extern to avoid multiple definitions)
extern __uint128_t limit = 0xFFFFFFFFFFFFFFFF;

// Print bits functions
void print_lsb_bits(__uint128_t value, int x);
void print_uint64(uint64_t value, int x);
std::string uint128_to_sci_string(__uint128_t value, int precision = 15);

// Mathematical functions
int byte_size(int r, int c, int d);
template<typename S>
int floor_log2(S x);
template<typename S>
int ceiling_log2(S x);

// Non-zero entry struct
template<typename T>
struct NNZ_Entry {
    int i;
    int j;
    int k;
    T value;            
};

template<typename T>
std::vector<NNZ_Entry<T>> generate_block_sparse_tensor(
    int rows, int cols, int depth,
    float density, T min_val, T max_val,
    int block_size = 8, int max_blocks = 20000);

template<typename T>
void find_entry(std::vector<NNZ_Entry<T>> entry_vec, int r, int c, int d, T val);

// MTTKRP function
template<typename T>
T** MTTKRP(
    int mode,
    T** M,
    T** A, T** B,
    int R,
    const std::vector<NNZ_Entry<T>>& entries);

// Matrix functions
template<typename T>
T** create_and_copy_matrix(T** basis, int rows, int cols);

template<typename T>
int compare_matricies(T** m1, T** m2, int rows, int cols);

template<typename T>
int compare_matricies_id(T** m1, T** m2, int rows, int cols);

// Print functions
template<typename T>
void print_entry_vec(const std::vector<NNZ_Entry<T>>& entry_vec);

template<typename T>
void print_matrix(T** matrix, int rows, int cols, int width = 6);

template<typename S>
int floor_log2(S x) 
{
    int res = -1;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}

template<typename S>
int ceiling_log2(S x)
{
    if (x == 1) return 0;

    int res = 0;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}

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

    __uint128_t total_entries = static_cast<__uint128_t>(rows) * cols * depth;
    uint64_t target_nnz = static_cast<uint64_t>(total_entries * density);

    std::vector<NNZ_Entry<T>> entries;
    entries.reserve(target_nnz);

    std::mt19937 rng(std::random_device{}());

    int stride = block_size * 2;
    std::uniform_int_distribution<int> i_start(0, std::max((rows - block_size) / stride, 1));
    std::uniform_int_distribution<int> j_start(0, std::max((cols - block_size) / stride, 1));
    std::uniform_int_distribution<int> k_start(0, std::max((depth - block_size) / stride, 1));

    std::uniform_int_distribution<T> int_val_dist(min_val, max_val);
    std::uniform_real_distribution<double> real_val_dist(static_cast<double>(min_val), static_cast<double>(max_val));
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);

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
                    if (dropout_dist(rng) < 0.5f) continue;

                    entries.push_back({i, j, k, generate_value()});
                    ++nnz_count;
                }
            }
        }
    }

    return entries;
}

template<typename T>
void find_entry(std::vector<NNZ_Entry<T>> entry_vec, int r, int c, int d, T val) {
    for (size_t i = 0; i < entry_vec.size(); i++) {
        if (entry_vec[i].i == r && entry_vec[i].j == c && entry_vec[i].k == d && entry_vec[i].value == val) {
            std::cout << "input exists in entry vec\n";
            return;
        }
    }
    std::cout << "input does not exist in entry vec\n";
    std::cout << r << " " << c << " " << d << "\n";
}

template<typename T>
T** MTTKRP(
    int mode,
    T** M,
    T** A, T** B,
    int R,
    const std::vector<NNZ_Entry<T>>& entries) 
{
    for (const auto& entry : entries) {
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

template<typename T>
T** create_and_copy_matrix(T** basis, int rows, int cols) 
{
    T** copy_matrix = new T*[rows];
    for (int i = 0; i < rows; ++i)
        copy_matrix[i] = new T[cols];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            copy_matrix[i][j] = basis[i][j];
        }
    }
    return copy_matrix;
}

template<typename T>
int compare_matricies(T** m1, T** m2, int rows, int cols) 
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (m1[i][j] != m2[i][j]) return 0;
        }
    }
    return 1;
}

template<typename T>
int compare_matricies_id(T** m1, T** m2, int rows, int cols) 
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (m1[i][j] != m2[i][j]) {
                std::cout << "values " << m1[i][j] << " and " << m2[i][j] << " dont match\n";
            }
        }
    }
    return 1;
}

template<typename T>
int compare_matricies_id(T** m1, T** m2, int rows, int cols) 
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (m1[i][j] != m2[i][j]) {
                std::cout << "values " << m1[i][j] << " and " << m2[i][j] << " dont match\n";
            }
        }
    }
    return 1;
}

template<typename T>
T* vectorize_matrix(T** m1, int rows, int cols) 
{
    T* ret_vector = new T[rows * cols];
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            ret_vector[i * cols + j] = m1[i][j];
        }
    }

    return ret_vector;
}

template<typename T>
void vector_to_array(T* a1, std::vector<T> v1)
{
    for(int i = 0; i < v1.size(); i++){
        a1[i] = v1[i];
    }
}


template<typename T>
void print_entry_vec(const std::vector<NNZ_Entry<T>>& entry_vec) {
    for (size_t i = 0; i < entry_vec.size(); ++i) {
        try {
            std::cout << "i:" << entry_vec[i].i
                      << " j:" << entry_vec[i].j
                      << " k:" << entry_vec[i].k
                      << " val:" << entry_vec[i].value << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error printing entry at index " << i << ": " << e.what() << "\n";
        }
    }
}

template<typename T>
void print_matrix(T** matrix, int rows, int cols, int width) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(width) << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
}


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

