#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <unordered_map>
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
// Namespace
// ==========================
namespace stats {
    // The probability of a data point falling OUTSIDE these ranges 
    // for a standard normal distribution.
    constexpr double PROB_OUTSIDE_3SD = 0.002699796063; // ~1 in 370
    constexpr double PROB_OUTSIDE_4SD = 0.000063342484; // ~1 in 15,787
    constexpr double PROB_OUTSIDE_5SD = 0.000000573303; // ~1 in 1,744,278
    
    // The thresholds used in your find_anomalies_mapped function
    constexpr float SIGMA_3 = 3.0f;
    constexpr float SIGMA_4 = 4.0f;
    constexpr float SIGMA_5 = 5.0f;
}

// ==========================
// Structs
// ==========================

// Represents a single nonzero entry in an N-dimensional sparse tensor.
template<typename T>
struct NNZ_Entry {
    std::vector<int> coords; // Replaces i, j, k
    T value;
};

struct MADResult {
    float median;
    float mad_sigma; // This is the "robust" standard deviation
};

// Represents the header of a binary tensor file
// Note: In the binary file, the 'dims' array should strictly follow this struct.
struct TensorHeader {
    int32_t rank;    // Number of dimensions (modes)
    int64_t nnz;     // Number of non-zeros
    // In binary format, write 'rank' integers for dimensions immediately after this struct
};

// Structure to hold statistics results
struct StatsResult {
    float mean;
    float std_dev;
};

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
//   - target nnz: Number of non zeros
//   - min_val, max_val: value range for nonzeros
//   - block_size: size of each "dense block"
//   - max_blocks: stop condition for block attempts
//
// Uses random number generators to select starting coordinates and fill small blocks.
template<typename T>
std::vector<NNZ_Entry<T>> generate_block_sparse_tensor_nd(
    const std::vector<int>& dims,   // tensor dimensions [D0, D1, D2, ...]
    uint64_t target_nnz,            // target number of non-zero entries
    T min_val, T max_val,           // value range
    int block_size,                 // edge size of dense sub-blocks
    int max_blocks,                 // cap for random block sampling
    float dropout_rate = 0.5f       // per-entry dropout probability
) {
    int rank = dims.size();
    if (rank < 2)
        throw std::invalid_argument("Tensor rank must be >= 2.");
    if (target_nnz == 0)
        return {}; // Return empty if no non-zeros are requested
    if (min_val > max_val)
        throw std::invalid_argument("Invalid value range.");

    // Note: Removed the calculation of total_entries and the density check.
    std::vector<NNZ_Entry<T>> entries;
    // Reserve space based on the requested target_nnz
    entries.reserve(target_nnz);

    std::mt19937 rng(std::random_device{}());

    // Distributions
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
    // std::uniform_int_distribution<int> block_pos_dist(0, 1000000); // Not used in the original loop logic
    
    auto generate_value = [&]() -> T {
        if constexpr (std::is_integral_v<T>) {
            // Need to handle the case where T is integral but T is not the same as int
            // Using a generic way to handle integral/floating point
            using value_type = T;
            std::uniform_int_distribution<value_type> dist(min_val, max_val);
            return dist(rng);
        } else if constexpr (std::is_floating_point_v<T>) {
            using value_type = T;
            std::uniform_real_distribution<value_type> dist(min_val, max_val);
            return dist(rng);
        } else {
            throw std::invalid_argument("Unsupported type for value generation.");
        }
    };

    // Compute stride = block spacing between start positions
    int stride = block_size * 2;

    // Generate blocks until target nonzeros reached
    uint64_t nnz_count = 0;
    int blocks_attempted = 0;

    while (nnz_count < target_nnz && blocks_attempted < max_blocks) {
        blocks_attempted++;

        // Random block start per dimension
        std::vector<int> block_start(rank);
        for (int r = 0; r < rank; ++r) {
            // Calculate the maximum start position index based on stride
            // Ensure the start position allows for a full block_size extent without exceeding dims[r]
            int max_start_index = dims[r] - block_size;
            int limit = std::max(max_start_index / stride, 0);

            // The original logic was potentially flawed in calculating the limit, 
            // relying on (dims[r] - block_size) / stride. Keeping the spirit but ensuring a valid distribution.
            // If limit is 0, the only possible start is 0.
            std::uniform_int_distribution<int> start_dist(0, limit);
            block_start[r] = start_dist(rng) * stride;
        }

        // Recursive n-dimensional block filling (iterative implementation)
        std::vector<int> idx(rank, 0);
        bool done = false;

        while (!done && nnz_count < target_nnz) {
            // Compute the actual coordinates for this entry
            std::vector<int> coord(rank);
            for (int r = 0; r < rank; ++r)
                coord[r] = block_start[r] + idx[r];

            // Check bounds (redundant if block_start calculation is perfect, but safer to keep)
            bool in_bounds = true;
            for (int r = 0; r < rank; ++r)
                if (coord[r] >= dims[r]) { in_bounds = false; break; }

            // If entry is in bounds and passes dropout, add it
            if (in_bounds && dropout_dist(rng) > dropout_rate) {
                entries.push_back({coord, generate_value()});
                ++nnz_count;
            }

            // Increment n-dimensional index inside the block
            for (int r = rank - 1; r >= 0; --r) {
                idx[r]++;
                if (idx[r] < block_size)
                    break; // Moved to the next entry in the block
                idx[r] = 0; // Dimension wraps around
                if (r == 0) done = true; // Block finished
            }
        }
    }
    
    // Crucial step: If we overshot the target_nnz in the last block, truncate the vector.
    if (entries.size() > target_nnz) {
        entries.resize(target_nnz);
    }

    return entries;
}

// -----------------------------------------------------------
// N-Dimensional Binary File Reader
// -----------------------------------------------------------
/**
 * Reads a tensor from a binary file consisting ONLY of raw entries.
 * No header (rank/nnz/dims) is expected in the file.
 * returns A vector of NNZ_Entry objects (converted to 0-indexed).
 */
/**
 * Reads a tensor from a binary file consisting ONLY of raw entries.
 * Checks for incomplete "rows" (missing coordinates or values).
 * * @param filename Path to the binary file.
 * @param rank The number of dimensions (modes) of the tensor.
 * @param nnz The number of non-zero entries to read.
 * @return A vector of NNZ_Entry objects.
 */
template<typename T>
std::vector<NNZ_Entry<T>> read_tensor_file_binary(const std::string &filename, int rank, int64_t nnz) 
{
    std::ifstream file(filename, std::ios::binary);
    std::vector<NNZ_Entry<T>> entries;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return entries;
    }

    // Use reserve to avoid multiple reallocations while allowing for an early exit
    entries.reserve(nnz);

    const size_t coords_bytes = rank * sizeof(int32_t);
    const size_t value_bytes = sizeof(T);

    for(size_t i = 0; i < nnz; ++i) {
        NNZ_Entry<T> current_entry;
        current_entry.coords.resize(rank);
        
        // 1. Try to read the 'rank' coordinates
        file.read(reinterpret_cast<char*>(current_entry.coords.data()), coords_bytes);
        
        // Check if we managed to read all 'rank' coordinates
        if (file.gcount() < static_cast<std::streamsize>(coords_bytes)) {
            std::cerr << "Error: Entry #" << i << " is incomplete. "
                      << "Expected " << rank << " coordinates, but only found " 
                      << (file.gcount() / sizeof(int32_t)) << " indices." << std::endl;
            break; 
        }
        
        // 2. Try to read the '1' value
        file.read(reinterpret_cast<char*>(&current_entry.value), value_bytes);
        
        // Check if we managed to read the value associated with those coordinates
        if (file.gcount() < static_cast<std::streamsize>(value_bytes)) {
            std::cerr << "Error: Entry #" << i << " is incomplete. "
                      << "Read " << rank << " coordinates, but the value is missing." << std::endl;
            break;
        }
        
        // Convert from 1-indexed to 0-indexed
        for(auto& c : current_entry.coords) {
            c -= 1; 
        }

        entries.push_back(std::move(current_entry));
    }

    if (entries.size() < static_cast<size_t>(nnz)) {
        std::cerr << "Warning: Only " << entries.size() << " full entries were read out of " 
                  << nnz << " requested." << std::endl;
    }

    return entries;
}

/**
 * Filters the entry vector to remove any non-zeros with out-of-bounds coordinates.
 * Assumes entry_vec is already 0-indexed.
 * * @param entry_vec The vector of sparse entries to be cleaned.
 * @param dims The maximum dimensions for each mode.
 */
template<typename T>
void filter_invalid_entries(std::vector<NNZ_Entry<T>>& entry_vec, const std::vector<int>& dims) {
    if (entry_vec.empty() || dims.empty()) return;

    size_t initial_size = entry_vec.size();
    int rank = static_cast<int>(dims.size());

    // std::remove_if moves "invalid" entries to the end of the vector
    auto new_end = std::remove_if(entry_vec.begin(), entry_vec.end(), [&](const NNZ_Entry<T>& entry) {
        // Check if the entry has the correct number of dimensions
        if (entry.coords.size() != static_cast<size_t>(rank)) {
            return true; 
        }

        // Check bounds for each mode
        for (int i = 0; i < rank; ++i) {
            if (entry.coords[i] < 0 || entry.coords[i] >= dims[i]) {
                return true; // Mark for removal
            }
        }
        
        return false; // Keep this entry
    });

    // Actually shrink the vector size
    entry_vec.erase(new_end, entry_vec.end());

    size_t removed_count = initial_size - entry_vec.size();
    if (removed_count > 0) {
        std::cout << "[Validation] Removed " << removed_count 
                  << " entries with out-of-bounds indices." << std::endl;
    }
}

// Check if entry exists (N-dimensional)
template<typename T>
bool find_entry(const std::vector<NNZ_Entry<T>>& entry_vec, const std::vector<int>& coords, T val) {
    for (const auto& entry : entry_vec) {
        if (entry.value == val && entry.coords == coords) {
            return true;
        }
    }
    return false;
}

/**
 * N-Dimensional Binary Search with Optional Masking (-1)
 * Requires: entry_vec must be sorted by coords initially.
 * If remove is true, found value is set to 0.
 */
template<typename T>
bool find_entry_binary(std::vector<NNZ_Entry<T>>& entry_vec, const std::vector<int>& target_coords, T val, bool mask = false) {
    if (entry_vec.empty()) return false;

    // Use lower_bound for O(log N) search on coords
    auto it = std::lower_bound(entry_vec.begin(), entry_vec.end(), target_coords, 
        [](const NNZ_Entry<T>& entry, const std::vector<int>& target) {
            return entry.coords < target;
        });

    // Verify coordinates match and the current value matches the search target
    if (it != entry_vec.end() && it->coords == target_coords && it->value == val) {
        if (mask) {
            it->value = T(0); // Maintain sort order by leaving coords alone
        }
        return true;
    }

    return false;
}

// Print the max and min indices for each mode
// Used to check if the .bin file was parsed correctly
template<typename T>
void print_tensor_stats(const std::vector<NNZ_Entry<T>>& entry_vec) {
    if (entry_vec.empty()) {
        std::cout << "Tensor is empty (0 non-zeros)." << std::endl;
        return;
    }

    // Determine rank from the first entry
    int rank = entry_vec[0].coords.size();
    
    // Initialize min/max vectors with the coordinates of the first entry
    std::vector<int> min_indices = entry_vec[0].coords;
    std::vector<int> max_indices = entry_vec[0].coords;

    // Iterate through all entries to find bounds
    for (const auto& entry : entry_vec) {
        for (int i = 0; i < rank; ++i) {
            if (entry.coords[i] < min_indices[i]) min_indices[i] = entry.coords[i];
            if (entry.coords[i] > max_indices[i]) max_indices[i] = entry.coords[i];
        }
    }

    // Output results
    std::cout << "========================================" << std::endl;
    std::cout << "Tensor Index Statistics (Internal 0-indexed)" << std::endl;
    std::cout << "Total NNZ: " << entry_vec.size() << std::endl;
    for (int i = 0; i < rank; ++i) {
        std::cout << "  Mode " << (i + 1) << ": Min Index = " << min_indices[i] 
                  << ", Max Index = " << max_indices[i] << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

/**
 * N-Dimensional MTTKRP (Flattened Row-Major)
 * mode: The target mode to update (1-indexed)
 * (M): Flattened target factor matrix (Size: dim_size * R)
 * factors: Vector of flattened factor matrices (Each size: dim_size_m * R)
 * (R): The decomposition rank
 * entries: The sparse tensor non-zero entries
 */
template<typename T>
T* MTTKRP_Naive(int mode, T* M, const std::vector<T*>& factors, int R,
                   const std::vector<NNZ_Entry<T>>& entries) 
{
    int target_dim_idx = mode - 1;   // 0-indexed mode
    int num_modes = factors.size(); // Tensor order (N)

    for (const auto& entry : entries) {
        int target_row = entry.coords[target_dim_idx];
        T val = entry.value;

        // The target row in a row-major flattened matrix starts at: target_row * R
        int target_offset = target_row * R;

        for (int r = 0; r < R; ++r) {
            T product_sum = 1;

            // Multiply contributions from all OTHER modes
            for (int m = 0; m < num_modes; ++m) {
                if (m == target_dim_idx) continue;
                
                int row_idx = entry.coords[m];
                // Accessing factors[m](row_idx, r) -> factors[m][row_idx * R + r]
                product_sum *= factors[m][row_idx * R + r];
            }

            // Accumulate into flattened target matrix M
            M[target_offset + r] += val * product_sum;
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

// Return true if matrices are exactly equal, false otherwise
template<typename T>
bool compare_matricies(T** m1, T** m2, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (m1[i][j] != m2[i][j]) return false;
    return true;
}

// Compares floating point/double matricies and outputs the absolute difference
template<typename T>
double compare_matricies_float(T** m1, T** m2, int rows, int cols) {
    double diff = 0.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            diff += std::abs(m1[i][j] - m2[i][j]);
    return diff / (rows * cols);
}

// Convert matrix to array
template<typename T>
T* mat_to_array(T** m1, int rows, int cols) {
    T* arr = new T[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            arr[i * cols + j] = m1[i][j];
        }
    }
    return arr;
}

template<typename T>
void print_matrix_to_file(const T* data, size_t rows, size_t cols, 
                       const std::string& filename, const std::string& mat_name,
                       int precision = 6, int width = 12) {
    std::ofstream outfile(filename, std::ios::app);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Set formatting
    outfile << std::fixed << std::setprecision(precision);
    outfile << mat_name << "\n";
    
    // Write matrix
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;  // Row-major indexing
            outfile << std::setw(width) << data[idx];
            
            if (j < cols - 1) {
                outfile << " ";  // Space between columns
            }
        }
        outfile << "\n";  // Newline after each row
    }
    outfile << "\n\n"; 
    
    outfile.close();
    std::cout << "Matrix written to " << filename 
              << " (" << rows << "x" << cols << ")\n";
    
}

template<typename T>
void print_differences_to_file(const T* m1, const T* m2, size_t rows, size_t cols, 
                       const std::string& filename, const std::string& m1_name,
                       const std::string& m2_name, int precision = 6, int width = 12) {
    std::ofstream outfile(filename, std::ios::app);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Set formatting
    outfile << std::fixed << std::setprecision(precision);
    
    // Write matrix (terminate early if there are too many differences)
    int line_count = 0;
    bool terminated_early = false;
    for (size_t i = 0; i < rows; i++) {
        if (line_count > 10000){
            terminated_early = true;
            break;
        }
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;  // Row-major indexing
            if(m1[idx] != m2[idx]){
                outfile << m1_name << " val: " << m1[idx] << ", " << m2_name << " val: " << m2[idx] 
                << " at idx (" << i << ", " << j << ")\n";
                line_count ++;
            }
        }
    }
    if(terminated_early) outfile << "\n couldn't write all of the differences to outfile";
    outfile << "\n\n"; 
    
    outfile.close();
    std::cout << "Differences written to " << filename 
              << " (" << rows << "x" << cols << ")\n";
    
}

template<typename T>
void print_differences_to_file_float(const T* m1, const T* m2, size_t rows, size_t cols, 
                               const std::string& filename, const std::string& m1_name,
                               const std::string& m2_name, T epsilon = 1e-4, 
                               int precision = 6, int width = 12) {
    std::ofstream outfile(filename, std::ios::app);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    outfile << "Comparing " << m1_name << " and " << m2_name << " (" << rows << "x" << cols << ")\n";
    outfile << "Threshold (Epsilon): " << epsilon << "\n";
    outfile << std::fixed << std::setprecision(precision);
    
    size_t diff_count = 0;
    size_t max_logs = 1000; // Prevent the file from becoming multi-gigabyte

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            T val1 = m1[idx];
            T val2 = m2[idx];
            
            // Calculate Absolute Difference
            T abs_diff = std::abs(val1 - val2);
            
            // Calculate Relative Difference (avoid division by zero)
            T max_val = std::max(std::abs(val1), std::abs(val2));
            T rel_diff = (max_val > 0) ? (abs_diff / max_val) : 0;

            // Trigger if absolute OR relative error is too high
            if (abs_diff > epsilon && rel_diff > epsilon) {
                diff_count++;
                if (diff_count <= max_logs) {
                    outfile << "Mismatch at (" << std::setw(6) << i << ", " << std::setw(3) << j << ") | "
                            << m1_name << ": " << std::setw(width) << val1 << " | "
                            << m2_name << ": " << std::setw(width) << val2 << " | "
                            << "Abs Diff: " << abs_diff << "\n";
                }
            }
        }
    }

    if (diff_count > max_logs) {
        outfile << "... and " << (diff_count - max_logs) << " more differences omitted.\n";
    }

    outfile << "Total Mismatches: " << diff_count << "\n";
    outfile << "--------------------------------------------------------\n\n";
    outfile.close();

    if (diff_count > 0) {
        std::cout << "[Warning] Found " << diff_count << " differences. Details written to " << filename << std::endl;
    } else {
        std::cout << "[Success] No significant differences found between " << m1_name << " and " << m2_name << std::endl;
    }
}

// --- Array Utilities ---

// Deep copy of a array (allocate and copy all entries)
template<typename T>
T* create_and_copy_array(T* basis, int size) {
    T* copy_arr = new T[size];
    for (int i = 0; i < size; ++i)
        copy_arr[i] = basis[i];
    return copy_arr;
}

// Return true if arrays are exactly equal, false otherwise
template<typename T>
bool compare_arrays(T* a1, T* a2, int size) {
    for (int i = 0; i < size; i++)
            if (a1[i] != a2[i]) return false;
    return true;
}

// Compares floating point/double arrays and outputs the absolute difference
template<typename T>
double compare_arrays_float(T* a1, T* a2, int size) {
    double diff = 0.0;
    for (int i = 0; i < size; i++)
            diff += std::abs(a1[i] - a2[i]);
    return diff / size;
}


// --- Printing Helpers ---

// Print all entries in a sparse tensor
template<typename T>
void print_entry_vec(const std::vector<NNZ_Entry<T>>& entry_vec) {
    if(entry_vec.size() == 0){
        std::cerr << "empty entry vec\n";
        return;
    }
    if(entry_vec[0].coords.size() > 7){
        std::cerr << "Does not accept more than seven dimensions\n";
        return; 
    }
    std::vector<std::string> modes = {"one", "two", "three", "four", "five", "six", "seven"};
    for (size_t i = 0; i < entry_vec.size(); ++i) {
        for(int j = 0; j < entry_vec[i].coords.size(); j++){
            std::cout << "mode " << modes[j] << " idx: " << entry_vec[i].coords[j] << ", ";
        }
        std::cout << "val: " << entry_vec[i].value << "\n";
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

// ==========================
// Statistical Functions
// ==========================

/**
 * Calculates the mean and standard deviation of a float vector.
 * Uses OpenMP for parallel reduction, matching the style of tensor_impl.h.
 */
StatsResult calculate_statistics(const std::vector<float>& data) {
    if (data.empty()) {
        return {0.0f, 0.0f};
    }

    size_t n = data.size();
    double sum = 0.0;
    double sq_sum = 0.0;

    // Parallel reduction for sum and squared sum
    // Using double for accumulation to minimize precision loss
    #pragma omp parallel for reduction(+:sum, sq_sum)
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
        sq_sum += data[i] * data[i];
    }

    float mean = static_cast<float>(sum / n);
    double variance = (sq_sum / n) - (mean * mean);
    
    // Ensure variance is non-negative (can happen due to floating point epsilon)
    if (variance < 0) variance = 0;

    return {mean, std::sqrt(static_cast<float>(variance))};
}


std::vector<float> clean_data(const std::vector<float>& data, StatsResult stats) {

    float mean = stats.mean;
    float sd = stats.std_dev;

    std::vector<float> new_data;

    // Pre-calculate threshold
    float thresh = 4.0f * sd;

    for (const float& val : data) {
        float diff = std::abs(val - mean);

        // Check 4 SD
        if (diff < thresh) {
            new_data.push_back(val);
        }
    }

    return new_data;
}

/**
 * Robustly filters outliers from MTTKRP runtimes using Median Absolute Deviation.
 * @param data The input timing data.
 * @param sigma_threshold Number of MAD-sigmas to allow (default 4.0).
 * @param epsilon Absolute difference floor to ignore micro-jitter.
 * @return A vector containing the "clean" runtimes.
 */
std::vector<float> clean_data_mad(const std::vector<float>& data, float sigma_threshold = 4.0f, float epsilon = 0.005f) {
    if (data.empty()) return {};
    if (data.size() == 1) return data;

    // 1. Find the Median
    std::vector<float> sorted_data = data;
    size_t n = sorted_data.size();
    std::sort(sorted_data.begin(), sorted_data.end());
    
    float median = (n % 2 == 0) ? (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0f : sorted_data[n/2];

    // 2. Calculate Absolute Deviations from Median
    std::vector<float> deviations;
    deviations.reserve(n);
    for (float val : data) {
        deviations.push_back(std::abs(val - median));
    }

    // 3. Find the Median of Absolute Deviations (MAD)
    std::sort(deviations.begin(), deviations.end());
    float mad = (n % 2 == 0) ? (deviations[n/2 - 1] + deviations[n/2]) / 2.0f : deviations[n/2];

    // 4. Calculate the Robust Sigma (MAD * Consistency Factor)
    // 1.4826 makes MAD equivalent to SD for normal distributions
    float robust_sigma = mad * 1.4826f;
    
    // 5. Define the threshold
    // We add epsilon to prevent flagging nanosecond jitter when sigma is tiny
    float thresh = sigma_threshold * robust_sigma;

    std::vector<float> new_data;
    new_data.reserve(n);

    for (const float& val : data) {
        float diff = std::abs(val - median);

        // Keep data if it's within the statistical threshold
        // OR if the total difference is smaller than our absolute noise floor (epsilon)
        if (diff <= thresh || diff <= epsilon) {
            new_data.push_back(val);
        }
    }

    return new_data;
}

/**
 * Calculates the count of anomalies at 3, 4, and 5 standard deviations.
 * The input dataset
 * The pre-calculated mean and standard deviation
 * A map where keys are {3, 4, 5} and values are the count of anomalies
 */
std::unordered_map<int, int> find_anomalies(const std::vector<float>& latencies, StatsResult stats, float min_meaningful_ms = 0.005f) // e.g., ignore anything smaller than 5 microseconds 
{
    std::unordered_map<int, int> anomaly_counts = {{3, 0}, {4, 0}, {5, 0}};
    
    // If SD is zero, there are no statistical anomalies
    if (stats.std_dev <= 0.0f) return anomaly_counts;

    float mean = stats.mean;
    float sd = stats.std_dev;

    for (const float& val : latencies) {
        float diff = std::abs(val - mean);

        // Only count if it's statistically significant AND physically meaningful
        if (diff > (3.0f * sd) && diff > min_meaningful_ms) {
            anomaly_counts[3]++;
            if (diff > (4.0f * sd)) {
                anomaly_counts[4]++;
                if (diff > (5.0f * sd)) {
                    anomaly_counts[5]++;
                }
            }
        }
    }
    return anomaly_counts;
}

// Helper function to calculate nCr (Binomial Coefficient)
double binomialCoefficient(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k; // Take advantage of symmetry

    double res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - k + i) / i;
    }
    return res;
}

// Function to calculate Binomial Probability
double binomialProbability(int n, int k, double p) {
    double nCr = binomialCoefficient(n, k);
    double prob = nCr * std::pow(p, k) * std::pow(1.0 - p, n - k);
    return prob;
}

MADResult calculate_mad(std::vector<float> data) {
    if (data.empty()) return {0.0f, 0.0f};

    // 1. Calculate Median
    size_t n = data.size();
    std::sort(data.begin(), data.end());
    float median = (n % 2 == 0) ? (data[n/2 - 1] + data[n/2]) / 2.0f : data[n/2];

    // 2. Calculate Absolute Deviations
    std::vector<float> deviations;
    deviations.reserve(n);
    for (float val : data) {
        deviations.push_back(std::abs(val - median));
    }

    // 3. Calculate Median of Deviations (MAD)
    std::sort(deviations.begin(), deviations.end());
    float mad = (n % 2 == 0) ? (deviations[n/2 - 1] + deviations[n/2]) / 2.0f : deviations[n/2];

    // 4. Convert to a "Normal-equivalent" Sigma
    // 1.4826 is the scale factor for normal distribution consistency
    return {median, mad * 1.4826f};
}

std::unordered_map<int, int> find_anomalies_mad(const std::vector<float>& data, MADResult res, float epsilon = 0.005f) {
    std::unordered_map<int, int> anomaly_counts = {{3, 0}, {4, 0}, {5, 0}};

    float thresh3 = 3.0f * res.mad_sigma;
    float thresh4 = 4.0f * res.mad_sigma;
    float thresh5 = 5.0f * res.mad_sigma;

    for (const float& val : data) {
        float diff = std::abs(val - res.median);

        // Apply both the statistical threshold (MAD) and the absolute threshold (epsilon)
        if (diff > thresh3 && diff > epsilon) {
            anomaly_counts[3]++;
            if (diff > thresh4) {
                anomaly_counts[4]++;
                if (diff > thresh5) {
                    anomaly_counts[5]++;
                }
            }
        }
    }

    return anomaly_counts;
}

#endif

