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

//Maximum 64 bit unsigned integer
__uint128_t limit = 0xFFFFFFFFFFFFFFFF;


//Functions used to print bits

//Function used to print binary representation of 128 bit number
void print_lsb_bits(__uint128_t value, int x) {
    if (x < 1 || x > 128) {
        std::cerr << "Error: x must be between 1 and 128\n";
        return;
    }

    for (int i = x - 1; i >= 0; --i) {
        int bit = (value >> i) & 1;
        std::cout<<bit;
    }
    std::cout << std::endl;
}

//Function converts 128 bit integer to string
std::string uint128_to_sci_string(__uint128_t value, int precision = 15) {
    if (value == 0) return "0.0e+0";

    __uint128_t temp = value;
    int exponent = 0;
    while (temp >= 10) {
        temp /= 10;
        ++exponent;
    }

    // Now extract significant digits
    __uint128_t scale = 1;
    for (int i = 0; i < precision; ++i) scale *= 10;
    __uint128_t digits = (value * scale) / __uint128_t(std::pow(10, exponent));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision)
        << static_cast<double>(digits) / scale << "e+" << exponent;
    return oss.str();
}

//Function used to print binary representation of 64 bit number
void print_uint64(uint64_t value, int x) {
    if (x < 1 || x > 64) {
        std::cerr << "Error: x must be between 1 and 64\n";
        return;
    }

    for (int i = x - 1; i >= 0; --i) {
        int bit = (value >> i) & 1;
        std::cout<<bit;
    }
    std::cout << std::endl;
}


//Mathematical functions

//Determine if the ALTO tensor needs 64 bit or 128 bit indices
int byte_size(int r, int c, int d)
{
    if(r * c * d >= limit) return 128;
    return 64;
}

//Log Method for bit manipulation
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

//Log Method for bit manipulation
template<typename S>
int ceiling_log2(S x)
{
    if(x == 1) return 0;

    int res = 0;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}


//Non zero entry vector functions

//Non Zero entry struct
template<typename T>
struct NNZ_Entry {
    int i;
    int j;
    int k;
    T value;            
};

//Generate vector of nnz structs
template<typename T>
std::vector<NNZ_Entry<T>> generate_block_sparse_tensor(
    int rows, int cols, int depth,
    float density, T min_val, T max_val,
    int block_size = 8, int max_blocks = 20000)
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

    // Make block start locations sparser by using strides
    int stride = block_size * 2;
    std::uniform_int_distribution<int> i_start(0, std::max((rows - block_size) / stride, 1));
    std::uniform_int_distribution<int> j_start(0, std::max((cols - block_size) / stride, 1));
    std::uniform_int_distribution<int> k_start(0, std::max((depth - block_size) / stride, 1));

    std::uniform_int_distribution<T> int_val_dist(min_val, max_val);
    std::uniform_real_distribution<double> real_val_dist(static_cast<double>(min_val), static_cast<double>(max_val));
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);  // For skipping entries

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
                    // Add random dropout to avoid filling all entries in block
                    if (dropout_dist(rng) < 0.5f) continue;

                    entries.push_back({i, j, k, generate_value()});
                    ++nnz_count;
                }
            }
        }
    }

    return entries;
}

//Find entry within the vector
template<typename T>
void find_entry(std::vector<NNZ_Entry<T>> entry_vec, int r, int c, int d, T val)
{
    for(int i = 0; i < entry_vec.size(); i++){
        if(entry_vec[i].i == r && entry_vec[i].j == c && 
            entry_vec[i].k == d && entry_vec[i].value == val){
                std::cout<<"input exists in entry vec"<<"\n";
                return;
        }
    }
    std::cout<<"input does not exist in entry vec"<<"\n";
    std::cout<<r<<" "<<c<<" "<<d<<"\n";
}

//Print each entry in the entry vector
template<typename T>
void print_entry_vec(std::vector<NNZ_Entry<T>> entry_vec)
{
    for(int i = 0; i < entry_vec.size(); i++){
        std::cout<<"i:"<<entry_vec[i].i<<" j:"<<entry_vec[i].j<<" k:"<<entry_vec[i].k<<" val:"<<entry_vec[i].value<<"\n";
    }
}
