#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <numeric>
#include "checksums.h"
#include "../../utility/utils.h"
#include "../../tensor_implementations/tensor_impl.h"
#include "../../tensor_implementations/alto_impl.h"
#include "../../tensor_implementations/blco_impl.h"

#if !defined(MTTKRP_VERSION_DEFAULT) && !defined(MTTKRP_VERSION_IN_PROGRESS) && !defined(MTTKRP_VERSION_NAIVE) && !defined(MTTKRP_VERSION_V1) && !defined(MTTKRP_VERSION_V2) && !defined(MTTKRP_VERSION_VECTORIZED) && !defined(MTTKRP_VERSION_ALTO)
#define MTTKRP_VERSION_ALL
#endif

#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_DEFAULT)
#include "../../gpu_code/one_to_one/one_to_one_kernels.h"
#endif

#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_IN_PROGRESS)
#include "../../gpu_code/in_progress/in_progress.h"  
#endif

#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_NAIVE)
#include "../../gpu_code/naive/naive_kernels.h"  
#endif

#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V1) || defined(MTTKRP_VERSION_V2)
#include "../../gpu_code/old/old_kernels.h"  
#endif

#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_VECTORIZED)
#include "../../gpu_code/vectorized/vectorized_kernels.h"  
#endif


//Tests MTTKRP using preconstructed ALTO
template<typename T, typename S>
bool test_mttkrp_alto(std::vector<NNZ_Entry<T>>& entry_vec, Alto_Tensor<T,S>& alto, 
std::vector<std::vector<T>> fmats, int mode, int nnz, int rank, std::vector<int> dims, std::string tensor_file)
{
    uint64_t test_checksum;
    auto it = CHECKSUMS.find(tensor_file);
    if(it == CHECKSUMS.end()){
        int decomp_rank = fmats[0].size() / dims[0];
        std::vector<T> input_matrix = fmats[mode - 1];
        entry_vec = read_tensor_file_binary<T>(tensor_file, rank, nnz);
        std::vector<T> test_matrix = MTTKRP_Naive(mode, input_matrix, fmats, decomp_rank, entry_vec);
        test_checksum = std::accumulate(test_matrix.begin(), test_matrix.end(), 0ULL);
    }
    else{
        std::vector<uint64_t> checksums = it->second;
        test_checksum = checksums[mode - 1];
    }

    alto.MTTKRP_Parallel(mode);
    std::vector<T> generated_matrix = alto.get_fmats()[mode - 1];
    uint64_t generated_checksum = std::accumulate(generated_matrix.begin(), generated_matrix.end(), 0ULL);
    if(generated_checksum == test_checksum){
        return true;
    }
    return false; 
}

//Tests MTTKRP using preconstructed BLCO
template<typename T, typename S>
bool test_mttkrp_blco(std::string version, std::vector<NNZ_Entry<T>>& entry_vec, Blco_Tensor<T,S>& blco, 
std::vector<std::vector<T>>& fmats, int mode, int nnz, int rank, std::vector<int> dims, std::string tensor_file)
{
    uint64_t test_checksum;
    auto it = CHECKSUMS.find(tensor_file);
    if(it == CHECKSUMS.end()){
        int decomp_rank = fmats[0].size() / dims[0];
        std::vector<T> input_matrix = fmats[mode - 1];
        entry_vec = read_tensor_file_binary<T>(tensor_file, rank, nnz);
        std::vector<T> test_matrix = MTTKRP_Naive(mode, input_matrix, fmats, decomp_rank, entry_vec);
        test_checksum = std::accumulate(test_matrix.begin(), test_matrix.end(), 0ULL);
    }
    else{
        std::vector<uint64_t> checksums = it->second;
        test_checksum = checksums[mode - 1];
    }
    bool is_csr = blco.get_total_bits_needed() > 74;
    std::vector<float> temp_vec = {0.0f};
    bool valid_version = false;
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_DEFAULT)
    if(version == "default"){
        valid_version = true;
        Initialize_MTTKRP<T, S>(mode, blco, temp_vec, 1);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_IN_PROGRESS)
    if(version == "in_progress"){
        valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return false;
        }
        MTTKRP_BLCO_in_progress(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_NAIVE)
    if(version == "naive"){
        valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return false;
        }
        MTTKRP_BLCO_Naive(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V1)
    if(version == "v1"){
        valid_version = true;
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return false;
        }
        MTTKRP_BLCO_v1(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V2)
    if(version == "v2"){
        valid_version = true;
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return false;
        }
        MTTKRP_BLCO_v2(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_VECTORIZED)
    if(version == "vectorized"){
        valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return false;
        }
        MTTKRP_BLCO_VEC(mode, blco, temp_vec);
    }
#endif
    else if(!valid_version){
        std::cerr << "invalid version or version not compiled in\n";
        return false;
    }

    std::vector<T> generated_matrix = blco.get_fmats()[mode - 1];
    uint64_t generated_checksum = std::accumulate(generated_matrix.begin(), generated_matrix.end(), 0ULL);
    if(generated_checksum == test_checksum){
        return true;
    }
    return false; 
}

template<typename T, typename S>
void time_mttkrp_file(std::string version, std::string filename, int user_mode, 
int nnz, int rank, std::vector<int> dims, int iterations)
{   
    int default_decomp_rank = 16;

    std::vector<std::vector<T>> generated_fmats;
    generated_fmats.resize(rank);
    for(int i = 0; i < rank; i++) generated_fmats[i] = generate_random_array_seed(
    dims[i] * default_decomp_rank, static_cast<T>(0), static_cast<T>(3), SEEDS[i]);

    std::vector<NNZ_Entry<T>> file_test_vec = read_tensor_file_binary<T>(filename, rank, nnz);
    
    Alto_Tensor<T,S> alto;
    Blco_Tensor<T,S> blco;
    if(version == "alto") {
        alto = Alto_Tensor<T,S>(filename, nnz, dims, generated_fmats, default_decomp_rank);
    }
    else{
        blco = Blco_Tensor<T,S>(filename, nnz, dims, generated_fmats, default_decomp_rank);
    }
    bool is_csr = false;
    if(version != "alto") is_csr = blco.get_total_bits_needed() > 74;

    auto run_mode = [&](int mode) {
        std::cout << "\n--- Benchmarking Mode " << mode << " ---\n";
        bool test_passed;
        std::vector<float> times;
        
        if(version == "alto") test_passed = test_mttkrp_alto<T,S>(file_test_vec, alto, generated_fmats, mode, nnz, rank, dims, filename);
        else test_passed = test_mttkrp_blco<T,S>(version, file_test_vec, blco, generated_fmats, mode, nnz, rank, dims, filename);

        if(!test_passed){
            std::cout << "Test failed, terminating benchmarking for mode " << mode << "\n";
            return; 
        }
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_DEFAULT)
    if(version == "default"){
        Initialize_MTTKRP<T, S>(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_IN_PROGRESS)
    if(version == "in_progress"){
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_in_progress(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_NAIVE)
    if(version == "naive"){
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_Naive(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V1)
    if(version == "v1"){
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v1(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V2)
    if(version == "v2"){
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v2(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_VECTORIZED)
    if(version == "vectorized"){
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_VEC(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_ALTO)
    if(version == "alto"){
        float duration;
        for(int i = 0; i < iterations; i++){
            auto start = std::chrono::high_resolution_clock::now(); 
            alto.MTTKRP_Parallel(mode);
            auto stop = std::chrono::high_resolution_clock::now(); 
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()); 
            times.push_back(duration);
        }
    }
#endif
    else{
        std::cerr << "invalid version or version not compiled in\n";
        return;
    }

    if(version != "alto") blco.reassign_fmat(mode, generated_fmats[mode - 1]);
    else alto.reassign_fmat(mode, generated_fmats[mode - 1]);

    std::vector<float> cleaned_times = clean_data_mad(times);
    
    if(cleaned_times.size() < iterations / 2){
        std::cout << "Anomalous data discarding results\n"
        << "had to discard " << iterations - cleaned_times.size() << " out of " << 
        iterations << "\n";
        return;
    }

    StatsResult standard_result = calculate_statistics(cleaned_times);

    std::cout << "Kernel time data\n";
    std::cout << "Mean: " << standard_result.mean << " ms\n";
    std::cout << "Standard Deviation: " << standard_result.std_dev << " ms\n";
    };

    if (user_mode == -1) {
        for (int m = 1; m <= rank; ++m) {
            run_mode(m);
        }
    } else {
        run_mode(user_mode);
    }
}

template<typename T, typename S>
void time_mttkrp_synthetic(std::string version, int user_mode, 
int nnz, int rank, std::vector<int> dims, int iterations)
{   
    int default_decomp_rank = 16;

    std::vector<std::vector<T>> generated_fmats;
    generated_fmats.resize(rank);
    for(int i = 0; i < rank; i++) generated_fmats[i] = generate_random_array_seed(
    dims[i] * default_decomp_rank, static_cast<T>(0), static_cast<T>(3), SEEDS[i]);

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = (0.05 * min_dim) + 1;
    int max_blocks = (nnz + block_size - 1) / block_size;
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,block_size,max_blocks);
    
    Alto_Tensor<T,S> alto;
    Blco_Tensor<T,S> blco;
    if(version == "alto") {
        alto = Alto_Tensor<T,S>(test_vec, dims, default_decomp_rank);
    }
    else{
        blco = Blco_Tensor<T,S>(test_vec, dims, default_decomp_rank);
    }
    bool is_csr = false;
    if(version != "alto") is_csr = blco.get_total_bits_needed() > 74;

    auto run_mode = [&](int mode) {
        std::cout << "\n--- Benchmarking Mode " << mode << " ---\n";
        bool test_passed;
        std::vector<float> times;
        
        if(version == "alto") test_passed = test_mttkrp_alto<T,S>(test_vec, alto, generated_fmats, mode, nnz, rank, dims, "None");
        else test_passed = test_mttkrp_blco<T,S>(version, test_vec, blco, generated_fmats, mode, nnz, rank, dims, "None");

        if(!test_passed){
            std::cout << "Test failed, terminating benchmarking for mode " << mode << "\n";
            return; 
        }
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_DEFAULT)
    if(version == "default"){
        Initialize_MTTKRP<T, S>(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_IN_PROGRESS)
    if(version == "in_progress"){
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_in_progress(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_NAIVE)
    if(version == "naive"){
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_Naive(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V1)
    if(version == "v1"){
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v1(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V2)
    if(version == "v2"){
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v2(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_VECTORIZED)
    if(version == "vectorized"){
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_VEC(mode, blco, times, iterations);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_ALTO)
    if(version == "alto"){
        float duration;
        for(int i = 0; i < iterations; i++){
            auto start = std::chrono::high_resolution_clock::now(); 
            alto.MTTKRP_Parallel(mode);
            auto stop = std::chrono::high_resolution_clock::now(); 
            duration = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()); 
            times.push_back(duration);
        }
    }
#endif
    else{
        std::cerr << "invalid version or version not compiled in\n";
        return;
    }

    if(version != "alto") blco.reassign_fmat(mode, generated_fmats[mode - 1]);
    else alto.reassign_fmat(mode, generated_fmats[mode - 1]);
   
    std::vector<float> cleaned_times = clean_data_mad(times);
    
    if(cleaned_times.size() < iterations / 2){
        std::cout << "Anomalous data discarding results\n"
        << "had to discard " << iterations - cleaned_times.size() << " out of " << 
        iterations << "\n";
        return;
    }
    
    StatsResult standard_result = calculate_statistics(cleaned_times);

    std::cout << "Kernel time data\n";
    std::cout << "Mean: " << standard_result.mean << " ms\n";
    std::cout << "Standard Deviation: " << standard_result.std_dev << " ms\n";
    };

    if (user_mode == -1) {
        for (int m = 1; m <= rank; ++m) {
            run_mode(m);
        }
    } else {
        run_mode(user_mode);
    }
}

int main(int argc, char* argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--list-versions" || std::string(argv[1]) == "-v")) {
        std::cout << "Available versions:\n"
                  << "  default      - Default BLCO implementation\n"
                  << "  in_progress  - In-progress BLCO implementation\n"
                  << "  naive        - Naive BLCO implementation\n"
                  << "  v1           - V1 BLCO implementation (3D only)\n"
                  << "  v2           - V2 BLCO implementation (3D only)\n"
                  << "  vectorized   - Vectorized BLCO implementation\n"
                  << "  alto         - ALTO implementation\n";
        return 0;
    }

    // Expected arguments: <version> <filename> <mode> <nnz> (three to five dimensions) <iterations> <Type>
    if (argc < 10 || argc > 12) {
        std::cerr << "Usage: " << argv[0] 
                  << " <version> <filename> <mode ('all' or an integer)> <nnz> (three to five different dimensions) <iterations> <Type>\n"
                  << " if you want to use a synthetically generated tensor use 'None' as your filename argument\n"
                  << " To see a list of available versions, use: " << argv[0] << " --list-versions\n";
        return 1;
    }

    std::string version = std::string(argv[1]);
    std::string filename = std::string(argv[2]);
    std::string mode_str = std::string(argv[3]);
    int mode = -1;
    if (mode_str != "all") {
        mode = std::stoi(mode_str);
    }
    int nnz = std::stoi(argv[4]);
    int rank = argc - 7;
    int iterations = std::stoi(argv[argc - 2]);
    std::string type = std::string(argv[argc - 1]);
    
    std::vector<int> dimensions;
    for(int i = 5; i < argc - 2; i++){
        dimensions.push_back(std::stoi(argv[i]));
    }

    int bits_needed = 0;
    for(int i = 0; i < rank; i++){
        bits_needed += ceiling_log2(dimensions[i]);
    }

    auto run_tests = [&](auto dummy1, auto dummy2) {
        using T = decltype(dummy1);
        using S = decltype(dummy2);
        
        if (filename == "None") {
            time_mttkrp_synthetic<T, S>(version, mode, nnz, rank, dimensions, iterations);
        } else {
            time_mttkrp_file<T, S>(version, filename, mode, nnz, rank, dimensions, iterations);
        }
    };

    if(bits_needed <= 64){
        if(type == "int") run_tests(int{}, uint64_t{});
        else if(type == "float") run_tests(float{}, uint64_t{});
        else if(type == "long int") run_tests(0ULL, uint64_t{});
        else if(type == "double") run_tests(double{}, uint64_t{});
        else{ 
            std::cerr << "Unsupported type. The supported types are int, float, long int, and double\n";
            return 1;
        }
    }
    else{
        if(type == "int") run_tests(int{}, __uint128_t{});
        else if(type == "float") run_tests(float{}, __uint128_t{});
        else if(type == "long int") run_tests(0ULL, __uint128_t{});
        else if(type == "double") run_tests(double{}, __uint128_t{});
        else{ 
            std::cerr << "Unsupported type. The supported types are int, float, long int, and double\n";
            return 1;
        }
    }

    return 0;
}