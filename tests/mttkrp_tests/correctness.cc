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


//Tests MTTKRP using a randomly generated tensor
template<typename T, typename S>
void test_mttkrp_synthetic(std::string version, int user_mode, int nnz, int rank, std::vector<int> dims)
{
    std::cout << "Testing MTTKRP using BLCO tensor";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";
    int default_decomp_rank = 16;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = (0.05 * min_dim) + 1;
    int max_blocks = (nnz + block_size - 1) / block_size;
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,
    block_size,max_blocks);

    std::vector<std::vector<T>> default_fmats;
    Alto_Tensor<T,S> alto;
    Blco_Tensor<T,S> blco;
    bool is_csr = false;
    if(version == "alto"){
        alto = Alto_Tensor<T,S>(test_vec, dims, default_decomp_rank);
        default_fmats = alto.get_fmats();
    }
    else{
        blco = Blco_Tensor<T,S>(test_vec, dims, default_decomp_rank);
        default_fmats = blco.get_fmats();
        is_csr = blco.get_total_bits_needed() > 74;
    }

    auto run_mode = [&](int mode) {
        std::cout << "\n--- Running Mode " << mode << " ---\n";
        std::vector<T> input_matrix = default_fmats[mode - 1];
        std::vector<T> test_matrix = MTTKRP_Naive<T>(mode, input_matrix, default_fmats, 
        default_decomp_rank, test_vec);

        std::vector<float> temp_vec = {0.0f};
        bool valid_version = false;
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_DEFAULT)
    if(version == "default"){
        bool valid_version = true;
        Initialize_MTTKRP<T, S>(mode, blco, temp_vec, 1);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_IN_PROGRESS)
    if(version == "in_progress"){
        bool valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_in_progress(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_NAIVE)
    if(version == "naive"){
        bool valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_Naive(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V1)
    if(version == "v1"){
        bool valid_version = true;
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v1(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V2)
    if(version == "v2"){
        bool valid_version = true;
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v2(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_VECTORIZED)
    if(version == "vectorized"){
        bool valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_VEC(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_ALTO)
    if(version == "alto"){
        bool valid_version = true;
        alto.MTTKRP_Parallel(mode);
    }
#endif
    else if(!valid_version){
        std::cerr << "invalid version or version not compiled in\n";
        return;
    }


    std::vector<std::vector<T>> factor_matricies;
    std::vector<T> modified_fmat;
    if(version != "alto"){
        factor_matricies = blco.get_fmats();
        modified_fmat = factor_matricies[mode - 1];
        blco.reassign_fmat(mode, default_fmats[mode - 1]);
        if(test_matrix == modified_fmat){
            std::cout << "tests passed, checksum: " << 
            std::accumulate(test_matrix.begin(), test_matrix.end(), 0) << "\n";
            return;
        }
        else{
            std::cout << "tests Failed" << "\n";
            print_differences_to_file(modified_fmat, test_matrix, dims[mode - 1], default_decomp_rank, 
            "diff.txt", "kernel ouput", "correct output");
            return;
        } 
    }
    else{
        factor_matricies = alto.get_fmats();
        modified_fmat = factor_matricies[mode - 1];
        alto.reassign_fmat(mode, default_fmats[mode - 1]);
        if(test_matrix == modified_fmat){
            std::cout << "tests passed, checksum: " << 
            std::accumulate(test_matrix.begin(), test_matrix.end(), 0ULL) << "\n";
            return;
        }
        else{
            std::cout << "tests Failed" << "\n";
            print_differences_to_file(alto.get_fmats()[mode - 1], test_matrix, dims[mode - 1], default_decomp_rank, 
            "diff.txt", "kernel ouput", "correct output");
            return;
        } 
    }
    };

    if (user_mode == -1) {
        for (int m = 1; m <= rank; ++m) {
            run_mode(m);
        }
    } else {
        run_mode(user_mode);
    }
}

//Tests MTTKRP using a tensor generated by a file
template<typename T, typename S>
void test_mttkrp_file(std::string version, std::string filename, 
int user_mode, int nnz, int rank, std::vector<int> dims)
{
    std::cout << "Testing MTTKRP using BLCO tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";
    int default_decomp_rank = 16;

    std::vector<std::vector<T>> default_fmats;
    int fmat_size;
    T min_val = static_cast<T>(0); T max_val = static_cast<T>(3);
    default_fmats.resize(rank);
    for(int i = 0; i < rank; i++){
        fmat_size = dims[i] * default_decomp_rank;
        default_fmats[i] = generate_random_array_seed(fmat_size, min_val, max_val, SEEDS[i]);
    }
    
    Blco_Tensor<T,S> blco;
    Alto_Tensor<T,S> alto;
    bool is_csr = false;
    if(version == "alto") alto = Alto_Tensor<T,S>(filename, nnz, dims, default_fmats, default_decomp_rank);
    else{
        blco = Blco_Tensor<T,S>(filename, nnz, dims, default_fmats, default_decomp_rank);
        is_csr = blco.get_total_bits_needed() > 74;
    } 

    auto run_mode = [&](int mode) {
        std::cout << "\n--- Running Mode " << mode << " ---\n";
        uint64_t test_checksum;
        std::vector<T> test_matrix;
        auto it = CHECKSUMS.find(filename);
        if(it == CHECKSUMS.end()){
            std::vector<NNZ_Entry<T>> test_vec = read_tensor_file_binary<T>(filename, rank, nnz);
            int min_dim = *(std::min_element(dims.begin(), dims.end()));
            int block_size = (0.05 * min_dim) + 1;
            int max_blocks = (nnz + block_size - 1) / block_size;
            std::vector<T> input_matrix = default_fmats[mode - 1];
            test_matrix = MTTKRP_Naive<T>(mode, input_matrix, default_fmats, 
            default_decomp_rank, test_vec);
            test_checksum = std::accumulate(test_matrix.begin(), test_matrix.end(), 0ULL);
        }
        else{
            std::vector<uint64_t> checksums = it->second;
            test_checksum = checksums[mode - 1];
        }
        
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
            return;
        }
        MTTKRP_BLCO_in_progress(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_NAIVE)
    if(version == "naive"){
        valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_Naive(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V1)
    if(version == "v1"){
        valid_version = true;
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v1(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_V2)
    if(version == "v2"){
        valid_version = true;
        if (rank != 3){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_v2(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_VECTORIZED)
    if(version == "vectorized"){
        valid_version = true;
        if (rank < 3 || rank > 5){
            std::cerr << "invalid rank\n";
            return;
        }
        MTTKRP_BLCO_VEC(mode, blco, temp_vec);
    }
#endif
#if defined(MTTKRP_VERSION_ALL) || defined(MTTKRP_VERSION_ALTO)
    if(version == "alto"){
        valid_version = true;
        alto.MTTKRP_Parallel(mode);
    }
#endif
    else if(!valid_version){
        std::cerr << "invalid version or version not compiled in\n";
        return;
    }

    std::vector<T> modified_fmat;
    if(version != "alto"){
        modified_fmat = blco.get_fmats()[mode - 1];
        uint64_t generated_checksum = std::accumulate(modified_fmat.begin(), modified_fmat.end(), 0ULL);
        blco.reassign_fmat(mode, default_fmats[mode - 1]);
        if(generated_checksum == test_checksum){
            std::cout << "Tests Passed!\n";
            return;
        }
        else{
            std::cout << "Tests Failed! MTTKRP Checksum: " << generated_checksum << " Correct Checksum: "
            << test_checksum << "\n";
            std::string diff_file;
            std::cout << "Enter y if you want a comprehensive difference file: ";
            std::cin >> diff_file;
            if(std::string(diff_file) != "y") return;
        }
    }
    else{
        modified_fmat = alto.get_fmats()[mode - 1];
        uint64_t generated_checksum = std::accumulate(modified_fmat.begin(), modified_fmat.end(), 0ULL);
        alto.reassign_fmat(mode, default_fmats[mode - 1]);
        if(generated_checksum == test_checksum){
            std::cout << "Tests Passed!\n";
            return;
        }
        else{
            std::cout << "Tests Failed! MTTKRP Checksum: " << generated_checksum << " Correct Checksum: "
            << test_checksum << "\n";
            std::string diff_file;
            std::cout << "Enter y if you want a comprehensive difference file: ";
            std::cin >> diff_file;
            if(std::string(diff_file) != "y") return;
        }
    }

    if(test_matrix.size() == 0){
        std::vector<NNZ_Entry<T>> test_vec = read_tensor_file_binary<T>(filename, rank, nnz);
        int min_dim = *(std::min_element(dims.begin(), dims.end()));
        int block_size = (0.05 * min_dim) + 1;
        int max_blocks = (nnz + block_size - 1) / block_size;
        std::vector<T> input_matrix = default_fmats[mode - 1];
        test_matrix = MTTKRP_Naive<T>(mode, input_matrix, default_fmats, 
        default_decomp_rank, test_vec);
    }
    print_differences_to_file(modified_fmat, test_matrix, dims[mode - 1], default_decomp_rank, 
    "diff.txt", "kernel ouput", "correct output");
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
    // Expected arguments: <version> <filename> <mode> <nnz> (three to five dimensions) <Type>
    if (argc < 9 || argc > 11) {
        std::cerr << "Usage: " << argv[0] 
                  << " <version> <filename> <mode ('all' or an integer)> <nnz> (three to five different dimensions) <Type>\n"
                  << " if you want to use a synthetically generated tensor use 'None' as your filename argument\n";
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
    int rank = argc - 6;
    std::string type = std::string(argv[argc - 1]);
    
    std::vector<int> dimensions;
    for(int i = 5; i < argc - 1; i++){
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
            test_mttkrp_synthetic<T, S>(version, mode, nnz, rank, dimensions);
        } else {
            test_mttkrp_file<T, S>(version, filename, mode, nnz, rank, dimensions);
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