#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cstdint>
#include "../../utility/utils.h"

// Generates and prints the checksums for every mode of the tensor using MTTKRP_Naive
template<typename T>
void generate_checksums(const std::string& filename, int rank, int nnz, const std::vector<int>& dims) {
    int default_decomp_rank = 16;
    T min_val = static_cast<T>(0); 
    T max_val = static_cast<T>(3);

    // Generate factor matrices identically to correctness/timing testing
    std::vector<std::vector<T>> default_fmats(rank);
    for(int i = 0; i < rank; i++){
        int fmat_size = dims[i] * default_decomp_rank;
        default_fmats[i] = generate_random_array_seed(fmat_size, min_val, max_val, SEEDS[i]);
    }

    std::cout << "Loading tensor: " << filename << " (NNZ: " << nnz << ", Rank: " << rank << ")...\n";
    std::vector<NNZ_Entry<T>> tensor_entries = read_tensor_file_binary<T>(filename, rank, nnz);

    std::cout << "Generating Checksums...\n";
    for (int mode = 1; mode <= rank; ++mode) {
        // Compute MTTKRP using the Naive implementation
        std::vector<T> test_matrix = MTTKRP_Naive<T>(mode, default_fmats[mode - 1], 
                                                     default_fmats, default_decomp_rank, tensor_entries);
        
        // Sum values using 0ULL (unsigned long long) to avoid integer limits overflowing
        uint64_t mode_checksum = std::accumulate(test_matrix.begin(), test_matrix.end(), 0ULL);
        
        std::cout << "Mode " << mode << ": " << mode_checksum << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Expected arguments: <filename> <nnz> <d1> <d2> ... <dn> <type>
    if (argc < 6 || argc > 9) {
        std::cerr << "Usage: " << argv[0] << " <filename> <nnz> <dim1> <dim2> <dim3> [dim4] [dim5] <type>\n";
        return 1;
    }

    std::string filename = std::string(argv[1]);
    int nnz = std::stoi(argv[2]);
    int rank = argc - 4; // argv[0]=binary, argv[1]=file, argv[2]=nnz, argv[argc-1]=type => argc - 4 = rank
    std::string type = std::string(argv[argc - 1]);
    
    std::vector<int> dimensions;
    for(int i = 3; i < argc - 1; i++){
        dimensions.push_back(std::stoi(argv[i]));
    }

    if (type == "int") {
        generate_checksums<int>(filename, rank, nnz, dimensions);
    } else if (type == "float") {
        generate_checksums<float>(filename, rank, nnz, dimensions);
    } else if (type == "long int") {
        generate_checksums<long long>(filename, rank, nnz, dimensions);
    } else if (type == "double") {
        generate_checksums<double>(filename, rank, nnz, dimensions);
    } else {
        std::cerr << "Unsupported type. The supported types are int, float, long int, and double\n";
        return 1;
    }

    return 0;
}
