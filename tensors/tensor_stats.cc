#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include "../utility/utils.h"

int main(int argc, char* argv[]) {
    if(argc != 5){
        std::cerr << "Usage: " << argv[0] << " <Type> <rank> <nnz> <filename> \n";
        return 1;
    }
    else if (std::string(argv[1]) == "int") {
        int rank = std::stoi(argv[2]);
        int non_zeros = std::stoi(argv[3]);
        std::string filename = std::string(argv[argc - 1]);
        std::vector<NNZ_Entry<int>> entry_vec = read_tensor_file_binary<int>(filename, rank, non_zeros);
        print_tensor_stats<int>(entry_vec);
        return 0;
    }
    else if (std::string(argv[1]) == "float") {
        int rank = std::stoi(argv[2]);
        int non_zeros = std::stoi(argv[3]);
        std::string filename = std::string(argv[argc - 1]);
        std::vector<NNZ_Entry<float>> entry_vec = read_tensor_file_binary<float>(filename, rank, non_zeros);
        print_tensor_stats<float>(entry_vec);
        return 0;
    }
    else if (std::string(argv[1]) == "double") {
        int rank = std::stoi(argv[2]);
        int non_zeros = std::stoi(argv[3]);
        std::string filename = std::string(argv[argc - 1]);
        std::vector<NNZ_Entry<double>> entry_vec = read_tensor_file_binary<double>(filename, rank, non_zeros);
        print_tensor_stats<double>(entry_vec);
        return 0;
    }
    else if (std::string(argv[1]) == "unsigned long long") {
        int rank = std::stoi(argv[2]);
        int non_zeros = std::stoi(argv[3]);
        std::string filename = std::string(argv[argc - 1]);
        std::vector<NNZ_Entry<unsigned long long>> entry_vec = read_tensor_file_binary<unsigned long long>(filename, rank, non_zeros);
        print_tensor_stats<unsigned long long>(entry_vec);
        return 0;
    }
    else{
        std::cout << "Unsupported type " << std::string(argv[1]) << 
        "supported types are: int, float, double, unsigned long long\n";
        return 1;
    }
}
