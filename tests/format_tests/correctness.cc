#include <iostream>
#include <string>
#include <vector>
#include "../../utility/utils.h"
#include "../../tensor_implementations/tensor_impl.h"
#include "../../tensor_implementations/alto_impl.h"
#include "../../tensor_implementations/blco_impl.h"   
#include "helper_functions.cc"   

int main(int argc, char* argv[]) {
    // Expected arguments: <alto_or_blco> <filename> <nnz> (three to five dimensions) <Type>
    if (argc < 8 || argc > 10) {
        std::cerr << "Usage: " << argv[0] 
                  << " <alto_or_blco> <filename> <nnz> (three to five different dimensions) <Type>\n"
                  << " if you want to use a synthetically generated tensor use 'None' as your filename argument\n";
        return 1;
    }

    int alto_or_blco = std::stoi(argv[1]);
    std::string filename = std::string(argv[2]);
    int nnz = std::stoi(argv[3]);
    int rank = argc - 5;
    std::string type = std::string(argv[argc - 1]);
    
    std::vector<int> dimensions;
    for(int i = 4; i < argc - 1; i++){
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
            test_tensor_synthetic<T, S>(alto_or_blco, nnz, rank, dimensions);
        } else {
            test_tensor_file<T, S>(alto_or_blco, filename, nnz, rank, dimensions);
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