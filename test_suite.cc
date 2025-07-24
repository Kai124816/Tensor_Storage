#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <unordered_set>
#include <tuple>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include "alto.h"  
#include "blco.h"

void test_large_alto_tensor()
{
    std::cout << "Testing large ALTO tensor\n";
    std::cout << "\n";

    const int R = 4000000, C = 8500000, D = 10;

    float freq = 0.00000000000007f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    std::cout<<"Entries:"<<"\n";
    print_entry_vec(test_vec);
    std::cout<<"\n";

    Alto_Tensor_3D<int,__uint128_t> alto(test_vec, R, C, D);

    int bits_printed = 50;

    std::cout << "ALTO bitmasks:\n";
    for (const auto& a : alto.get_modemasks()) {
        print_lsb_bits(a,bits_printed);
    }
    std::cout << "\n";

    std::vector<ALTOEntry<int,__uint128_t>> tensor = alto.get_alto();

    std::cout << "ALTO entries:\n";
    for (const auto& e : tensor) {
        std::cout << "Linear Index: ";
        print_lsb_bits(e.linear_index, bits_printed);
        std::cout << "Value: " << e.value << std::endl;
    }
    std::cout << "\n";

    for(int i = 0; i<tensor.size(); i++){
        int row_ind = alto.get_mode_idx(tensor[i].linear_index,1);
        int col_ind = alto.get_mode_idx(tensor[i].linear_index,2);
        int depth_ind = alto.get_mode_idx(tensor[i].linear_index,3);
        int val = tensor[i].value;
        find_entry(test_vec, row_ind, col_ind, depth_ind, val);
    }
    std::cout << "\n";

    std::cout << "Tests finished!\n";
}


void test_large_blco_tensor()
{
    std::cout << "Testing large BLCO tensor\n";
    std::cout << "\n";

    const int R = 40000000, C = 85000000, D = 10000000;

    float freq = 0.000000000000000000003f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    std::cout<<"Entries"<<"\n";
    print_entry_vec(test_vec);
    std::cout<<"\n";

    BLCO_Tensor_3D<int,__uint128_t> blco(test_vec, R, C, D);

    int bits_printed = 64;

    std::cout << "BLCO bitmasks:\n";
    for (const auto& a : blco.get_modemasks()) {
        print_lsb_bits(a,bits_printed);
    }
    std::cout << "\n";

    const std::pair<std::vector<uint64_t>, std::vector<int>> blco_indexes = blco.get_blco();

    std::cout << "BLCO entries:\n";
    for (int i = 0; i < blco_indexes.first.size(); i++) {
        std::cout << "Block: " << blco.find_block(i)<< "\n" ;
        std::cout << "Linear Index: ";  
        print_uint64(blco_indexes.first[i],bits_printed);
        std::cout << "Value: " << blco_indexes.second[i] << "\n";
        std::cout << "\n";
    }
    std::cout << "\n";

    for(int i = 0; i<test_vec.size(); i++){
        int row_ind = blco.get_mode_idx_blco(blco_indexes.first[i],i,1);
        int col_ind = blco.get_mode_idx_blco(blco_indexes.first[i],i,2);
        int depth_ind = blco.get_mode_idx_blco(blco_indexes.first[i],i,3);
        int val = blco_indexes.second[i];
        find_entry(test_vec, row_ind, col_ind, depth_ind, val);
    }

    std::cout << "\n";

    std::cout << "Tests finished!\n";
}


int main() {
    test_large_blco_tensor();
    return 0;
}
