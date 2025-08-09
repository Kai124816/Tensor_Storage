#include "utils.h"
#include "tensor_impl.h"   
#include "alto_impl.h"   
#include "blco_impl.h"


void test_large_blco_tensor()
{
    std::cout << "Testing large BLCO tensor\n";
    std::cout << "\n";

    const int R = 40000000, C = 85000000, D = 10000000;

    float freq = 0.000000000000000000003f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    std::cout<<test_vec.size()<<"\n";

    std::cout<<"Entries:"<<"\n";
    print_entry_vec(test_vec);
    std::cout<<"\n";

    BLCO_Tensor_3D<int,__uint128_t> blco(test_vec, R, C, D);

    int bits_printed = 64;

    std::cout << "BLCO bitmasks:\n";
    for (const auto& a : blco.get_modemasks()) {
        print_lsb_bits(a,bits_printed);
    }
    std::cout << "\n";

    const std::vector<BLCO_BLOCK_CPU<int>> blco_indexes = blco.get_blco();

    std::cout << "BLCO entries:\n";
    for (int i = 0; i < blco_indexes.size(); i++) {
        int block = blco_indexes[i].block;
        for(int j = 0; j < blco_indexes[i].indexes.size(); j++){
            std::cout << "Block: " << block << "\n" ;
            std::cout << "Linear Index: ";  
            print_uint64(blco_indexes[i].indexes[j],bits_printed);
            std::cout << "Value: " << blco_indexes[i].values[j] << "\n";
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    for (int i = 0; i < blco_indexes.size(); i++) {
        int block = blco_indexes[i].block;
        for(int j = 0; j < blco_indexes[i].indexes.size(); j++){
            std::cout << "Block: " << block << "\n" ;
            std::cout << "Linear Index: ";  
            print_uint64(blco_indexes[i].indexes[j],bits_printed);
            std::cout << "Value: " << blco_indexes[i].values[j] << "\n";
            std::cout << "\n";
        }
    }

    for (int i = 0; i < blco_indexes.size(); i++) {
        int block = blco_indexes[i].block;
        for(int j = 0; j < blco_indexes[i].indexes.size(); j++){
            int row_ind = blco.get_mode_idx_blco(blco_indexes[i].indexes[j],block,1);
            int col_ind = blco.get_mode_idx_blco(blco_indexes[i].indexes[j],block,2);
            int depth_ind = blco.get_mode_idx_blco(blco_indexes[i].indexes[j],block,3);
            int val = blco_indexes[i].values[j];
            find_entry(test_vec, row_ind, col_ind, depth_ind, val);
        }
    }
    std::cout << "\n";

    std::cout << "Tests finished!\n";
}

int main() {
    test_large_blco_tensor();
    return 0;
};