#include "../../utility/utils.h"
#include "../../tensor_implementations/alto_impl.h"   

//Generates a random ALTO tensor based on your parameters and tests encoding
template<typename T, typename S>
void test_alto_tensor(int nnz, int rank, std::vector<int> dims)
{
    std::cout << "Testing ALTO tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";


    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = 0.05 * min_dim;
    int max_blocks = (nnz + block_size - 1) / block_size;
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,block_size,max_blocks);

    Alto_Tensor<T,S> alto(test_vec,dims);

    int bits_printed = 0;
    for(int i = 0; i < rank; i++) bits_printed += ceiling_log2(dims[i]);

    std::cout << "ALTO bitmasks:\n";
    const std::vector<S> masks = alto.get_modemasks();
    for (int i = 0; i < masks.size(); i++) {
        print_lsb_bits(masks[i],bits_printed);
    }
    std::cout << "\n";

    const std::vector<ALTOEntry<T,S>> alto_indexes = alto.get_alto();

    int not_found = 0;
    std::vector<std::vector<int>> visited;
    for(int i = 0; i < alto_indexes.size(); i++){
        std::vector<int> decoded_dims;
        for(int j = 0; j < rank; j++){
            decoded_dims.push_back(alto.get_mode_idx_alto(alto_indexes[i].linear_index,j + 1));
        }
        T val = alto_indexes[i].value;
        auto it = std::find(visited.begin(), visited.end(), decoded_dims);
        if(!find_entry(test_vec, decoded_dims, val) || it != visited.end()) not_found++;
    }
    
    if(not_found == 0){
        std::cout << "Tests Passed\n";
    }
    else{
        std::cout << "Tests Failed, " << not_found << " entries out of " << nnz << " where not found\n";
    }

    std::cout<<"\n";
}

void run_multiple_tests()
{
    std::vector<int> dims_3_s = {100,100,100};
    std::vector<int> dims_3_l = {10000000,10000000,10000000};
    std::vector<int> dims_4_s = {100,100,100,100};
    std::vector<int> dims_4_l = {65536,65536,65536,65536};
    std::vector<int> dims_5_s = {100,100,100,100,100};
    std::vector<int> dims_5_l = {8192,8192,8192,8192,8192};
    std::vector<int> dims_6_s = {100,100,100,100,100,100};
    std::vector<int> dims_6_l = {2048,2048,2048,2048,2048,2048};
    std::vector<int> dims_7_s = {100,100,100,100,100,100,100};
    std::vector<int> dims_7_l = {512,512,512,512,512,512,512};

    test_alto_tensor<int,uint64_t>(100, 3, dims_3_s);
    test_alto_tensor<int,__uint128_t>(100, 3, dims_3_l);
    test_alto_tensor<int,uint64_t>(100, 4, dims_4_s);
    test_alto_tensor<int,__uint128_t>(100, 4, dims_4_l);
    test_alto_tensor<int,uint64_t>(100, 5, dims_5_s);
    test_alto_tensor<int,__uint128_t>(100, 5, dims_5_l);
    test_alto_tensor<int,uint64_t>(100, 6, dims_6_s);
    test_alto_tensor<int,__uint128_t>(100, 6, dims_6_l);
    test_alto_tensor<int,uint64_t>(100, 7, dims_7_s);
    test_alto_tensor<int,__uint128_t>(100, 7, dims_7_l);

    test_alto_tensor<float,uint64_t>(100, 3, dims_3_s);
    test_alto_tensor<float,__uint128_t>(100, 3, dims_3_l);
    test_alto_tensor<float,uint64_t>(100, 4, dims_4_s);
    test_alto_tensor<float,__uint128_t>(100, 4, dims_4_l);
    test_alto_tensor<float,uint64_t>(100, 5, dims_5_s);
    test_alto_tensor<float,__uint128_t>(100, 5, dims_5_l);
    test_alto_tensor<float,uint64_t>(100, 6, dims_6_s);
    test_alto_tensor<float,__uint128_t>(100, 6, dims_6_l);
    test_alto_tensor<float,uint64_t>(100, 7, dims_7_s);
    test_alto_tensor<float,__uint128_t>(100, 7, dims_7_l);

    test_alto_tensor<long int,uint64_t>(100, 3, dims_3_s);
    test_alto_tensor<long int,__uint128_t>(100, 3, dims_3_l);
    test_alto_tensor<long int,uint64_t>(100, 4, dims_4_s);
    test_alto_tensor<long int,__uint128_t>(100, 4, dims_4_l);
    test_alto_tensor<long int,uint64_t>(100, 5, dims_5_s);
    test_alto_tensor<long int,__uint128_t>(100, 5, dims_5_l);
    test_alto_tensor<long int,uint64_t>(100, 6, dims_6_s);
    test_alto_tensor<long int,__uint128_t>(100, 6, dims_6_l);
    test_alto_tensor<long int,uint64_t>(100, 7, dims_7_s);
    test_alto_tensor<long int,__uint128_t>(100, 7, dims_7_l);

    test_alto_tensor<double,uint64_t>(100, 3, dims_3_s);
    test_alto_tensor<double,__uint128_t>(100, 3, dims_3_l);
    test_alto_tensor<double,uint64_t>(100, 4, dims_4_s);
    test_alto_tensor<double,__uint128_t>(100, 4, dims_4_l);
    test_alto_tensor<double,uint64_t>(100, 5, dims_5_s);
    test_alto_tensor<double,__uint128_t>(100, 5, dims_5_l);
    test_alto_tensor<double,uint64_t>(100, 6, dims_6_s);
    test_alto_tensor<double,__uint128_t>(100, 6, dims_6_l);
    test_alto_tensor<double,uint64_t>(100, 7, dims_7_s);
    test_alto_tensor<double,__uint128_t>(100, 7, dims_7_l);
}

int main(int argc, char* argv[]) {
    if ((argc < 4 || argc > 10) && argc != 1) {
        std::cerr << "Usage: " << argv[0] 
                  << "<nnz> (up to seven different dimensions) <Type> or no arguments for comprehensive testing\n";
        return 1;
    }
    else if(argc != 1){
        int nnz = std::stoi(argv[1]);
        int rank = argc - 3;
        std::string type = std::string(argv[argc - 1]);
        std::vector<int> dimensions;
        for(int i = 2; i < argc - 1; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_alto_tensor<int,uint64_t>(nnz, rank, dimensions);
            else if(type == "float") test_alto_tensor<float,uint64_t>(nnz, rank, dimensions);
            else if(type == "long int") test_alto_tensor<long int,uint64_t>(nnz, rank, dimensions);
            else if(type == "double") test_alto_tensor<double,uint64_t>(nnz, rank, dimensions);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, long int and double\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_alto_tensor<int,__uint128_t>(nnz, rank, dimensions);
            else if(type == "float") test_alto_tensor<float,__uint128_t>(nnz, rank, dimensions);
            else if(type == "long int") test_alto_tensor<long int,__uint128_t>(nnz, rank, dimensions);
            else if(type == "double") test_alto_tensor<double,__uint128_t>(nnz, rank, dimensions);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, long int and double\n";
                return 1;
            }
        }
    }
    else{
        run_multiple_tests();
    }

    return 0;
}

