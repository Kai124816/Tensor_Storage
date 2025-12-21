#include "../../utility/utils.h"
#include "../../tensor_implementations/blco_impl.h"   
#include "../../gpu_code/3D_kernels.h"
#include "../../gpu_code/4D_kernels.h"
#include "../../gpu_code/5D_kernels.h"

//Tests and times MTTKRP using BLCO tensor 
template<typename T, typename S>
void mttkrp_on_tensor(std::string filename, std::vector<int> dims, int nnz, int mode, std::string diff_file = "diff.txt")
{
    std::cout << "Testing mode " << mode << " MTTKRP using BLCO Tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << dims.size() << 
    "\n" << "Non Zero Entries " << nnz << "\n" << 
    "dimensions \n";

    int rank = dims.size();

    for(int i = 0; i < rank - 1; i++){
        std::cout << dims[i] << " x ";
    }
    std::cout << dims[rank - 1] << "\n\n";
    
    std::vector<NNZ_Entry<T>> test_vec;
    if(filename == "-none"){
        int min_dim = *(std::min_element(dims.begin(), dims.end()));
        int block_size = 0.05 * min_dim;
        int max_blocks = (nnz + block_size - 1) / block_size;
        test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,block_size,max_blocks);
    }
    else test_vec = read_tensor_file_binary<T>(filename, dims);

    Blco_Tensor<T,S> blco(test_vec,dims);

    std::vector<T> output_matrix;
    if(rank == 3) output_matrix = MTTKRP_BLCO_3D(mode,blco); //Matrix is flattened
    else if(rank == 4) output_matrix = MTTKRP_BLCO_4D(mode,blco);
    else if(rank == 5) output_matrix = MTTKRP_BLCO_5D(mode,blco);
    else{
        std::cerr << "invalid rank\n";
        return;
    }

    std::vector<T*> fmats = blco.get_fmats();
    int decomp_rank = blco.get_factor_rank();
    T* input_matrix = create_and_copy_array(fmats[mode - 1], dims[mode - 1] * decomp_rank);
    T* test_matrix = MTTKRP_Naive(mode, input_matrix, fmats, decomp_rank, test_vec);

    bool test_passed;
    if constexpr (std::is_floating_point<T>::value){
        double diff = compare_arrays_float<T>(output_matrix.data(), test_matrix, dims[mode-1] * decomp_rank);
        if(diff > 0.00001) test_passed = false;
        else test_passed = true;
        std::cout << "Average difference of: " << diff << "\n";
    }
    else test_passed = compare_arrays<T>(output_matrix.data(), test_matrix, dims[mode-1] * decomp_rank);

    if(test_passed) std::cout << "Tests Passed!\n";
    else{
        std::cout << "Tests Failed!\n";
        int mat_size = dims[mode - 1] * decomp_rank;
        if(mat_size < 100000){
            std::cout << "Outputing Matricies to File " << diff_file << "\n";

            std::ofstream outfile(diff_file, std::ios::app);
            print_matrix_to_file(output_matrix.data(), dims[mode - 1], decomp_rank, diff_file, "GPU Matrix");
            print_matrix_to_file(test_matrix, dims[mode - 1], decomp_rank, diff_file, "CPU Matrix");
            print_entry_vec(test_vec);
        }
        else if(mat_size > 100000){
            std::cout << "Outputing Matricies to File " << diff_file << "\n";

            std::ofstream outfile(diff_file, std::ios::app);
            print_differences_to_file(output_matrix.data(), test_matrix, dims[mode - 1], decomp_rank, diff_file, "GPU Matrix", "CPU Matrix");
        }
    }
    
    delete[] test_matrix;
    
    std::cout<<"\n";
}

void run_multiple_tests(int dims)
{
    std::vector<int> dims_3_s = {100,100,100};
    std::vector<int> dims_3_l = {10000000,10000000,10000000};
    std::vector<int> dims_4_s = {10,10,100,100};
    std::vector<int> dims_4_l = {65536,65536,65536,65536};
    std::vector<int> dims_5_s = {10,10,10,10,100};
    std::vector<int> dims_5_l = {8192,8192,8192,8192,8192};
    std::vector<std::vector<int>> inputs = {dims_3_s, dims_3_l, dims_4_s, dims_4_l, dims_5_s, dims_5_l};

    if(dims >= 3 && dims <= 5){
        std::string file = "-none"; //Placeholder
        std::vector<std::string> datatypes = {"int", "float", "long long", "double"};
        int offset = 2 * (dims - 3); //Offset into inputs array
        std::string datatype;
        int mode;

        for(int j = 0; j < dims; ++j){
            mode = j + 1;
            for(int i = 0; i < datatypes.size(); ++i){
                datatype = datatypes[i];
                if(datatype == "int"){
                    std::cout << "----------Testing integer values----------\n\n";
                    mttkrp_on_tensor<int,uint64_t>(file, inputs[offset], 100, mode);
                    mttkrp_on_tensor<int,__uint128_t>(file, inputs[offset + 1], 100, mode);
                }
                else if(datatype == "float"){
                    std::cout << "----------Testing single precision floating point values----------\n\n";
                    mttkrp_on_tensor<float,uint64_t>(file, inputs[offset], 100, mode);
                    mttkrp_on_tensor<float,__uint128_t>(file, inputs[offset + 1], 100, mode);
                }
                else if(datatype == "long long"){
                    std::cout << "----------Testing long long values----------\n\n";
                    mttkrp_on_tensor<unsigned long long,uint64_t>(file, inputs[offset], 100, mode);
                    mttkrp_on_tensor<unsigned long long,__uint128_t>(file, inputs[offset + 1], 100, mode);
                }
                else if(datatype == "double"){
                    std::cout << "----------Testing double precision floating point values----------\n\n";
                    mttkrp_on_tensor<double,uint64_t>(file, inputs[offset], 100, mode);
                    mttkrp_on_tensor<double,__uint128_t>(file, inputs[offset + 1], 100, mode);
                }
            }
        }
    }
    else{
        std::cerr << "invalid dimensions\n";
    }  
}


int main(int argc, char* argv[]) 
{
    if ((argc > 10) || (argc < 8 && argc != 2)) {
        std::cerr << "Usage: " << argv[0] 
                  << " <filename> <nnz> (three to five different dimensions) <mode> <Type>\n"
                  << "if you want to test it on a synthetically generated tensor pass in -none as the file name\n"
                  << "if you want to run comprehensive tests with synthetic tensors pass in 3, 4, or 5 as arguments\n"
                  << "for the dimensions you want to test\n";
        return 1;
    }
    else if(argc == 2){
        int num_dims = std::stoi(argv[1]);
        run_multiple_tests(num_dims);
    }
    else{
        std::string filename = argv[1];
        int non_zeros = std::stoi(argv[2]);
        int rank = argc - 5;
        std::vector<int> dimensions;
        for(int i = 3; i < argc - 2; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }
        int mode = std::stoi(argv[argc - 2]);
        std::string type = std::string(argv[argc - 1]);

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed < 64){
            if(std::string(argv[argc - 1]) == "int") mttkrp_on_tensor<int,uint64_t>(filename, dimensions, non_zeros, mode);
            else if(std::string(argv[argc - 1]) == "float") mttkrp_on_tensor<float,uint64_t>(filename, dimensions, non_zeros, mode);
            else if(std::string(argv[argc - 1]) == "long long") mttkrp_on_tensor<unsigned long long,uint64_t>(filename, dimensions, non_zeros, mode);
            else if(std::string(argv[argc - 1]) == "double") mttkrp_on_tensor<double,uint64_t>(filename, dimensions, non_zeros, mode);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, unsigned long long, and double\n";
                return 1;
            }
        }
        else{
            if(std::string(argv[argc - 1]) == "int") mttkrp_on_tensor<int,__uint128_t>(filename, dimensions, non_zeros, mode);
            else if(std::string(argv[argc - 1]) == "float") mttkrp_on_tensor<float,__uint128_t>(filename, dimensions, non_zeros, mode);
            else if(std::string(argv[argc - 1]) == "long long") mttkrp_on_tensor<unsigned long long,__uint128_t>(filename, dimensions, non_zeros, mode);
            else if(std::string(argv[argc - 1]) == "double") mttkrp_on_tensor<double,__uint128_t>(filename, dimensions, non_zeros, mode);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, unsigned long long, and double\n";
                return 1;
            }
        }
    }

    return 0;
}