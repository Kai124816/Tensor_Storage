#include "../../utility/utils.h"
#include "../../tensor_implementations/blco_impl.h"   
#include "../../gpu_code/3D_kernels.h"
#include "../../gpu_code/4D_kernels.h"
#include "../../gpu_code/5D_kernels.h"

//Tests and times MTTKRP using BLCO tensor 
template<typename T, typename S>
void mttkrp_on_tensor(std::string filename, std::vector<int> dims, int nnz, int mode)
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
    
    std::vector<NNZ_Entry<T>> test_vec = read_tensor_file_binary<T>(filename, rank, nnz);

    Blco_Tensor<T,S> blco(test_vec,dims);
    bool is_csr = blco.get_total_bits_needed() > 74;

    std::vector<T> output_matrix;
    std::vector<float> times = {0.0f};
    if(rank == 3 && !is_csr) output_matrix = MTTKRP_BLCO_3D(mode, blco, times); //Matrix is flattened
    else if(rank == 4 && !is_csr) output_matrix = MTTKRP_BLCO_4D(mode, blco, times);
    else if(rank == 5 && !is_csr) output_matrix = MTTKRP_BLCO_5D(mode, blco, times);
    else if(rank == 4 && is_csr) output_matrix = MTTKRP_BLCO_CSR_4D(mode, blco, times);
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
    }
    else test_passed = compare_arrays<T>(output_matrix.data(), test_matrix, dims[mode-1] * decomp_rank);
    delete[] test_matrix;

    if(!test_passed){
        std::cout << "error in output, terminating program\n";
    }
}

int main(int argc, char* argv[]) 
{
    if ((argc > 10) || (argc < 8)) {
        std::cerr << "Usage: " << argv[0] 
                  << " <filename> <nnz> (three to five different dimensions) <mode> <Type>\n"
                  << "if you want to test it on a synthetically generated tensor pass in -none as the file name\n";
        return 1;
    }
    else{
        std::string filename = argv[1];
        int mode = std::stoi(argv[2]);
        int non_zeros = std::stoi(argv[3]);
        int rank = argc - 5;
        std::vector<int> dimensions;
        for(int i = 4; i < argc - 1; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }
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