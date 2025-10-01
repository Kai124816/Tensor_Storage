#include "../utility/utils.h"   
#include "../ALTO/alto_impl.h"   
#include "../BLCO/blco_impl.h"
#include "../BLCO/gpu_functions.h"


template<typename T, typename S>
void mttkrp_benchmark(const std::string &filename, int nnz, int rows, int cols, int depth, int mode)
{
    std::cout << "Testing mode "<< mode <<" MTTKRP using BLCO Tensor\n";
    std::cout << "Tensor info: "<< rows << " x " << cols << 
    " x " << depth << ", " << nnz << " non zero entries\n";

    int dims[3] = {rows, cols, depth};

    std::vector<NNZ_Entry<T>> test_vec = read_tensor_file<T>(filename,nnz);

    if(test_vec.empty()) return;

    auto construction_start = std::chrono::high_resolution_clock::now();
    BLCO_Tensor_3D<T,S> blco(test_vec, rows, cols, depth);
    auto construction_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(construction_end - construction_start).count();
    std::cout<<"Constructing the tensor took "<< duration << " ms\n";

    std::vector<T**> fmats = blco.get_fmats();
    T** input_matrix = fmats[mode - 1];
    T** copy_input_matrix = create_and_copy_matrix(input_matrix,dims[mode - 1],blco.get_rank());

    auto start = std::chrono::high_resolution_clock::now();
    T** output_matrix = MTTKRP_BLCO(mode,blco);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for MTTKRP: " << duration << " ms\n\n";

    int nt_modes1[3] = {2, 1, 1};
    int nt_modes2[3] = {3, 3, 2};

    T** test_matrix = MTTKRP(mode,copy_input_matrix,fmats[nt_modes1[mode-1] - 1],fmats[nt_modes2[mode-1] - 1],blco.get_rank(),test_vec);

    bool test_passed;
    if constexpr (std::is_floating_point<T>::value){
        test_passed = compare_matricies_float(output_matrix,test_matrix,dims[mode-1],blco.get_rank());
    }
    else test_passed = compare_matricies<T>(output_matrix,test_matrix,dims[mode-1],blco.get_rank());

    if(!test_passed){
        std::cout<<"Test Failed creating mismatch log!\n";
    
        std::ofstream logfile("mismatch_log.txt");
        if constexpr (std::is_floating_point<T>::value){
            compare_matricies_id_float(output_matrix,test_matrix,dims[mode-1],blco.get_rank(),logfile);
        }
        else compare_matricies_id<T>(output_matrix,test_matrix,dims[mode-1],blco.get_rank(),logfile);

        std::cout<<"created mismatch log\n";
    }
    
    for(int i = 0; i < dims[mode-1]; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] 
                  << " <filename> <nnz> <rows> <cols> <depth> <mode> <Type>\n";
        return 1;
    }
    else{
        std::string filename = argv[1];
        int nnz   = std::stoi(argv[2]);
        int rows  = std::stoi(argv[3]);
        int cols  = std::stoi(argv[4]);
        int depth = std::stoi(argv[5]);
        int mode  = std::stoi(argv[6]);

        if(ceiling_log2(rows) + ceiling_log2(cols) + ceiling_log2(depth) < 64){
            if(std::string(argv[7]) == "int") mttkrp_benchmark<int,uint64_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "float") mttkrp_benchmark<float,uint64_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "unsigned long long int") mttkrp_benchmark<unsigned long long int,uint64_t>(filename, nnz, rows, cols, depth, mode);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long long int\n";
                return 1;
            }
        }
        else{
            if(std::string(argv[7]) == "int") mttkrp_benchmark<int,__uint128_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "float") mttkrp_benchmark<float,__uint128_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "unsigned long long int") mttkrp_benchmark<unsigned long long int,__uint128_t>(filename, nnz, rows, cols, depth, mode);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long long int\n";
                return 1;
            }
        }
    }

    return 0;
}
