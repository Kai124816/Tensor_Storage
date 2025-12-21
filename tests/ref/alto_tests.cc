#include "../utility/utils.h"
#include "../ALTO/alto_impl.h"      

//Generates a random BLCO tensor based on your parameters and tests encoding 
void test_alto_tensor(int nnz, int rows, int cols, int depth)
{
    std::cout << "Testing ALTO tensor\n";
    std::cout << "Tensor info: "<< rows << " x " << cols << 
    " x " << depth << ", " << nnz << " non zero entries\n";
    std::cout << "\n";
    
    double total_entries = static_cast<double>(rows) * cols * depth;
    double freq = static_cast<double>(nnz) / total_entries;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(rows,cols,depth,freq,0,100);

    Alto_Tensor_3D<int,__uint128_t> alto(test_vec, rows, cols, depth);

    int bits_printed = ceiling_log2(rows) + ceiling_log2(cols) + ceiling_log2(depth);

    std::cout << "ALTO bitmasks:\n";
    for (const auto& a : alto.get_modemasks()) {
        print_lsb_bits(a,bits_printed);
    }
    std::cout << "\n";

    const std::vector<ALTOEntry<int,__uint128_t>> alto_indexes = alto.get_alto();

    int not_found = 0;
    for(int i = 0; i < alto_indexes.size(); i++){
        int row_ind = alto.get_mode_idx(alto_indexes[i].linear_index,1);
        int col_ind = alto.get_mode_idx(alto_indexes[i].linear_index,2);
        int depth_ind = alto.get_mode_idx(alto_indexes[i].linear_index,3);
        int val = alto_indexes[i].value;
        if(!find_entry(test_vec, row_ind, col_ind, depth_ind, val)) not_found++;
    }
    
    if(not_found == 0){
        std::cout << "Tests Passed\n";
    }
    else{
        std::cout << "Tests Failed, " << not_found << " entries out of " << nnz << " where not found\n";
    }
}

//Tests and times MTTKRP using BLCO tensor 
template<typename T, typename S>
void mttkrp_on_tensor(const std::string &filename, int nnz, int rows, int cols, int depth, int mode)
{
    std::cout << "Testing MTTKRP using ALTO Tensor\n";
    std::cout << "Tensor info: "<< rows << " x " << cols << 
    " x " << depth << ", " << nnz << " non zero entries\n";
    std::cout << "\n";

    int dims[3] = {rows, cols, depth};

    std::vector<NNZ_Entry<T>> test_vec = read_tensor_file_binary<T>(filename);

    if(test_vec.empty()) return;

    std::cout << "Constructing Tensor:\n";
    std::cout<<"\n";
    Alto_Tensor_3D<T,S> alto(test_vec, rows, cols, depth);

    std::vector<T**> fmats = alto.get_fmats();
    T** input_matrix = fmats[mode - 1];
    T** copy_input_matrix = create_and_copy_matrix(input_matrix,dims[mode - 1],alto.get_rank());

    std::cout<<"Conducting MTTKRP:"<<"\n\n";

    auto start = std::chrono::high_resolution_clock::now();
    T** output_matrix = alto.MTTKRP_Alto_Parallel(mode);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for MTTKRP: " << duration << " ms\n\n";

    int nt_modes1[3] = {2, 1, 1};
    int nt_modes2[3] = {3, 3, 2};

    T** test_matrix = MTTKRP(mode,copy_input_matrix,fmats[nt_modes1[mode-1] - 1],fmats[nt_modes2[mode-1] - 1],alto.get_rank(),test_vec);

    bool test_passed;
    if constexpr (std::is_floating_point<T>::value){
        test_passed = compare_matricies_float(output_matrix,test_matrix,dims[mode-1],alto.get_rank());
    }
    else test_passed = compare_matricies<T>(output_matrix,test_matrix,dims[mode-1],alto.get_rank());

    std::cout<<"comparing matrices\n\n";
    if(test_passed) std::cout<<"Test Passed!"<<"\n";
    else{
        std::cout<<"Test Failed! Check mismatch_log.txt for more info"<<"\n";
        std::ofstream logfile("mismatch_log.txt");
        if constexpr (std::is_floating_point<T>::value){
            compare_matricies_id_float(output_matrix,test_matrix,dims[mode-1],alto.get_rank(),logfile);
        }
        else compare_matricies_id<T>(output_matrix,test_matrix,dims[mode-1],alto.get_rank(),logfile);
    }
    
    for(int i = 0; i < dims[mode-1]; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}


int main(int argc, char* argv[]) {
    if (argc != 8 && argc != 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <filename> <nnz> <rows> <cols> <depth> <mode> <Type>\n"
                  << "or <nnz> <rows> <cols> <depth>\n";
        return 1;
    }
    else if(argc == 8){
        std::string filename = argv[1];
        int nnz   = std::stoi(argv[2]);
        int rows  = std::stoi(argv[3]);
        int cols  = std::stoi(argv[4]);
        int depth = std::stoi(argv[5]);
        int mode  = std::stoi(argv[6]);

        if(ceiling_log2(rows) + ceiling_log2(cols) + ceiling_log2(depth) < 64){
            if(std::string(argv[7]) == "int") mttkrp_on_tensor<int,uint64_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "float") mttkrp_on_tensor<float,uint64_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "unsigned long long int") mttkrp_on_tensor<unsigned long long int,uint64_t>(filename, nnz, rows, cols, depth, mode);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long long int\n";
                return 1;
            }
        }
        else{
            if(std::string(argv[7]) == "int") mttkrp_on_tensor<int,__uint128_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "float") mttkrp_on_tensor<float,__uint128_t>(filename, nnz, rows, cols, depth, mode);
            else if(std::string(argv[7]) == "unsigned long long int") mttkrp_on_tensor<unsigned long long int,__uint128_t>(filename, nnz, rows, cols, depth, mode);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long long int\n";
                return 1;
            }
        }
    }
    else{
        int nnz = std::stoi(argv[1]);
        int rows = std::stoi(argv[2]);
        int cols = std::stoi(argv[3]);
        int depth = std::stoi(argv[4]);

        test_alto_tensor(nnz, rows, cols, depth);
    }

    return 0;
}