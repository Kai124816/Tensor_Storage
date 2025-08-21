#include "utils.h"
#include "tensor_impl.h"   
#include "alto_impl.h"      

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

void alto_mttkrp()
{
    std::cout << "Testing ALTO MTTKRP\n";
    std::cout << "\n";

    const int R = 100, C = 250, D = 75;

    float freq = 0.000052f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    std::cout<<"Entries:"<<"\n";
    print_entry_vec(test_vec);
    std::cout<<"\n";

    Alto_Tensor_3D<int,__uint128_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 3 factor matrix before MTTKRP:"<<"\n";

    int** input_matrix = alto.get_fmats()[2];

    print_matrix(input_matrix, D, alto.get_rank());

    std::cout<<"\n";

    std::cout<<"Mode 3 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto(3);

    print_matrix(output_matrix, D, alto.get_rank());
}

//Testing Paralell alto with less than 4 non-zeros per fiber
void alto_mttkrp_paralell_1()
{
    std::cout << "Testing ALTO MTTKRP Paralell version - less than 4 non-zero entries per fiber\n";
    std::cout << "\n";

    const int R = 100, C = 250, D = 75;

    float freq = 0.000052f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 3 factor matrix before MTTKRP:"<<"\n";

    int** input_matrix = alto.get_fmats()[2];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,D,alto.get_rank());

    print_matrix(input_matrix, D, alto.get_rank());

    std::cout<<"\n";

    std::cout<<"Mode 3 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto_Parallel(3);

    print_matrix(output_matrix, D, alto.get_rank());

    int** test_matrix = MTTKRP(3,copy_input_matrix,alto.get_fmats()[0],alto.get_fmats()[1],alto.get_rank(),test_vec);

    if(compare_matricies(output_matrix,test_matrix,D,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";
    std::cout<<"\n";

    for(int i = 0; i < D; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}


//Testing Paralell alto with more than 4 non-zeros per fiber
void alto_mttkrp_paralell_2()
{
    std::cout << "Testing ALTO MTTKRP Paralell version - less than 4 non-zero entries per fiber\n";
    std::cout << "\n";

    const int R = 100, C = 250, D = 75;

    float freq = 0.000052f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 2 factor matrix before MTTKRP:"<<"\n";

    int** input_matrix = alto.get_fmats()[1];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,C,alto.get_rank());

    print_matrix(input_matrix, C, alto.get_rank());
    std::cout<<"\n";

    std::cout<<"Mode 2 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto_Parallel(2);

    print_matrix(output_matrix, C, alto.get_rank());
    std::cout<<"\n";

    int** test_matrix = MTTKRP(2,copy_input_matrix,alto.get_fmats()[0],alto.get_fmats()[2],alto.get_rank(),test_vec);

    if(compare_matricies(output_matrix,test_matrix,C,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";

    for(int i = 0; i < C; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}

//Testing Paralell alto with more than 4 non-zeros per fiber
void alto_mttkrp_paralell_3()
{
    std::cout << "Testing ALTO MTTKRP Paralell version - less than 4 non-zero entries per fiber\n";
    std::cout << "\n";

    const int R = 100, C = 250, D = 75;

    float freq = 0.000052f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 1 factor matrix before MTTKRP:"<<"\n";

    int** input_matrix = alto.get_fmats()[0];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,R,alto.get_rank());

    print_matrix(input_matrix, R, alto.get_rank());
    std::cout<<"\n";

    std::cout<<"Mode 1 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto_Parallel(1);

    print_matrix(output_matrix, R, alto.get_rank());
    std::cout<<"\n";

    int** test_matrix = MTTKRP(1,copy_input_matrix,alto.get_fmats()[1],alto.get_fmats()[2],alto.get_rank(),test_vec);

    if(compare_matricies(output_matrix,test_matrix,R,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";

    for(int i = 0; i < R; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}

//Testing Paralell alto with more than 4 non-zeros per fiber
void alto_mttkrp_paralell_4()
{
    std::cout << "Testing ALTO MTTKRP Paralell version - more than 4 non-zero entries per fiber\n";
    std::cout << "\n";

    const int R = 100, C = 75, D = 250;

    float freq = 0.024f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 3 factor matrix before MTTKRP:"<<"\n";

    std::vector<int**> fmats = alto.get_fmats();
    int** input_matrix = fmats[2];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,D,alto.get_rank());

    print_matrix(input_matrix, D, alto.get_rank());
    std::cout<<"\n";

    std::cout<<"Mode 3 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto_Parallel(3);

    print_matrix(output_matrix, D, alto.get_rank());
    std::cout<<"\n";

    int** test_matrix = MTTKRP(3,copy_input_matrix,alto.get_fmats()[0],alto.get_fmats()[1],alto.get_rank(),test_vec);

    if(compare_matricies(output_matrix,test_matrix,D,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";

    for(int i = 0; i < D; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}

//Testing Paralell alto with more than 4 non-zeros per fiber
void alto_mttkrp_paralell_5()
{
    std::cout << "Testing ALTO MTTKRP Paralell version - more than 4 non-zero entries per fiber\n";
    std::cout << "\n";

    const int R = 100, C = 250, D = 75;

    float freq = 0.024f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 2 factor matrix before MTTKRP:"<<"\n";

    std::vector<int**> fmats = alto.get_fmats();
    int** input_matrix = fmats[1];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,C,alto.get_rank());

    print_matrix(input_matrix, C, alto.get_rank());
    std::cout<<"\n";

    std::cout<<"Mode 2 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto_Parallel(2);

    print_matrix(output_matrix, C, alto.get_rank());
    std::cout<<"\n";

    int** test_matrix = MTTKRP(2,copy_input_matrix,alto.get_fmats()[0],alto.get_fmats()[2],alto.get_rank(),test_vec);

    if(compare_matricies(output_matrix,test_matrix,C,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";

    for(int i = 0; i < C; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}

//Testing Paralell alto with more than 4 non-zeros per fiber
void alto_mttkrp_paralell_6()
{
    std::cout << "Testing ALTO MTTKRP Paralell version - more than 4 non-zero entries per fiber\n";
    std::cout << "\n";

    const int R = 250, C = 100, D = 75;

    float freq = 0.024f;

    std::vector<NNZ_Entry<int>> test_vec = generate_block_sparse_tensor(R,C,D,freq,0,100);

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::cout<<"Mode 1 factor matrix before MTTKRP:"<<"\n";

    std::vector<int**> fmats = alto.get_fmats();
    int** input_matrix = fmats[0];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,R,alto.get_rank());

    print_matrix(input_matrix, R, alto.get_rank());
    std::cout<<"\n";

    std::cout<<"Mode 1 factor matrix after MTTKRP:"<<"\n";

    int** output_matrix = alto.MTTKRP_Alto_Parallel(1);

    print_matrix(output_matrix, R, alto.get_rank());
    std::cout<<"\n";

    int** test_matrix = MTTKRP(1,copy_input_matrix,alto.get_fmats()[1],alto.get_fmats()[2],alto.get_rank(),test_vec);

    if(compare_matricies(output_matrix,test_matrix,R,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";

    for(int i = 0; i < R; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}

//MTTKRP with a real life tensor from a tns file
void alto_mttkrp_paralell_file_input(const std::string &filename)
{
    std::cout << "Testing ALTO MTTKRP Paralell version - with a Tensor file\n";
    std::cout << "\n";

    const int R = 23344784, C = 23344784, D = 166;

    float freq = 0.024f;

    std::vector<NNZ_Entry<int>> test_vec = read_tensor_file<int>(filename,99546551);

    if(test_vec.empty()) return;

    std::cout<<"constructing tensor\n";

    Alto_Tensor_3D<int,uint64_t> alto(test_vec, R, C, D);

    std::vector<int**> fmats = alto.get_fmats();
    int** input_matrix = fmats[0];
    int** copy_input_matrix = create_and_copy_matrix(input_matrix,R,alto.get_rank());

    std::cout<<"Performing MTTKRP algorithm in parallel\n";
    int** output_matrix = alto.MTTKRP_Alto_Parallel(1);

    int** test_matrix = MTTKRP(1,copy_input_matrix,alto.get_fmats()[1],alto.get_fmats()[2],alto.get_rank(),test_vec);

    std::cout<<"Comparing output to test matrix\n";
    if(compare_matricies(output_matrix,test_matrix,R,alto.get_rank())) std::cout<<"Test Passed!"<<"\n";
    else std::cout<<"Test Failed!"<<"\n";

    for(int i = 0; i < R; i++){
        delete[] copy_input_matrix[i];
    }
    delete[] copy_input_matrix;

    std::cout<<"\n";
}




int main() {
    alto_mttkrp_paralell_file_input("fb-m.tns");
    return 0;
};
