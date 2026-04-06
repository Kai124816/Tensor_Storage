#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include "../../utility/utils.h"
#include "../../tensor_implementations/tensor_impl.h"
#include "../../tensor_implementations/alto_impl.h"
#include "../../tensor_implementations/blco_impl.h"      

//Tests ALTO/BLCO encoding using a file
template<typename T, typename S>
bool test_tensor_file(int alto_or_blco, std::string filename, int nnz, int rank, std::vector<int> dims)
{
    std::cout << "Testing " << (alto_or_blco == 0 ? "ALTO" : "BLCO") << " tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";
    int default_decomp_rank = 10;

    std::vector<std::vector<T>> default_fmats;
    default_fmats.resize(rank);
    for(int i = 0; i < rank; i++) default_fmats[i] = generate_random_array_seed(
    dims[i] * default_decomp_rank, static_cast<T>(0), static_cast<T>(3), SEEDS[i]);

    std::cout << "--- Testing File Construction ---\n";
    std::vector<NNZ_Entry<T>> file_generated_vec = read_tensor_file_binary<T>(filename, rank, nnz);
    
    // Sort file_test_vec exactly in the lexicographical coordinate order
    std::sort(file_generated_vec.begin(), file_generated_vec.end(), [](const NNZ_Entry<T>& a, const NNZ_Entry<T>& b) {
        for(size_t i = 0; i < a.coords.size(); ++i) {
            if (a.coords[i] != b.coords[i]) return a.coords[i] < b.coords[i];
        }
        return false;
    });
    
    Alto_Tensor<T,S> alto;
    Blco_Tensor<T,S> blco;
    std::vector<NNZ_Entry<T>> object_generated_vec;
    if(alto_or_blco == 0){
        alto = Alto_Tensor<T,S>(filename, nnz, dims, default_fmats, default_decomp_rank);
        object_generated_vec = alto.create_entry_vec();
    }
    else{
        blco = Blco_Tensor<T,S>(filename, nnz, dims, default_fmats, default_decomp_rank);
        object_generated_vec = blco.create_entry_vec();
    }   

    // Sort file_generated_vec exactly in the same lexicographical order
    std::sort(object_generated_vec.begin(), object_generated_vec.end(), [](const NNZ_Entry<T>& a, const NNZ_Entry<T>& b) {
        for(size_t i = 0; i < a.coords.size(); ++i) {
            if (a.coords[i] != b.coords[i]) return a.coords[i] < b.coords[i];
        }
        return false;
    });

    bool file_match = true;
    if (file_generated_vec.size() != object_generated_vec.size()) {
        file_match = false;
    } else {
        for (size_t i = 0; i < file_generated_vec.size(); i++) {
            if (file_generated_vec[i].coords != object_generated_vec[i].coords 
                || file_generated_vec[i].value != object_generated_vec[i].value) {
                file_match = false;
                break;
            }
        }
    }

    if(!file_match) std::cout << "File Test Failed! entry vectors don't match\n";
    else std::cout << "File Test Passed! Entry vectors match\n";

    return file_match;
}

//Tests ALTO/BLCO encoding using a synthetically generated tensor
template<typename T, typename S>
bool test_tensor_synthetic(int alto_or_blco, int nnz, int rank, std::vector<int> dims)
{
    std::cout << "Testing " << (alto_or_blco == 0 ? "ALTO" : "BLCO") << " tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";
    int default_decomp_rank = 10;

    std::cout << "\n--- Testing Random Vector Construction ---\n";
    std::vector<NNZ_Entry<T>> rand_test_vec = generate_block_sparse_tensor_nd<T>(dims, nnz, static_cast<T>(1), static_cast<T>(10), 5, 1000, 0.5f);
    
    // Sort rand_test_vec
    std::sort(rand_test_vec.begin(), rand_test_vec.end(), [](const NNZ_Entry<T>& a, const NNZ_Entry<T>& b) {
        for(size_t i = 0; i < a.coords.size(); ++i) {
            if (a.coords[i] != b.coords[i]) return a.coords[i] < b.coords[i];
        }
        return false;
    });

    Alto_Tensor<T,S> alto;
    Blco_Tensor<T,S> blco;
    std::vector<NNZ_Entry<T>> rand_generated_vec;
    if(alto_or_blco == 0){
        alto = Alto_Tensor<T,S>(rand_test_vec, dims, default_decomp_rank);
        rand_generated_vec = alto.create_entry_vec();
    }
    else{
        blco = Blco_Tensor<T,S>(rand_test_vec, dims, default_decomp_rank);
        rand_generated_vec = blco.create_entry_vec();
    }   

    // Sort rand_generated_vec
    std::sort(rand_generated_vec.begin(), rand_generated_vec.end(), [](const NNZ_Entry<T>& a, const NNZ_Entry<T>& b) {
        for(size_t i = 0; i < a.coords.size(); ++i) {
            if (a.coords[i] != b.coords[i]) return a.coords[i] < b.coords[i];
        }
        return false;
    });

    bool rand_match = true;
    if (rand_generated_vec.size() != rand_test_vec.size()) {
        rand_match = false;
    } else {
        for (size_t i = 0; i < rand_generated_vec.size(); i++) {
            if (rand_generated_vec[i].coords != rand_test_vec[i].coords || rand_generated_vec[i].value != rand_test_vec[i].value) {
                rand_match = false;
                break;
            }
        }
    }

    if(!rand_match) std::cout << "Random Vector Test Failed! entry vectors don't match\n";
    else std::cout << "Random Vector Test Passed! Entry vectors match\n";

    return rand_match;
}