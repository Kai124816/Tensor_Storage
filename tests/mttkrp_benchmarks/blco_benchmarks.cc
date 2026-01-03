#include "../../utility/utils.h"
#include "../../tensor_implementations/blco_impl.h"  
#include "../../gpu_code/3D_kernels.h"
#include "../../gpu_code/4D_kernels.h"
#include "../../gpu_code/5D_kernels.h"

//Generates a random ALTO tensor based on your parameters and tests encoding
template<typename T, typename S>
void test_allocation(std::string filename, int nnz, int rank, std::vector<int> dims, int iterations)
{
    std::cout << "Benchmarking BLCO tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";

    std::vector<NNZ_Entry<T>> test_vec;
    if(filename == "-none"){
        int min_dim = *(std::min_element(dims.begin(), dims.end()));
        int block_size = (0.05 * min_dim) + 1;
        int max_blocks = (nnz + block_size - 1) / block_size;
        test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,block_size,max_blocks);
    }
    else test_vec = read_tensor_file_binary<T>(filename, rank, nnz);

    Blco_Tensor<T,S> blco(test_vec,dims);
    int num_blocks = blco.get_num_blocks();

    std::vector<float> allocation_times;
    std::vector<float> deallocation_times;
    for(int i = 0; i < iterations + 1; i++){
        auto alloc_start = std::chrono::high_resolution_clock::now();

        MTTKRP_Device_Resources<T> resources = allocate_mttkrp_resources(blco);

        auto alloc_stop = std::chrono::high_resolution_clock::now();
        auto alloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_stop - alloc_start);
        float alloc_duration_f = static_cast<float>(alloc_duration.count());

        auto dealloc_start = std::chrono::high_resolution_clock::now();

        deallocate_mttkrp_resources(resources, num_blocks);

        auto dealloc_stop = std::chrono::high_resolution_clock::now();
        auto dealloc_duration = std::chrono::duration_cast<std::chrono::milliseconds>(dealloc_stop - dealloc_start);
        float dealloc_duration_f = static_cast<float>(dealloc_duration.count());

        if(i != 0) {allocation_times.push_back(alloc_duration_f); allocation_times.push_back(dealloc_duration_f);}
    }

    //Clean Results and get output
    std::vector<float> cleaned_alloc_times = clean_data_mad(allocation_times);
    MADResult alloc_results = calculate_mad(cleaned_alloc_times);
    std::unordered_map<int, int> anomaly_counts_alloc = find_anomalies_mad(cleaned_alloc_times, alloc_results);

    //Calculate the P values for anomalous data
    double p_val_1 = binomialProbability(cleaned_alloc_times.size(), anomaly_counts_alloc[3], stats::PROB_OUTSIDE_3SD);
    double p_val_2 = binomialProbability(cleaned_alloc_times.size(), anomaly_counts_alloc[4], stats::PROB_OUTSIDE_4SD);
    double p_val_3 = binomialProbability(cleaned_alloc_times.size(), anomaly_counts_alloc[5], stats::PROB_OUTSIDE_5SD);

    if(p_val_1 < 0.01){
        std::cout << "Anomalous allocation data discarding results\n" << anomaly_counts_alloc[3]
        << " results are more than three standard deviations away from the mean\n";
        return;
    }
    if(p_val_2 < 0.01){
        std::cout << "Anomalous allocation data discarding results\n" << anomaly_counts_alloc[4]
        << " results are more than four standard deviations away from the mean\n";
        return;
    }
    if(p_val_3 < 0.01){
        std::cout << "Anomalous allocation data discarding results\n" << anomaly_counts_alloc[5]
        << " results are more than five standard deviations away from the mean\n";
        return;
    }

    //Clean Results and get output
    std::vector<float> cleaned_dealloc_times = clean_data_mad(deallocation_times);
    MADResult dealloc_results = calculate_mad(cleaned_dealloc_times);
    std::unordered_map<int, int> anomaly_counts_dealloc = find_anomalies_mad(cleaned_dealloc_times, dealloc_results);

    //Calculate the P values for anomalous data
    p_val_1 = binomialProbability(cleaned_dealloc_times.size(), anomaly_counts_dealloc[3], stats::PROB_OUTSIDE_3SD);
    p_val_2 = binomialProbability(cleaned_dealloc_times.size(), anomaly_counts_dealloc[4], stats::PROB_OUTSIDE_4SD);
    p_val_3 = binomialProbability(cleaned_dealloc_times.size(), anomaly_counts_dealloc[5], stats::PROB_OUTSIDE_5SD);

    if(p_val_1 < 0.01){
        std::cout << "Anomalous deallocation data discarding results\n" << anomaly_counts_dealloc[3]
        << " results are more than three standard deviations away from the mean\n";
        return;
    }
    if(p_val_2 < 0.01){
        std::cout << "Anomalous deallocation data discarding results\n" << anomaly_counts_dealloc[4]
        << " results are more than four standard deviations away from the mean\n";
        return;
    }
    if(p_val_3 < 0.01){
        std::cout << "Anomalous deallocation data discarding results\n" << anomaly_counts_dealloc[5]
        << " results are more than five standard deviations away from the mean\n";
        return;
    }

    StatsResult alloc_stats = calculate_statistics(allocation_times);
    StatsResult dealloc_stats = calculate_statistics(deallocation_times);

    std::cout << "Allocation Time Data:\n";
    std::cout << "Mean: " << alloc_stats.mean << " ms\n";
    std::cout << "Standard Deviation: " << alloc_stats.std_dev << " ms\n";

    std::cout << "Deallocation Time Data:\n";
    std::cout << "Mean: " << dealloc_stats.mean << " ms\n";
    std::cout << "Standard Deviation: " << dealloc_stats.std_dev << " ms\n";
}

//Generates a random ALTO tensor based on your parameters and tests encoding
template<typename T, typename S>
void test_kernel(std::string filename, int mode, int nnz, int rank, std::vector<int> dims, int iterations)
{
    std::cout << "Benchmarking BLCO tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";

    std::vector<NNZ_Entry<T>> test_vec;
    if(filename == "-none"){
        int min_dim = *(std::min_element(dims.begin(), dims.end()));
        int block_size = (0.05 * min_dim) + 1;
        int max_blocks = (nnz + block_size - 1) / block_size;
        test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,block_size,max_blocks);
    }
    else test_vec = read_tensor_file_binary<T>(filename, rank, nnz);

    Blco_Tensor<T,S> blco(test_vec,dims);
    bool is_csr = blco.get_total_bits_needed() > 74;

    std::vector<T> output_matrix;
    std::vector<float> temp_vec = {0.0f};
    if(rank == 3 && !is_csr) output_matrix = MTTKRP_BLCO_3D<T,S>(mode, blco, temp_vec);
    else if(rank == 4 && !is_csr) output_matrix = MTTKRP_BLCO_4D<T,S>(mode, blco, temp_vec);
    else if(rank == 5 && !is_csr) output_matrix = MTTKRP_BLCO_5D<T,S>(mode, blco, temp_vec);
    else if(rank == 4 && is_csr) output_matrix = MTTKRP_BLCO_CSR_4D<T,S>(mode, blco, temp_vec);
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

    if(!test_passed) {std::cout << "test failed terminating benchmarking\n"; return;}

    std::vector<float> times;
    std::vector<T> temp;
    if(rank == 3 && !is_csr) temp = MTTKRP_BLCO_3D<T,S>(mode, blco, times, iterations); 
    else if(rank == 4 && !is_csr) temp = MTTKRP_BLCO_4D<T,S>(mode, blco, times, iterations); 
    else if(rank == 5 && !is_csr) temp = MTTKRP_BLCO_5D<T,S>(mode, blco, times, iterations); 
    else if(rank == 4 && is_csr) output_matrix = MTTKRP_BLCO_CSR_4D<T,S>(mode, blco, times, iterations);

    std::vector<float> cleaned_times = clean_data_mad(times);
    MADResult mad_results = calculate_mad(cleaned_times);
    std::unordered_map<int, int> anomaly_counts = find_anomalies_mad(cleaned_times, mad_results);

    //Calculate the P values for anomalous data
    double p_val_1 = binomialProbability(cleaned_times.size(), anomaly_counts[3], stats::PROB_OUTSIDE_3SD);
    double p_val_2 = binomialProbability(cleaned_times.size(), anomaly_counts[4], stats::PROB_OUTSIDE_4SD);
    double p_val_3 = binomialProbability(cleaned_times.size(), anomaly_counts[5], stats::PROB_OUTSIDE_5SD);

    if(p_val_1 < 0.01){
        std::cout << "Anomalous data discarding results\n" << anomaly_counts[3]
        << " results are more than three standard deviations away from the mean\n";
        return;
    }
    if(p_val_2 < 0.01){
        std::cout << "Anomalous data discarding results\n" << anomaly_counts[4]
        << " results are more than four standard deviations away from the mean\n";
        return;
    }
    if(p_val_3 < 0.01){
        std::cout << "Anomalous data discarding results\n" << anomaly_counts[5]
        << " results are more than five standard deviations away from the mean\n";
        return;
    }

    StatsResult standard_result = calculate_statistics(cleaned_times);

    std::cout << "MTTKRP Kernel Time Data:\n";
    std::cout << "Mean: " << standard_result.mean << " ms\n";
    std::cout << "Standard Deviation: " << standard_result.std_dev << " ms\n";
}

int main(int argc, char* argv[]) {
    if (argc < 9 || argc > 12) {
        std::cerr << "Usage: " << argv[0] 
                  << "kernel <filename or -none> <mode> <nnz> (three to five different dimensions) <Type> <Num iterations>\n"
                  << "or: allocation <filename or -none> <nnz> (three to five different dimensions) <Type> <Num iterations>\n";
        return 1;
    }
    else if (argv[1] == "kernel" && argc == 9) {
        std::cerr << "Kernel Testing Usage: " << argv[0] 
                  << "kernel <filename or -none> <mode> <nnz> (three to five different dimensions) <Type> <Num iterations>\n";
        return 1;
    }
    else if (argv[1] == "allocation" && argc == 12) {
        std::cerr << "Allocation Testing Usage: " << argv[0] 
                  << "allocation <filename or -none> <nnz> (three to five different dimensions) <Type> <Num iterations>\n";
        return 1;
    }
    else if (std::string(argv[1]) == "allocation") {
        std::string file = std::string(argv[2]);
        int nnz = std::stoi(argv[3]);
        int rank = argc - 6;
        std::string type = std::string(argv[argc - 2]);
        int iter = std::stoi(argv[argc - 1]);
        std::vector<int> dimensions;
        for(int i = 4; i < argc - 2; ++i){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; ++i){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_allocation<int,uint64_t>(file, nnz, rank, dimensions, iter);
            else if(type == "float") test_allocation<float,uint64_t>(file, nnz, rank, dimensions, iter);
            else if(type == "long long") test_allocation<unsigned long long,uint64_t>(file, nnz, rank, dimensions, iter);
            else if(type == "double") test_allocation<double,uint64_t>(file, nnz, rank, dimensions, iter);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and unsigned long long\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_allocation<int,__uint128_t>(file, nnz, rank, dimensions, iter);
            else if(type == "float") test_allocation<float,__uint128_t>(file, nnz, rank, dimensions, iter);
            else if(type == "long int") test_allocation<unsigned long long,__uint128_t>(file, nnz, rank, dimensions, iter);
            else if(type == "double") test_allocation<double,__uint128_t>(file, nnz, rank, dimensions, iter);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and unsigned long long\n";
                return 1;
            }
        }
    }
    else if (std::string(argv[1]) == "kernel") {
        std::string file = std::string(argv[2]);
        int mode = std::stoi(argv[3]);
        int nnz = std::stoi(argv[4]);
        int rank = argc - 7;
        std::string type = std::string(argv[argc - 2]);
        int iter = std::stoi(argv[argc - 1]);
        std::vector<int> dimensions;
        for(int i = 5; i < argc - 2; ++i){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; ++i){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_kernel<int,uint64_t>(file, mode, nnz, rank, dimensions, iter);
            else if(type == "float") test_kernel<float,uint64_t>(file, mode, nnz, rank, dimensions, iter);
            else if(type == "long int") test_kernel<unsigned long long,uint64_t>(file, mode, nnz, rank, dimensions, iter);
            else if(type == "double") test_kernel<double,uint64_t>(file, mode, nnz, rank, dimensions, iter);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and unsigned long long\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_kernel<int,__uint128_t>(file, mode, nnz, rank, dimensions, iter);
            else if(type == "float") test_kernel<float,__uint128_t>(file, mode, nnz, rank, dimensions, iter);
            else if(type == "long int") test_kernel<unsigned long long,__uint128_t>(file, mode, nnz, rank, dimensions, iter);
            else if(type == "double") test_kernel<double,__uint128_t>(file, mode, nnz, rank, dimensions, iter);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and unsigned long long\n";
                return 1;
            }
        }
    }
    else{
        std::cerr << "Specify whether you want to test the kernel or allocation\n";
        return 1;
    }

    return 0;
}