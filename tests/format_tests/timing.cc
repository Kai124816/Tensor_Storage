#include <utility>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "../../utility/utils.h"
#include "../../tensor_implementations/tensor_impl.h"
#include "../../tensor_implementations/alto_impl.h"
#include "../../tensor_implementations/blco_impl.h" 
#include "helper_functions.cc"    

//Generates a random ALTO/BLCO tensor based on a file and tests timing
template<typename T, typename S>
void time_tensor_file(int alto_or_blco, std::string filename, int nnz, int rank, std::vector<int> dims)
{   
    int default_decomp_rank = 10;
    std::vector<float> times;

    std::vector<std::vector<T>> generated_fmats;
    generated_fmats.resize(rank);
    for(int i = 0; i < rank; i++) generated_fmats[i] = generate_random_array_seed(
    dims[i] * default_decomp_rank, static_cast<T>(0), static_cast<T>(3), SEEDS[i]);

    bool correct = test_tensor_file<T,S>(alto_or_blco, filename, nnz, rank, dims);
    if(!correct){
        std::cout << "test failed terminating benchmarking" << std::endl;
        return;
    }

    float duration;
    if(alto_or_blco == 0){
        for(int i = 0; i < 100; i++){
            auto start = std::chrono::high_resolution_clock::now(); 
            Alto_Tensor<T,S> alto(filename, nnz, dims, generated_fmats, default_decomp_rank);
            auto stop = std::chrono::high_resolution_clock::now(); 
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); 
            times.push_back(duration);
        }
    }
    else{
        for(int i = 0; i < 100; i++){
            auto start = std::chrono::high_resolution_clock::now(); 
            Blco_Tensor<T,S> blco(filename, nnz, dims, generated_fmats, default_decomp_rank);
            auto stop = std::chrono::high_resolution_clock::now(); 
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); 
            times.push_back(duration);
        }
    }

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

    std::cout << "Construction time data\n";
    std::cout << "Mean: " << standard_result.mean << " ms\n";
    std::cout << "Standard Deviation: " << standard_result.std_dev << " ms\n";
}

//Generates a random ALTO/BLCO tensor synthetically and tests timing
template<typename T, typename S>
void time_tensor_synthetic(int alto_or_blco, int nnz, int rank, std::vector<int> dims)
{   
    int default_decomp_rank = 10;
    std::vector<float> times;

    std::vector<NNZ_Entry<T>> rand_test_vec = generate_block_sparse_tensor_nd<T>(dims, nnz, static_cast<T>(1), static_cast<T>(10), 5, 1000, 0.5f);
    bool correct = test_tensor_synthetic<T,S>(alto_or_blco, nnz, rank, dims);
    if(!correct){
        std::cout << "test failed terminating benchmarking" << std::endl;
        return;
    }

    Alto_Tensor<T,S> alto;
    Blco_Tensor<T,S> blco;
    float duration;
    if(alto_or_blco == 0){
        for(int i = 0; i < 100; i++){
            auto start = std::chrono::high_resolution_clock::now(); 
            Alto_Tensor<T,S> alto(rand_test_vec, dims, default_decomp_rank);
            auto stop = std::chrono::high_resolution_clock::now(); 
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); 
            times.push_back(duration);
        }
    }
    else{
        for(int i = 0; i < 100; i++){
            auto start = std::chrono::high_resolution_clock::now(); 
            Blco_Tensor<T,S> blco(rand_test_vec, dims, default_decomp_rank);
            auto stop = std::chrono::high_resolution_clock::now(); 
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(); 
            times.push_back(duration);
        }
    }

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

    std::cout << "Construction time data\n";
    std::cout << "Mean: " << standard_result.mean << " ms\n";
    std::cout << "Standard Deviation: " << standard_result.std_dev << " ms\n";
}

int main(int argc, char* argv[]) {
    // Expected arguments: <alto_or_blco> <filename> <nnz> (three to five dimensions) <Type>
    if (argc < 8 || argc > 10) {
        std::cerr << "Usage: " << argv[0] 
                  << " <alto_or_blco> <filename> <nnz> (three to five different dimensions) <Type>\n"
                  << " if you want to use a synthetically generated tensor use 'None' as your filename argument\n";
        return 1;
    }

    int alto_or_blco = std::stoi(argv[1]);
    std::string filename = std::string(argv[2]);
    int nnz = std::stoi(argv[3]);
    int rank = argc - 5;
    std::string type = std::string(argv[argc - 1]);
    
    std::vector<int> dimensions;
    for(int i = 4; i < argc - 1; i++){
        dimensions.push_back(std::stoi(argv[i]));
    }

    int bits_needed = 0;
    for(int i = 0; i < rank; i++){
        bits_needed += ceiling_log2(dimensions[i]);
    }

    auto run_times = [&](auto dummy1, auto dummy2) {
        using T = decltype(dummy1);
        using S = decltype(dummy2);
        
        if (filename == "None") {
            time_tensor_synthetic<T, S>(alto_or_blco, nnz, rank, dimensions);
        } else {
            time_tensor_file<T, S>(alto_or_blco, filename, nnz, rank, dimensions);
        }
    };

    if(bits_needed <= 64){
        if(type == "int") run_times(int{}, uint64_t{});
        else if(type == "float") run_times(float{}, uint64_t{});
        else if(type == "long int") run_times(0ULL, uint64_t{});
        else if(type == "double") run_times(double{}, uint64_t{});
        else{ 
            std::cerr << "Unsupported type. The supported types are int, float, long int, and double\n";
            return 1;
        }
    }
    else{
        if(type == "int") run_times(int{}, __uint128_t{});
        else if(type == "float") run_times(float{}, __uint128_t{});
        else if(type == "long int") run_times(0ULL, __uint128_t{});
        else if(type == "double") run_times(double{}, __uint128_t{});
        else{ 
            std::cerr << "Unsupported type. The supported types are int, float, long int, and double\n";
            return 1;
        }
    }

    return 0;
}