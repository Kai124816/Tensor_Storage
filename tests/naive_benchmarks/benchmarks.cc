#include "../../utility/utils.h"
#include "../../tensor_implementations/blco_impl.h"  
#include "../../gpu_code/naive_kernels.h"

//Generates a random ALTO tensor based on your parameters and tests encoding
template<typename T, typename S>
void test_kernel(std::string filename, int mode, int nnz, int rank, std::vector<int> dims, int iterations)
{
    std::cout << "Benchmarking BLCO tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";

    std::vector<NNZ_Entry<T>> test_vec = read_tensor_file_binary<T>(filename, rank, nnz);

    Blco_Tensor<T,S> blco(test_vec,dims);

    std::vector<float> temp_vec = {0.0f};
    std::vector<T> output_matrix =  MTTKRP_BLCO_Naive<T,S>(mode, blco, temp_vec);

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
    std::vector<T> temp =  MTTKRP_BLCO_Naive<T,S>(mode, blco, times, iterations);

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
    if (argc < 9 || argc > 11) {
        std::cerr << "Usage: " << argv[0] 
                  << "<filename> <mode> <nnz> (three to five different dimensions) <Type> <Num iterations>\n";
        return 1;
    }
    else{
        std::string file = std::string(argv[1]);
        int mode = std::stoi(argv[2]);
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

    return 0;
}