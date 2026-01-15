#include "../../utility/utils.h"
#include "../../tensor_implementations/blco_impl.h"   
#include "../../gpu_code/old_kernels.h" // Assuming Version 1 and 2 are here

// Helper to handle the validation against a naive CPU implementation
template<typename T, typename S>
bool validate_result(int mode, std::vector<T>& output, const Blco_Tensor<T,S>& blco, const std::vector<NNZ_Entry<T>>& nnz_vec) {
    std::vector<T*> fmats = blco.get_fmats();
    int decomp_rank = blco.get_factor_rank();
    std::vector<int> dims = blco.get_dims();
    
    T* input_matrix = create_and_copy_array(fmats[mode - 1], dims[mode - 1] * decomp_rank);
    T* test_matrix = MTTKRP_Naive(mode, input_matrix, fmats, decomp_rank, nnz_vec);

    bool passed = false;
    if constexpr (std::is_floating_point<T>::value){
        double diff = compare_arrays_float<T>(output.data(), test_matrix, dims[mode-1] * decomp_rank);
        std::cout << "Validation: Average difference = " << diff << "\n";
        passed = (diff < 0.00001);
    } else {
        passed = compare_arrays<T>(output.data(), test_matrix, dims[mode-1] * decomp_rank);
    }
    
    free(input_matrix);
    return passed;
}

// Benchmarking function for a specific version
template<typename T, typename S>
void benchmark_blco_version(int version, std::string filename, int mode, int nnz, int rank, std::vector<int> dims, int iterations)
{
    std::cout << "\n--- Benchmarking BLCO Version " << version << " ---\n";
    
    // 1. Data Loading/Generation
    std::vector<NNZ_Entry<T>> test_vec;
    if(filename == "-none"){
        int min_dim = *(std::min_element(dims.begin(), dims.end()));
        int block_size = (0.05 * min_dim) + 1;
        int max_blocks = (nnz + block_size - 1) / block_size;
        test_vec = generate_block_sparse_tensor_nd<T>(dims, nnz, 0, 100, block_size, max_blocks);
    } else {
        test_vec = read_tensor_file_binary<T>(filename, rank, nnz);
    }

    Blco_Tensor<T,S> blco(test_vec, dims);
    std::vector<float> warmup_times;
    std::vector<T> result;

    // 2. Validation & Warmup
    if (version == 1) result = MTTKRP_BLCO_v1<T,S>(mode, blco, warmup_times, 1);
    else              result = MTTKRP_BLCO_v2<T,S>(mode, blco, warmup_times, 1);

    if (!validate_result(mode, result, blco, test_vec)) {
        std::cout << "Validation FAILED for Version " << version << ". Skipping benchmark.\n";
        return;
    }
    std::cout << "Validation PASSED.\n";

    // 3. Execution for Iterations
    std::vector<float> times;
    if (version == 1) MTTKRP_BLCO_v1<T,S>(mode, blco, times, iterations);
    else              MTTKRP_BLCO_v2<T,S>(mode, blco, times, iterations);

    // 4. Statistics and Anomaly Detection
    std::vector<float> cleaned_times = clean_data_mad(times);
    MADResult mad_results = calculate_mad(cleaned_times);
    std::unordered_map<int, int> anomalies = find_anomalies_mad(cleaned_times, mad_results);

    // Binomial test for anomalies (as per your requirement)
    if (binomialProbability(cleaned_times.size(), anomalies[3], stats::PROB_OUTSIDE_3SD) < 0.01 ||
        binomialProbability(cleaned_times.size(), anomalies[4], stats::PROB_OUTSIDE_4SD) < 0.01 ||
        binomialProbability(cleaned_times.size(), anomalies[5], stats::PROB_OUTSIDE_5SD) < 0.01) {
        std::cout << "Anomalous data detected! Probability of outlier distribution is too high. Results might be unreliable.\n";
    }

    StatsResult stats = calculate_statistics(cleaned_times);
    std::cout << "Results for V" << version << " (after " << iterations << " iterations):\n";
    std::cout << "  Mean Execution Time: " << stats.mean << " ms\n";
    std::cout << "  Std Dev:             " << stats.std_dev << " ms\n";
}

int main(int argc, char* argv[]) {
    // Expected Usage: ./benchmark <version 1/2/all> <filename/-none> <mode> <nnz> <d1> <d2> <d3> <type> <iterations>
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << " <v1/v2/all> <file> <mode> <nnz> <d1> <d2> <d3> <type> <iters>\n";
        return 1;
    }

    std::string ver_choice = argv[1];
    std::string file = argv[2];
    int mode = std::stoi(argv[3]);
    int nnz = std::stoi(argv[4]);
    std::vector<int> dims = {std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7])};
    std::string type = argv[8];
    int iter = std::stoi(argv[9]);

    auto run_bench = [&](int v) {
        if (type == "float") benchmark_blco_version<float, uint64_t>(v, file, mode, nnz, 3, dims, iter);
        else if (type == "double") benchmark_blco_version<double, uint64_t>(v, file, mode, nnz, 3, dims, iter);
        else if (type == "int") benchmark_blco_version<int, uint64_t>(v, file, mode, nnz, 3, dims, iter);
    };

    if (ver_choice == "v1" || ver_choice == "all") run_bench(1);
    if (ver_choice == "v2" || ver_choice == "all") run_bench(2);

    return 0;
}