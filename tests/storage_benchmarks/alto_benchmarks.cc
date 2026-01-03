#include "../../utility/utils.h"
#include "../../tensor_implementations/alto_impl.h"   

//Generates a random ALTO tensor based on your parameters and tests encoding
template<typename T, typename S>
void test_alto_tensor(std::string filename, int nnz, int rank, std::vector<int> dims, int iterations)
{
    std::cout << "Benchmarking ALTO tensor\n";
    std::cout << "Tensor info ...\n" << "Rank " << rank << 
    "\n" << "Non Zero Entries " << nnz << "\n\n";

    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    std::vector<NNZ_Entry<T>> test_vec;
    if(filename == "-none"){
        int min_dim = *(std::min_element(dims.begin(), dims.end()));
        int block_size = 0.05 * min_dim;
        int max_blocks = (nnz + block_size - 1) / block_size;
        test_vec = generate_block_sparse_tensor_nd<T>(dims,nnz,0,100,block_size,max_blocks);
    }
    else test_vec = read_tensor_file_binary<T>(filename, dims);

    std::vector<float> times;
    bool terminate = false;
    for(int i = 0; i < iterations + 1; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        Alto_Tensor<T,S> alto(test_vec,dims);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        
        // Convert to float
        float duration_f = static_cast<float>(duration.count());
        if(i != 0) times.push_back(duration_f); //First run is a warm up run so it's results are discarded

        const std::vector<ALTOEntry<T,S>> alto_indexes = alto.get_alto();

        int not_found = 0;
        std::vector<std::vector<int>> visited;
        for(int i = 0; i < alto_indexes.size(); i++){
            std::vector<int> decoded_dims;
            for(int j = 0; j < rank; j++){
                decoded_dims.push_back(alto.get_mode_idx_alto(alto_indexes[i].linear_index,j + 1));
            }
            T val = alto_indexes[i].value;
            auto it = std::find(visited.begin(), visited.end(), decoded_dims);
            if(!find_entry(test_vec, decoded_dims, val) || it != visited.end()) not_found++;
        }
        
        if(not_found > 0){
            std::cout << "Tests Failed, terminating benchmarking\n";
            terminate = true;
            
        }

        if(terminate) break;
    }
    if(terminate) return;

    StatsResult results = calculate_statistics(times);
    std::unordered_map<int, int> anomaly_counts = find_anomalies(times, results);

    //Calculate the P values for anomalous data
    double p_val_1 = binomialProbability(iterations, anomaly_counts[3], stats::PROB_OUTSIDE_3SD);
    double p_val_2 = binomialProbability(iterations, anomaly_counts[4], stats::PROB_OUTSIDE_4SD);
    double p_val_3 = binomialProbability(iterations, anomaly_counts[5], stats::PROB_OUTSIDE_5SD);

    if(p_val_1 < 0.05){
        std::cout << "Anomalous data discarding results\n" << anomaly_counts[3]
        << " results are more than three standard deviations away from the mean\n";
        // return;
    }
    if(p_val_2 < 0.05){
        std::cout << "Anomalous data discarding results\n" << anomaly_counts[4]
        << " results are more than four standard deviations away from the mean\n";
        // return;
    }
    if(p_val_3 < 0.05){
        std::cout << "Anomalous data discarding results\n" << anomaly_counts[5]
        << " results are more than five standard deviations away from the mean\n";
        // return;
    }

    std::cout << "ALTO construction benchmarks \n";
    std::cout << "Mean: " << results.mean << "\n";
    std::cout << "Standard Deviation: " << results.std_dev << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 7 || argc > 12) {
        std::cerr << "Usage: " << argv[0] 
                  << "<filename> <nnz> (three to seven different dimensions) <Type> <Num iterations>\n";
        return 1;
    }
    else{
        std::string file = std::string(argv[1]);
        int nnz = std::stoi(argv[2]);
        int rank = argc - 4;
        std::string type = std::string(argv[argc - 2]);
        int iter = std::stoi(argv[argc - 1]);
        std::vector<int> dimensions;
        for(int i = 3; i < argc - 2; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_alto_tensor<int,uint64_t>(file, nnz, rank, dimensions, iter);
            else if(type == "float") test_alto_tensor<float,uint64_t>(file, nnz, rank, dimensions, iter);
            else if(type == "long int") test_alto_tensor<long int,uint64_t>(file, nnz, rank, dimensions, iter);
            else if(type == "double") test_alto_tensor<double,uint64_t>(file, nnz, rank, dimensions, iter);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_alto_tensor<int,__uint128_t>(file, nnz, rank, dimensions, iter);
            else if(type == "float") test_alto_tensor<float,__uint128_t>(file, nnz, rank, dimensions, iter);
            else if(type == "long int") test_alto_tensor<long int,__uint128_t>(file, nnz, rank, dimensions, iter);
            else if(type == "double") test_alto_tensor<double,__uint128_t>(file, nnz, rank, dimensions, iter);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }

    return 0;
}