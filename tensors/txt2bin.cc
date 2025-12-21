#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

// =====================
// N-Dimensional Structures
// =====================

/**
 * For binary storage, we write:
 * 1. int32_t rank
 * 2. int32_t dimensions[rank]
 * 3. int64_t nnz
 * 4. Data blocks: [coord_0, coord_1, ..., coord_N-1, value]
 */

// =====================
// Converter Function
// =====================
template <typename T>
void convertTxtToBin(const std::string &inputFile, const std::string &outputFile, int rank) {
    if (rank <= 0) {
        throw std::invalid_argument("Rank must be a positive integer.");
    }

    std::ifstream fin(inputFile);
    if (!fin.is_open()) {
        throw std::runtime_error("Error: could not open input file " + inputFile);
    }

    // First pass: find max indices for each dimension and count nnz
    std::vector<int32_t> max_indices(rank, 0);
    int64_t nnz = 0;
    
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int32_t coord;
        for (int d = 0; d < rank; ++d) {
            if (!(ss >> coord)) break;
            max_indices[d] = std::max(max_indices[d], coord);
        }
        nnz++;
    }

    // Rewind input
    fin.clear();
    fin.seekg(0, std::ios::beg);

    std::ofstream fout(outputFile, std::ios::binary);
    if (!fout.is_open()) {
        throw std::runtime_error("Error: could not open output file " + outputFile);
    }

    // --- Write Header ---
    // 1. Rank
    fout.write(reinterpret_cast<const char*>(&rank), sizeof(int32_t));
    // 2. Dimensions (assuming 0-indexed input, so size = max_index + 1)
    for (int d = 0; d < rank; ++d) {
        int32_t dim_size = max_indices[d] + 1;
        fout.write(reinterpret_cast<const char*>(&dim_size), sizeof(int32_t));
    }
    // 3. NNZ
    fout.write(reinterpret_cast<const char*>(&nnz), sizeof(int64_t));

    // --- Write Entries ---
    std::vector<int32_t> coords(rank);
    T value;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        
        // Read coordinates
        for (int d = 0; d < rank; ++d) {
            ss >> coords[d];
        }
        // Read value
        ss >> value;

        // Write coordinates and value to binary
        fout.write(reinterpret_cast<const char*>(coords.data()), rank * sizeof(int32_t));
        fout.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    fin.close();
    fout.close();

    std::cout << "Successfully converted " << nnz << " entries (Rank " << rank << ") to " << outputFile << "\n";
}

// =====================
// Main with type switch
// =====================
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.txt> <output.bin> <rank> <int|float|double>\n"
                  << "Example: " << argv[0] << " data.txt data.bin 4 float\n";
        return 1;
    }

    std::string inputFile  = argv[1];
    std::string outputFile = argv[2];
    int rank               = std::stoi(argv[3]);
    std::string type       = argv[4];

    try {
        if (type == "int") {
            convertTxtToBin<int>(inputFile, outputFile, rank);
        } else if (type == "float") {
            convertTxtToBin<float>(inputFile, outputFile, rank);
        } else if (type == "double") {
            convertTxtToBin<double>(inputFile, outputFile, rank);
        } else {
            throw std::runtime_error("Unsupported type: " + type);
        }
    } catch (const std::exception &e) {
        std::cerr << "Conversion failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}