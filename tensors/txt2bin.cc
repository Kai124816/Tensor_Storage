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
void convertTxtToBin(const std::string &inputFile, const std::string &outputFile, int rank, bool writeHeader = false) {
    std::ifstream fin(inputFile);
    if (!fin.is_open()) {
        throw std::runtime_error("Error: could not open input file " + inputFile);
    }

    // Pass 1: Count NNZ and validate format
    int64_t nnz = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines
        nnz++;
    }

    // Rewind
    fin.clear();
    fin.seekg(0, std::ios::beg);

    std::ofstream fout(outputFile, std::ios::binary);
    if (!fout.is_open()) {
        throw std::runtime_error("Error: could not open output file " + outputFile);
    }

    // ONLY write header if you update your reader to expect it!
    // Our previous 'read_tensor_file_binary' assumes NO header.
    if (writeHeader) {
        int32_t r32 = static_cast<int32_t>(rank);
        fout.write(reinterpret_cast<const char*>(&r32), sizeof(int32_t));
        // Note: Dimensions are omitted here for brevity as they require a 
        // third pass or storing all max values during Pass 1.
        fout.write(reinterpret_cast<const char*>(&nnz), sizeof(int64_t));
    }

    // Pass 2: Write entries
    std::vector<int32_t> coords(rank);
    T value;
    int64_t processed = 0;

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        for (int d = 0; d < rank; ++d) {
            if (!(ss >> coords[d])) {
                std::cerr << "Error parsing coordinate at line " << processed + 1 << std::endl;
            }
        }
        if (!(ss >> value)) {
            // Some tensors have no value; default to 1.0 if missing
            value = static_cast<T>(1);
        }

        // Write [coords][value]
        fout.write(reinterpret_cast<const char*>(coords.data()), rank * sizeof(int32_t));
        fout.write(reinterpret_cast<const char*>(&value), sizeof(T));
        
        processed++;
        if (processed % 1000000 == 0) {
            std::cout << "\rProgress: " << (processed * 100 / nnz) << "%" << std::flush;
        }
    }

    std::cout << "\nDone. Converted " << processed << " entries.\n";
    fin.close();
    fout.close();
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