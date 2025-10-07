#include <iostream>
#include <fstream>
#include <cstdint>
#include <limits>
#include <string>

// =====================
// Tensor Structures
// =====================
template <typename T>
struct TensorEntry {
    int32_t i, j, k;  // coordinates
    T val;            // value (templated)
};

struct TensorHeader {
    int32_t rows;
    int32_t cols;
    int32_t depth;
    int64_t nnz;
};

// =====================
// Converter Function
// =====================
template <typename T>
void convertTxtToBin(const std::string &inputFile, const std::string &outputFile) {
    std::ifstream fin(inputFile);
    if (!fin.is_open()) {
        throw std::runtime_error("Error: could not open input file " + inputFile);
    }

    std::ofstream fout(outputFile, std::ios::binary);
    if (!fout.is_open()) {
        throw std::runtime_error("Error: could not open output file " + outputFile);
    }

    // First pass: find max indices and nnz
    int32_t max_i = 0, max_j = 0, max_k = 0;
    int64_t nnz = 0;
    TensorEntry<T> entry;

    while (fin >> entry.i >> entry.j >> entry.k >> entry.val) {
        max_i = std::max(max_i, entry.i);
        max_j = std::max(max_j, entry.j);
        max_k = std::max(max_k, entry.k);
        nnz++;
    }

    // Rewind input
    fin.clear();
    fin.seekg(0, std::ios::beg);

    // Write header
    TensorHeader header;
    header.rows  = max_i + 1;  // assuming 0-based indexing
    header.cols  = max_j + 1;
    header.depth = max_k + 1;
    header.nnz   = nnz;

    fout.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write entries
    while (fin >> entry.i >> entry.j >> entry.k >> entry.val) {
        fout.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    }

    fin.close();
    fout.close();

    std::cout << "Converted " << nnz << " entries to binary file " << outputFile << "\n";
    std::cout << "Tensor dimensions: "
              << header.rows << " x "
              << header.cols << " x "
              << header.depth << "\n";
}

// =====================
// Main with type switch
// =====================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " input.txt output.bin [int|float|double]\n";
        return 1;
    }

    std::string inputFile  = argv[1];
    std::string outputFile = argv[2];
    std::string type       = argv[3];

    try {
        if (type == "int") {
            convertTxtToBin<int>(inputFile, outputFile);
        } else if (type == "float") {
            convertTxtToBin<float>(inputFile, outputFile);
        } else if (type == "double") {
            convertTxtToBin<double>(inputFile, outputFile);
        } else {
            throw std::runtime_error("Unsupported type: " + type +
                                     ". Use int, float, or double.");
        }
    } catch (const std::exception &e) {
        std::cerr << "Conversion failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
