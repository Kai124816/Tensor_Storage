#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <cstdio>   // for std::remove, std::rename

#include <string>
#include <cctype>

bool is_data_line(const std::string &line) {
    bool has_digit = false;

    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];

        if (std::isdigit(c)) {
            has_digit = true;
        }
        else if (c == '-' || c == '+') {
            // Allowed only if it's at the start or right after 'e'/'E'
            if (i > 0 && !(line[i-1] == 'e' || line[i-1] == 'E')) {
                return false;
            }
        }
        else if (c == '.' ) {
            // decimal point allowed, but not multiple consecutive
            // no extra check here, but could enforce stricter rules
        }
        else if (c == 'e' || c == 'E') {
            // must have seen at least one digit before
            if (!has_digit) return false;
        }
        else if (!std::isspace(static_cast<unsigned char>(c))) {
            return false; // invalid character
        }
    }

    return has_digit;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file.tns>\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string tmp_file   = input_file + ".tmp";

    std::ifstream fin(input_file);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open " << input_file << "\n";
        return 1;
    }

    std::ofstream fout(tmp_file);
    if (!fout.is_open()) {
        std::cerr << "Error: could not open " << tmp_file << "\n";
        return 1;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (is_data_line(line)) {
            fout << line << "\n";
        }
    }

    fin.close();
    fout.close();

    // Replace original file with cleaned file
    if (std::remove(input_file.c_str()) != 0) {
        std::cerr << "Error: could not remove original file\n";
        return 1;
    }
    if (std::rename(tmp_file.c_str(), input_file.c_str()) != 0) {
        std::cerr << "Error: could not rename temp file\n";
        return 1;
    }

    std::cout << "Cleaned file written in place: " << input_file << "\n";
    return 0;
}
