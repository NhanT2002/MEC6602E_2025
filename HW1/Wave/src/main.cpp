#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

std::tuple<int, double, double> readInputFile(const std::string& filename) {
    std::ifstream input_file(filename);
    if (!input_file) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return {};
    }

    std::string line;
    std::vector<std::string> values;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string key;
        char equalSign;
        if (iss >> key >> equalSign && equalSign == '=') {
            std::string value;
            if (iss >> value) {
                std::cout << "Key: " << key << ", Value: " << value << std::endl;
                values.push_back(value);
            }
        }
    }
    int N = std::stoi(values[0]);
    double c = std::stod(values[1]);
    double CFL = std::stod(values[2]);

    return {N, c, CFL};
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument

    std::cout << "Input file: " << input_filename << std::endl;

    auto [N, c, CFL] = readInputFile(input_filename);

    std::vector<double> x(N);
    std::vector<double> u(N);
    double dx = M_PI / (N - 1);
    for (int i = 0; i < N; ++i) {
        x[i] = i * dx;
        if (0.5 <= x[i] && x[i] <= 1.0) {
            u[i] = 1.0;
        } 
        else {
            u[i] = 0.0;
        }
        std::cout << "x[" << i << "] = " << x[i] << ", u[" << i << "] = " << u[i] << std::endl;

    }

    return 0;
}