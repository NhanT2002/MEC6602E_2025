#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include "solver.h"
#include "parameters.h"

std::tuple<int, double, double, double, std::string> readInputFile(const std::string& filename) {
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
    double t_final = std::stod(values[3]);
    std::string output_filename = values[4];

    return {N, c, CFL, t_final, output_filename};
}

std::tuple<std::vector<double>, std::vector<double>> initializeWaveFields(int N, double xi, double xf) {
    std::vector<double> x(N);
    std::vector<double> u(N);
    double dx = (xf - xi) / (N - 1);
    for (int i = 0; i < N; ++i) {
        x[i] = xi + i * dx;
        if (0.5 <= x[i] && x[i] <= 1.0) {
            u[i] = 1.0;
        } 
        else {
            u[i] = 0.0;
        }
    }
    return {x, u};
}

void writeSolutionToFile(const std::string& filename, const double t, const std::vector<double>& u) {
    std::ofstream output_file(filename, std::ios::app);
    if (!output_file) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    output_file << t << " ";

    for (size_t i = 0; i < u.size(); ++i) {
        output_file << u[i] << " ";
    }
    output_file << std::endl;
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument


    std::cout << "Input file: " << input_filename << std::endl;

    auto [N, c, CFL, t_final, output_filename] = readInputFile(input_filename);

    auto [x, u] = initializeWaveFields(N, 0.0, M_PI);

    double dx = (x[1] - x[0]);
    double dt = CFL * dx / c;
    parameters params(N, c, CFL, t_final, dx, dt, output_filename);
    params.print();

    std::vector<double> u_np1(N);
    std::ofstream clear_file(params.output_filename_, std::ios::trunc);
    writeSolutionToFile(params.output_filename_, 0, x);
    for (double t = 0; t < t_final; t += dt) {
        writeSolutionToFile(params.output_filename_, t, u);
        // explicitBackward(params, u, u_np1);
        // explicitForward(params, u, u_np1);
        forwardTimeCenteredSpace(params, u, u_np1);
        u = u_np1;
    }

    return 0;
}