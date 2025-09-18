#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include "solver.h"
#include "parameters.h"
#include "helper.h"

std::tuple<int, int, double, int, std::string, std::string, double> readInputFile(const std::string& filename) {
    std::ifstream input_file(filename);
    if (!input_file) {
        std::cerr << "Error opening input file: " << filename << std::endl;
        return {};
    }

    std::string line;
    std::vector<std::string> values;
    std::string algorithm;
    double theta;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string key;
        char equalSign;
        if (iss >> key >> equalSign && equalSign == '=') {
            if (key == "algorithm") {
                iss >> algorithm;
                if (iss >> theta) {
                    std::cout << "Key: " << key << ", Value: " << algorithm << ", Theta: " << theta << std::endl;
                } else {
                    theta = 0.0;
                    if (algorithm == "hybridExplicitImplicit") {
                        std::cout << "Theta not provided for hybridExplicitImplicit, a default value of 0.0 has been assigned" << std::endl;
                    } else {
                        std::cout << "Key: " << key << ", Value: " << algorithm << std::endl;
                    }
                }
            } else {
                std::string value;
                if (iss >> value) {
                    std::cout << "Key: " << key << ", Value: " << value << std::endl;
                    values.push_back(value);
                }
            }
            
        }
    }
    int N = std::stoi(values[0]);
    double outletBoundaryCondition = std::stod(values[1]);
    double CFL = std::stod(values[2]);
    int it_max = std::stoi(values[3]);
    std::string output_filename = values[4];

    return {N, outletBoundaryCondition, CFL, it_max, output_filename, algorithm, theta};
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument

    std::cout << "Input file: " << input_filename << std::endl;

    auto [N, outletBoundaryCondition, CFL, it_max, output_filename, algorithm, theta] = readInputFile(input_filename);

    std::vector<std::string> valid_algorithms = {"macCormack", "beamWarming"};
    if (std::find(valid_algorithms.begin(), valid_algorithms.end(), algorithm) == valid_algorithms.end()) {
        std::cerr << "Invalid algorithm specified: " << algorithm << std::endl;
        std::cerr << "Valid options are: ";
        for (const auto& alg : valid_algorithms) {
            std::cerr << alg << "  ";
        }
        std::cerr << std::endl;
        return 1;
    }
    std::cout << "Using algorithm: " << algorithm << std::endl;

    double dx = (10 - 0) / (N - 1);

    parameters params(N, outletBoundaryCondition, CFL, dx, output_filename, theta);
    params.print();

    auto [x, A, Q1, Q2, Q3] = initializeQ(params);
    auto [E1, E2, E3] = initializeE(params, A, Q1, Q2, Q3);
    auto [S1, S2, S3] = initializeS(params, x, A, Q1, Q2, Q3);

    printVector(x, "x");
    printVector(A, "A");
    printVector(Q1, "Q1");
    printVector(Q2, "Q2");
    printVector(Q3, "Q3");
    printVector(E1, "E1");
    printVector(E2, "E2");
    printVector(E3, "E3");
    printVector(S1, "S1");
    printVector(S2, "S2");
    printVector(S3, "S3");

    for (unsigned int i = 0; i < params.N_; ++i) {
        auto [rho, u, e, p] = primitiveVariables(Q1[i], Q2[i], Q3[i], A[i], params.gamma_);
        std::cout << "Cell " << i << ": rho = " << rho << ", u = " << u << ", e = " << e << ", p = " << p << std::endl;
    }

    double res1 = 1e12;
    double res2 = 1e12;
    double res3 = 1e12;
    int it = 1;

    if (algorithm == "macCormack") {
        while (res1 > 1e-12) {
            std::vector<double> Q1_np1(params.N_);
            std::vector<double> Q2_np1(params.N_);
            std::vector<double> Q3_np1(params.N_);

            updateBoundaryConditions(params, Q1, Q2, Q3);

            macCormack(params, x, A, Q1, Q2, Q3, Q1_np1, Q2_np1, Q3_np1, E1, E2, E3, S1, S2, S3);

            res1 = l2Norm(Q1_np1, Q1);
            res2 = l2Norm(Q2_np1, Q2);
            res3 = l2Norm(Q3_np1, Q3);

            std::cout << "Iteration " << it << " res: " << res1 << " " << res2 << " " << res3 << std::endl;

            if (it >= it_max) {
                break;
            }

            it++;
        }
    }
    

    return 0;
}