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
#include "writeSolution.h"

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

    double dx = (10.0 - 0.0) / (N - 1);

    parameters params(N, outletBoundaryCondition, CFL, dx, output_filename, theta);
    params.print();

    auto [x, A, Q1, Q2, Q3] = initializeQ(params);
    auto [E1, E2, E3] = initializeE(params, A, Q1, Q2, Q3);
    auto [S1, S2, S3] = initializeS(params, x, A, Q1, Q2, Q3);

    double res1 = 1e12;
    double res2 = 1e12;
    double res3 = 1e12;
    double firstRes1 = 0.0;
    double firstRes2 = 0.0;
    double firstRes3 = 0.0;
    int it = 1;

    std::vector<double> res1_history;
    std::vector<double> res2_history;
    std::vector<double> res3_history;

    if (algorithm == "macCormack") {
        std::vector<double> Q1_np1(params.N_+2);
        std::vector<double> Q2_np1(params.N_+2);
        std::vector<double> Q3_np1(params.N_+2);
        std::vector<double> rhorho(params.N_+2);
        std::vector<double> uu(params.N_+2);
        std::vector<double> ee(params.N_+2);
        std::vector<double> pp(params.N_+2);
        std::vector<double> machmach(params.N_+2);
        while (res1 > 1e-12) {

            updateBoundaryConditions(params, x, A, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3);

            macCormack(params, x, A, Q1, Q2, Q3, Q1_np1, Q2_np1, Q3_np1, E1, E2, E3, S1, S2, S3);

            for (unsigned int i = 1; i < params.N_+1; ++i) {
                auto [rho, u, e, p] = primitiveVariables(Q1_np1[i], Q2_np1[i], Q3_np1[i], A[i], params.gamma_);
                rhorho[i] = rho;
                uu[i] = u;
                ee[i] = e;
                pp[i] = p;
                machmach[i] = u / std::sqrt(params.gamma_ * p / rho);
            }

            res1 = l2Norm(Q1_np1, Q1);
            res2 = l2Norm(Q2_np1, Q2);
            res3 = l2Norm(Q3_np1, Q3);

            if (it == 1) {
                firstRes1 = res1;
                firstRes2 = res2;
                firstRes3 = res3;
            }

            res1 /= firstRes1;
            res2 /= firstRes2;
            res3 /= firstRes3;

            res1_history.push_back(res1);
            res2_history.push_back(res2);
            res3_history.push_back(res3);

            std::cout << "Iteration " << it << " res: " << res1 << " " << res2 << " " << res3 << std::endl;

            if (it >= it_max) {
                break;
            }

            Q1 = Q1_np1;
            Q2 = Q2_np1;
            Q3 = Q3_np1;


            auto [E1_new, E2_new, E3_new] = initializeE(params, A, Q1, Q2, Q3);
            auto [S1_new, S2_new, S3_new] = initializeS(params, x, A, Q1, Q2, Q3);

            E1 = E1_new;
            E2 = E2_new;
            E3 = E3_new;
            S1 = S1_new;
            S2 = S2_new;
            S3 = S3_new;

            it++;
        }

        writeSolution(x, Q1, Q2, Q3, E1, E2, E3, S1, S2, S3, rhorho, uu, pp, ee, machmach, params.output_filename_);
        writeConvergenceHistory(res1_history, res2_history, res3_history, "convergence_" + params.output_filename_);
    }
    

    return 0;
}