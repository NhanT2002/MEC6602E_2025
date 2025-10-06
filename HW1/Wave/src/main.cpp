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

std::tuple<int, double, double, double, std::string, std::string, double> readInputFile(const std::string& filename) {
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
    double c = std::stod(values[1]);
    double CFL = std::stod(values[2]);
    double t_final = std::stod(values[3]);
    std::string output_filename = values[4];

    return {N, c, CFL, t_final, output_filename, algorithm, theta};
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

void writeObservationToFile(const std::string& filename, const double t_obs) {
    
    std::string filename_obs = filename;
    size_t pos = filename_obs.find_last_of('.'); 
    filename_obs.insert(pos, "_obs"); 

    std::ifstream input_file(filename);
    std::ofstream obs_file(filename_obs, std::ios::trunc);

    if (!input_file) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }

    if (!obs_file) {
        std::cerr << "Error opening output_obs file: " << filename_obs << std::endl;
        return;
    }

    std::string line;
    std::getline(input_file, line);
    obs_file << line << std::endl;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        double t;
        iss >> t; 
        if (t >= t_obs) {
            obs_file << line << std::endl; 
            break;
        }
    }

    input_file.close();
    obs_file.close();
}




int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument


    std::cout << "Input file: " << input_filename << std::endl;

    auto [N, c, CFL, t_final, output_filename, algorithm, theta] = readInputFile(input_filename);

    double xi = 0.0;
    double xf = M_PI;
    auto [x, u] = initializeWaveFields(N, xi, xf);

    double dx = (x[1] - x[0]);
    double dt = CFL * dx / c;
    parameters params(N, c, CFL, t_final, dx, dt, output_filename, theta);
    params.print();

    double x_obs = 2.5;
    double t_obs = (x_obs - (1.0 + 0.5) / 2.0) / params.c_;


    std::vector<std::string> valid_algorithms = {"explicitBackward", "explicitForward", "forwardTimeCenteredSpace", "leapFrog",
                                                 "laxWendroff", "lax", "hybridExplicitImplicit", "rungeKutta4", "tremblayTran"};
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
    


    std::vector<double> u_np1(N);
    std::ofstream clear_file(params.output_filename_, std::ios::trunc);
    writeSolutionToFile(params.output_filename_, 0, x);
    if (algorithm == "explicitBackward") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            explicitBackward(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "explicitForward") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            explicitForward(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "forwardTimeCenteredSpace") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            forwardTimeCenteredSpace(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "leapFrog") {
        std::vector<double> u_nm1(N);
        // Initialisation de u_nm1 avec forwardTimeCenteredSpace
        // %%%%%%%%%%%%%%% Est-ce qu'on devrait writeSolutionToFile cette étape? %%%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%% En ce moment, on considère que la condition intiale est appliquée au temps n-1 %%%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%% et que l'initialisation se fait entre les temps -1 et 0 %%%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%% (condition initiale t_-1 non enregistrée) %%%%%%%%%%%%%%%
        u_nm1 = u;
        explicitBackward(params, u_nm1, u);
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            leapFrog(params, u, u_np1, u_nm1);
            u_nm1 = u;
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "laxWendroff") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            laxWendroff(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "lax") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            lax(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "hybridExplicitImplicit") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            hybridExplicitImplicit(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "rungeKutta4") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            rungeKutta4(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    } else if (algorithm == "tremblayTran") {
        for (double t = 0; t < t_final; t += dt) {
            writeSolutionToFile(params.output_filename_, t, u);
            tremblayTran(params, u, u_np1);
            u = u_np1;
        }
        writeSolutionToFile(params.output_filename_, t_final, u);
    }


    writeObservationToFile(params.output_filename_, t_obs);

    return 0;
}