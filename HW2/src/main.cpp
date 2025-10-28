#include  "read_PLOT3D.h"
#include "SpatialDiscretization.h"
#include "TemporalDiscretization.h"
#include "Multigrid.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>

void read_input_file(const std::string& filename, double& Mach, double& alpha, double& p_inf, double& T_inf, int& multigrid, double& CFL_number, 
                     int& residual_smoothing, double& k2, double& k4, int& it_max, std::string& output_file, std::string& checkpoint_file, 
                     std::string& mesh_file, int& num_threads) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file." << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string key;
        char equalSign;
        if (iss >> key >> equalSign) {
            if (key == "Mach") {
                iss >> Mach;
            } else if (key == "alpha") {
                iss >> alpha;
                alpha = alpha * M_PI / 180.0;  // Convert alpha from degrees to radians
            } else if (key == "p_inf") {
                iss >> p_inf;
            } else if (key == "T_inf") {
                iss >> T_inf;
            } else if (key == "multigrid") {
                iss >> multigrid;
            } else if (key == "CFL_number") {
                iss >> CFL_number;
            } else if (key == "residual_smoothing") {
                iss >> residual_smoothing;
            } else if (key == "k2") {
                iss >> k2;
            } else if (key == "k4") {
                iss >> k4;
            } else if (key == "it_max") {
                iss >> it_max;
            } else if (key == "output_file") {
                iss >> output_file;
            } else if (key == "checkpoint_file") {
                iss >> checkpoint_file;
            } else if (key == "mesh_file") {
                iss >> mesh_file;
            } else if (key == "num_threads") {
                iss >> num_threads;
            }
        }
    }


    std::cout << "Read parameters from input file:\n";
    std::cout << "mesh_file = " << mesh_file << "\n";
    std::cout << "Mach = " << Mach << "\n";
    std::cout << "alpha = " << alpha / M_PI * 180.0<< "\n";
    std::cout << "p_inf = " << p_inf << "\n";
    std::cout << "T_inf = " << T_inf << "\n";
    std::cout << "multigrid = " << multigrid << "\n";
    std::cout << "CFL_number = " << CFL_number << "\n";
    std::cout << "residual_smoothing = " << residual_smoothing << "\n";
    std::cout << "k2 = " << k2 << "\n";
    std::cout << "k4 = " << k4 << "\n";
    std::cout << "it_max = " << it_max << "\n";

    inputFile.close();
}

int main(int argc, char* argv[]) {
    // Ensure the input file is passed as a command line argument
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1]; // The input filename is passed as the first argument
 

    // Read parameters from the input file
    double Mach, alpha, p_inf, T_inf, CFL_number, k2_coeff, k4_coeff;
    int multigrid, residual_smoothing, it_max, num_threads;
    std::string output_file, checkpoint_file, mesh_file;

    read_input_file(input_filename, Mach, alpha, p_inf, T_inf, multigrid, CFL_number, residual_smoothing, k2_coeff, k4_coeff, it_max, output_file, checkpoint_file, mesh_file, num_threads);

    omp_set_num_threads(num_threads); // Set number of threads
    int max_threads = omp_get_max_threads();
    int eig_threads = Eigen::nbThreads();
    std::cout << "Maximum available threads----------------: " << max_threads << std::endl;
    std::cout << "Eigen threads-----------------------------: " << eig_threads << std::endl;

    auto start = std::chrono::high_resolution_clock::now();


    // Read the PLOT3D mesh from a file
    auto [x, y] = read_PLOT3D_mesh(mesh_file);
    // std::cout << "x size: " << x.size() << " element" << std::endl;
    // std::cout << "y size: " << y.size() << " element" << std::endl;
    // std::cout << "x" << x << std::endl;
    // std::cout << "y" << y << std::endl;


    double rho_inf = p_inf/(T_inf*287);

    double a = std::sqrt(1.4*p_inf/rho_inf);
    double Vitesse = Mach*a;
    double u_inf = Vitesse*std::cos(alpha);
    double v_inf = Vitesse*std::sin(alpha);
    double E_inf = p_inf/((1.4-1)*rho_inf) + 0.5*std::pow(Vitesse, 2);

    double l_ref = 1.0;
    double U_ref = std::sqrt(p_inf/rho_inf);

    double rho = 1.0;
    double u = u_inf/U_ref;
    double v = v_inf/U_ref;
    double E = E_inf/(U_ref*U_ref);
    double T = 1.0;
    double p = 1.0;

    Eigen::ArrayXXd W_0, W_1, W_2, W_3;
    std::vector<std::vector<double>> iteration_residuals;
    std::vector<double> iteration_times;
    iteration_times = std::vector<double>{};
    iteration_residuals = std::vector<std::vector<double>>{};
    if (multigrid == 1) {        
        std::vector<std::vector<double>> Residuals;

        SpatialDiscretization h_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, Mach, U_ref);
        Multigrid multigrid_solver(h_state, CFL_number, residual_smoothing, k2_coeff, k4_coeff);

        SpatialDiscretization h2_state = multigrid_solver.mesh_restriction(h_state);
        // h2_state.k4_coeff = 4;
        multigrid_solver.restriction(h_state, h2_state);
        SpatialDiscretization h4_state = multigrid_solver.mesh_restriction(h2_state);
        // h4_state.k4_coeff = 4;
        multigrid_solver.restriction(h2_state, h4_state);
        SpatialDiscretization h8_state = multigrid_solver.mesh_restriction(h4_state);
        // h8_state.k4_coeff = 4;
        multigrid_solver.restriction(h4_state, h8_state);

        // std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h8_state, 10);
        // multigrid_solver.prolongation(h8_state, h4_state);
        // std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h4_state, 10);
        // multigrid_solver.prolongation(h4_state, h2_state);
        // std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h2_state, 10);
        // multigrid_solver.prolongation(h2_state, h_state); // Starting grid
        std::cout << "starting cycle\n";

        // W cycle        
        h_state.run_even();
        h_state.update_Rd0();
        for (int it = 1; it < it_max; it++) {
            std::tie(W_0, W_1, W_2, W_3, Residuals) = multigrid_solver.restriction_timestep(h_state, 1, it);

            // iteration_residuals.push_back({Residuals[0][0], Residuals[0][1], Residuals[0][2], Residuals[0][3]});
            // auto end_time = std::chrono::high_resolution_clock::now(); // End timer
            // std::chrono::duration<double> elapsed = end_time - start;
            // iteration_times.push_back(elapsed.count());

            if (multigrid_solver.multigrid_convergence) {

                // auto [W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex] = cell_dummy_to_vertex_centered_airfoil(W_0(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
                //                                                                                   W_1(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
                //                                                                                   W_2(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
                //                                                                                   W_3(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)));
                // std::ostringstream outputFile;
                // outputFile << "output_files/output_Mach_" << std::fixed << std::setprecision(2) << Mach
                //         << "_alpha_" << std::fixed << std::setprecision(2) << h_state.alpha*180/M_PI
                //         << "_mesh_" << mesh_file.substr(mesh_file.find_last_of('/') + 1) << ".q";
                // write_plot3d_2d(W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex, Mach, h_state.alpha, 0, 0, rho_inf, U_ref, outputFile.str());

                // h_state.alpha += 0.1*M_PI/180;
                // double u_inf = Vitesse*std::cos(h_state.alpha);
                // double v_inf = Vitesse*std::sin(h_state.alpha);
                // double u = u_inf/U_ref;
                // double v = v_inf/U_ref;
                // h_state.u = u;
                // h_state.v = v;
                // std::cout << "Converged at iteration: " << it << std::endl;
                // std::cout << "alpha: " << h_state.alpha << std::endl;
                // h2_state.alpha = h_state.alpha;
                // h2_state.u = h_state.u;
                // h2_state.v = h_state.v;
                // h4_state.alpha = h_state.alpha;
                // h4_state.u = h_state.u;
                // h4_state.v = h_state.v;
                // h8_state.alpha = h_state.alpha;
                // h8_state.u = h_state.u;
                // h8_state.v = h_state.v;
                // it = 0;

                

                break;
            }
            
            multigrid_solver.restriction(h_state, h2_state);
            multigrid_solver.restriction_timestep(h2_state, 1, -1);
            multigrid_solver.restriction(h2_state, h4_state);
            multigrid_solver.restriction_timestep(h4_state, 1, -1);
            multigrid_solver.restriction(h4_state, h8_state);
            multigrid_solver.restriction_timestep(h8_state, 1, -1);

            multigrid_solver.prolongation(h8_state, h4_state);
            multigrid_solver.restriction_timestep(h4_state, 1, -1);
            multigrid_solver.restriction(h4_state, h8_state);
            multigrid_solver.restriction_timestep(h8_state, 1, -1);
            multigrid_solver.prolongation(h8_state, h4_state);
            multigrid_solver.prolongation(h4_state, h2_state);
            multigrid_solver.restriction_timestep(h2_state, 1, -1);

            multigrid_solver.restriction(h2_state, h4_state);
            multigrid_solver.restriction_timestep(h4_state, 1, -1);
            multigrid_solver.restriction(h4_state, h8_state);
            multigrid_solver.restriction_timestep(h8_state, 1, -1);
            multigrid_solver.prolongation(h8_state, h4_state);
            multigrid_solver.restriction_timestep(h4_state, 1, -1);
            multigrid_solver.restriction(h4_state, h8_state);
            multigrid_solver.restriction_timestep(h8_state, 1, -1);

            multigrid_solver.prolongation(h8_state, h4_state);
            multigrid_solver.prolongation(h4_state, h2_state);
            multigrid_solver.prolongation_smooth(h2_state, h_state);
            h_state.update_conservative_variables();
            h_state.run_odd();
            

            
        }
        // save_time_residuals(iteration_times, iteration_residuals, checkpoint_file);
    }

    else {
        TemporalDiscretization FVM(x, y, rho, u, v, E, T, p, Mach, U_ref, CFL_number, residual_smoothing, k2_coeff, k4_coeff);
        std::tie(W_0, W_1, W_2, W_3, iteration_residuals, iteration_times) = FVM.RungeKutta(it_max);
        // for (int i = 0; i < it_max; i++) {            
        //     std::tie(W_0, W_1, W_2, W_3, iteration_residuals, iteration_times) = FVM.RungeKutta(it_max);
        //     save_time_residuals(iteration_times, iteration_residuals, checkpoint_file);

        //     auto [W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex] = cell_dummy_to_vertex_centered_airfoil(W_0(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
        //                                                                                           W_1(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
        //                                                                                           W_2(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
        //                                                                                           W_3(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)));
        //     std::ostringstream outputFile;
        //     outputFile << "output_files/output_Mach_" << std::fixed << std::setprecision(2) << Mach
        //             << "_alpha_" << std::fixed << std::setprecision(2) << FVM.current_state.alpha*180/M_PI
        //             << "_mesh_" << mesh_file.substr(mesh_file.find_last_of('/') + 1) << ".q";
        //     write_plot3d_2d(W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex, Mach, FVM.current_state.alpha, 0, 0, rho_inf, U_ref, outputFile.str());

        //     if (FVM.current_state.alpha >= 17*M_PI/180) {
        //         break;
        //     }

        //     FVM.current_state.alpha += 0.1*M_PI/180;
        //     u_inf = Vitesse*std::cos(FVM.current_state.alpha);
        //     v_inf = Vitesse*std::sin(FVM.current_state.alpha);
        //     FVM.current_state.u = u_inf/U_ref;
        //     FVM.current_state.v = v_inf/U_ref;

        // }
        
        // SpatialDiscretization h_state(x, y, rho, u, v, E, T, p, k2_coeff, k4_coeff, Mach, U_ref);
        // h_state.run_even();
        // std::cout << "fluxy_0\n" << h_state.fluxy_0 << std::endl;
        // std::cout << "fluxy_1\n" << h_state.fluxy_1 << std::endl;
    }

    


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serialDuration = end - start;
    std::cout << "\nSolver duration: " << serialDuration.count() << " seconds\n";

    auto [W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex] = cell_dummy_to_vertex_centered_airfoil(W_0(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
                                                                                                  W_1(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
                                                                                                  W_2(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)),
                                                                                                  W_3(Eigen::seq(1, W_0.rows()-2), Eigen::seq(1, W_0.cols()-2)));
    write_plot3d_2d(W_0_vertex, W_1_vertex, W_2_vertex, W_3_vertex, Mach, alpha, 0, 0, rho_inf, U_ref, output_file);

    return 0;

}














