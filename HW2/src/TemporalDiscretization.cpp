#include "TemporalDiscretization.h"
#include "mesh.h"
#include "helper.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

TemporalDiscretization::TemporalDiscretization(SpatialDiscretization &spatialDiscretization, double CFL, int it_max)
    : spatialDiscretization_(spatialDiscretization), CFL_(CFL), it_max_(it_max) {

    int ncells = spatialDiscretization_.ncells_;
    dt_cells.resize(ncells);
}

void TemporalDiscretization::compute_dt() {

    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double lambdaI = spatialDiscretization_.LambdaI[cell];
        double lambdaJ = spatialDiscretization_.LambdaJ[cell];
        double volume = spatialDiscretization_.mesh_.volume[cell];

        double dt_cell = CFL_ * volume / (lambdaI + lambdaJ);
        dt_cells[cell] = dt_cell;
    }
}

void TemporalDiscretization::eulerStep() {
    spatialDiscretization_.run_odd();
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double dt = dt_cells[cell];
        spatialDiscretization_.W0[cell] -= dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc0[cell] - spatialDiscretization_.Rd0[cell]);
        spatialDiscretization_.W1[cell] -= dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc1[cell] - spatialDiscretization_.Rd1[cell]);
        spatialDiscretization_.W2[cell] -= dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc2[cell] - spatialDiscretization_.Rd2[cell]);
        spatialDiscretization_.W3[cell] -= dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc3[cell] - spatialDiscretization_.Rd3[cell]);
    }
}

void TemporalDiscretization::RungeKuttaStep() {
    std::vector<double> W0_0 = spatialDiscretization_.W0;
    std::vector<double> W1_0 = spatialDiscretization_.W1;
    std::vector<double> W2_0 = spatialDiscretization_.W2;
    std::vector<double> W3_0 = spatialDiscretization_.W3;

    // Stage 1
    spatialDiscretization_.run_odd();
    std::vector<double> Rd0_0 = spatialDiscretization_.Rd0;
    std::vector<double> Rd1_0 = spatialDiscretization_.Rd1;
    std::vector<double> Rd2_0 = spatialDiscretization_.Rd2;
    std::vector<double> Rd3_0 = spatialDiscretization_.Rd3;
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double dt = dt_cells[cell];
        spatialDiscretization_.W0[cell] = W0_0[cell] - a1 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc0[cell] - Rd0_0[cell]);
        spatialDiscretization_.W1[cell] = W1_0[cell] - a1 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc1[cell] - Rd1_0[cell]);
        spatialDiscretization_.W2[cell] = W2_0[cell] - a1 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc2[cell] - Rd2_0[cell]);
        spatialDiscretization_.W3[cell] = W3_0[cell] - a1 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc3[cell] - Rd3_0[cell]);
    }

    // Stage 2
    spatialDiscretization_.run_even();
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double dt = dt_cells[cell];
        spatialDiscretization_.W0[cell] = W0_0[cell] - a2 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc0[cell] - Rd0_0[cell]);
        spatialDiscretization_.W1[cell] = W1_0[cell] - a2 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc1[cell] - Rd1_0[cell]);
        spatialDiscretization_.W2[cell] = W2_0[cell] - a2 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc2[cell] - Rd2_0[cell]);
        spatialDiscretization_.W3[cell] = W3_0[cell] - a2 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc3[cell] - Rd3_0[cell]);
    }

    // Stage 3 - update dissipative residuals
    spatialDiscretization_.run_odd();
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        Rd0_0[cell] = b3 * spatialDiscretization_.Rd0[cell] + (1.0 - b3) * Rd0_0[cell];
        Rd1_0[cell] = b3 * spatialDiscretization_.Rd1[cell] + (1.0 - b3) * Rd1_0[cell];
        Rd2_0[cell] = b3 * spatialDiscretization_.Rd2[cell] + (1.0 - b3) * Rd2_0[cell];
        Rd3_0[cell] = b3 * spatialDiscretization_.Rd3[cell] + (1.0 - b3) * Rd3_0[cell];
    }
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double dt = dt_cells[cell];
        spatialDiscretization_.W0[cell] = W0_0[cell] - a3 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc0[cell] - Rd0_0[cell]);
        spatialDiscretization_.W1[cell] = W1_0[cell] - a3 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc1[cell] - Rd1_0[cell]);
        spatialDiscretization_.W2[cell] = W2_0[cell] - a3 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc2[cell] - Rd2_0[cell]);
        spatialDiscretization_.W3[cell] = W3_0[cell] - a3 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc3[cell] - Rd3_0[cell]);
    }

    // Stage 4
    spatialDiscretization_.run_even();
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double dt = dt_cells[cell];
        spatialDiscretization_.W0[cell] = W0_0[cell] - a4 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc0[cell] - Rd0_0[cell]);
        spatialDiscretization_.W1[cell] = W1_0[cell] - a4 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc1[cell] - Rd1_0[cell]);
        spatialDiscretization_.W2[cell] = W2_0[cell] - a4 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc2[cell] - Rd2_0[cell]);
        spatialDiscretization_.W3[cell] = W3_0[cell] - a4 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc3[cell] - Rd3_0[cell]);
    }

    // Stage 5 - final update of dissipative residuals
    spatialDiscretization_.run_odd();
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        Rd0_0[cell] = b5 * spatialDiscretization_.Rd0[cell] + (1.0 - b5) * Rd0_0[cell];
        Rd1_0[cell] = b5 * spatialDiscretization_.Rd1[cell] + (1.0 - b5) * Rd1_0[cell];
        Rd2_0[cell] = b5 * spatialDiscretization_.Rd2[cell] + (1.0 - b5) * Rd2_0[cell];
        Rd3_0[cell] = b5 * spatialDiscretization_.Rd3[cell] + (1.0 - b5) * Rd3_0[cell];
    }
    for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
        double dt = dt_cells[cell];
        spatialDiscretization_.W0[cell] = W0_0[cell] - a5 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc0[cell] - Rd0_0[cell]);
        spatialDiscretization_.W1[cell] = W1_0[cell] - a5 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc1[cell] - Rd1_0[cell]);
        spatialDiscretization_.W2[cell] = W2_0[cell] - a5 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc2[cell] - Rd2_0[cell]);
        spatialDiscretization_.W3[cell] = W3_0[cell] - a5 * dt / spatialDiscretization_.mesh_.volume[cell] * (spatialDiscretization_.Rc3[cell] - Rd3_0[cell]);
    }
}

void TemporalDiscretization::solve() {
    spatialDiscretization_.run_odd();
    for (int it = 0; it < it_max_; ++it) {
        compute_dt();
        RungeKuttaStep();
        auto [cl, cd, cm] = spatialDiscretization_.compute_aerodynamics_coefficients();
        std::vector<double> R0(spatialDiscretization_.Rc0.size());
        std::vector<double> R1(spatialDiscretization_.Rc1.size());
        std::vector<double> R2(spatialDiscretization_.Rc2.size());
        std::vector<double> R3(spatialDiscretization_.Rc3.size());
        for (auto cell : spatialDiscretization_.mesh_.fluidCells) {
            R0[cell] = spatialDiscretization_.Rc0[cell] - spatialDiscretization_.Rd0[cell];
            R1[cell] = spatialDiscretization_.Rc1[cell] - spatialDiscretization_.Rd1[cell];
            R2[cell] = spatialDiscretization_.Rc2[cell] - spatialDiscretization_.Rd2[cell];
            R3[cell] = spatialDiscretization_.Rc3[cell] - spatialDiscretization_.Rd3[cell];
        }

        std::vector<double> R0_wall;
        for (size_t i=0;i<spatialDiscretization_.mesh_.ghostCells.size();++i) {
            for (auto cell : spatialDiscretization_.mesh_.adjacentCells[i]) {
                R0_wall.push_back(R0[cell]);
            }
        }

        std::vector<double> R0_farfield;
        for (auto face : spatialDiscretization_.mesh_.farfieldFaces) {
            int leftCell = spatialDiscretization_.mesh_.faces[face].leftCell;
            int rightCell = spatialDiscretization_.mesh_.faces[face].rightCell;
            int leftCellType = spatialDiscretization_.mesh_.cell_types[leftCell];
            int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
            R0_farfield.push_back(R0[fluidCell]);
        }

        double res_r0 = l2Norm(R0);
        double res_r1 = l2Norm(R1);
        double res_r2 = l2Norm(R2);
        double res_r3 = l2Norm(R3);
        double res_r0_wall = l2Norm(R0_wall);
        double res_r0_farfield = l2Norm(R0_farfield);

        std::cout << "Iteration " << it+1 << ": "
                  << " R0 = " << res_r0 << ", R1 = " << res_r1 << ", R2 = " << res_r2 << ", R3 = " << res_r3
                  << ", cl = " << cl << ", cd = " << cd << ", cm = " << cm << ", R0_wall = " << res_r0_wall << ", R0_farfield = " << res_r0_farfield << "\n";
        // std::cout << "Rc0[135] = " << spatialDiscretization_.Rc0[135] << " Rc0[18495] = " << spatialDiscretization_.Rc0[18495] << "\n";
        // std::cout << "Rd0[135] = " << spatialDiscretization_.Rd0[135] << " Rd0[18495] = " << spatialDiscretization_.Rd0[18495] << "\n";

        if ((it + 1) % 250 == 0) {
            // check if input_out_filename already exists; if so, append a number to avoid overwriting
            std::string input_out_filename = "output/output.cgns";
            std::ifstream infile_check(input_out_filename);
            if (infile_check.good()) {
                infile_check.close();
                std::string base_name, extension;
                auto dot_pos = input_out_filename.find_last_of('.');
                if (dot_pos != std::string::npos) {
                    base_name = input_out_filename.substr(0, dot_pos);
                    extension = input_out_filename.substr(dot_pos);
                } else {
                    base_name = input_out_filename;
                    extension = "";
                }
                int file_index = 1;;
                std::string new_filename;
                do {
                    new_filename = base_name + "_" + std::to_string(file_index) + extension;
                    ++file_index;
                } while (std::ifstream(new_filename).good());
                std::cout << "Output file " << input_out_filename << " already exists. Using new filename: " << new_filename << std::endl;
                input_out_filename = new_filename;
            }

            if (spatialDiscretization_.mesh_.writeToCGNSWithCellData(input_out_filename, spatialDiscretization_)) {
                std::cerr << "Failed to write CGNS with cell data." << std::endl;
            } else {
                std::cout << "Wrote CGNS with cell data: " << input_out_filename << std::endl;
            }
        }

        if (res_r0 < 1e-12 || std::isnan(res_r0)) {
            break;
        }

    }
}