#include "TemporalDiscretization.h"
#include "mesh.h"
#include "helper.h"
#include <iostream>
#include <vector>

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
    spatialDiscretization_.mesh_.writeToCGNSWithCellData("out_rk1.cgns", spatialDiscretization_);

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

        double res_r0 = l2Norm(R0);
        double res_r1 = l2Norm(R1);
        double res_r2 = l2Norm(R2);
        double res_r3 = l2Norm(R3);

        std::cout << "Iteration " << it+1 << ": "
                  << " res_rho = " << res_r0
                  << ", res_rho_u = " << res_r1
                  << ", res_rho_v = " << res_r2
                  << ", res_rho_E = " << res_r3 << std::endl;

    }
}