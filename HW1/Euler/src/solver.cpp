#include "solver.h"
#include "helper.h"
#include <omp.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>


void macCormack(parameters& params, const std::vector<double>& x, const std::vector<double>& A,
                const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3,
                std::vector<double>& Q1_np1, std::vector<double>& Q2_np1, std::vector<double>& Q3_np1,
                const std::vector<double>& E1, const std::vector<double>& E2, const std::vector<double>& E3,
                const std::vector<double>& S1, const std::vector<double>& S2, const std::vector<double>& S3) {
    
    std::vector<double> dt(params.N_+2);
    for (unsigned int i = 1; i < params.N_+1; ++i) {
        auto [rho, u, e, p] = primitiveVariables(Q1[i], Q2[i], Q3[i], A[i], params.gamma_);
        double c = std::sqrt(params.gamma_ * p / rho);
        dt[i] = params.CFL_ * params.dx_ / (std::abs(u) + c);
    }
    // Predictor step
    std::vector<double> Q1_pred(params.N_+2);
    std::vector<double> Q2_pred(params.N_+2);
    std::vector<double> Q3_pred(params.N_+2);
    for (unsigned int i = 1; i < params.N_+1; ++i) {
        Q1_pred[i] = Q1[i] - dt[i] / params.dx_ * (E1[i] - E1[i-1]) + dt[i] * S1[i];
        Q2_pred[i] = Q2[i] - dt[i] / params.dx_ * (E2[i] - E2[i-1]) + dt[i] * S2[i];
        Q3_pred[i] = Q3[i] - dt[i] / params.dx_ * (E3[i] - E3[i-1]) + dt[i] * S3[i];
    }

    auto [E1_pred, E2_pred, E3_pred] = initializeE(params, A, Q1_pred, Q2_pred, Q3_pred);
    auto [S1_pred, S2_pred, S3_pred] = initializeS(params, x, A, Q1_pred, Q2_pred, Q3_pred);

    updateBoundaryConditions(params, x, A, Q1_pred, Q2_pred, Q3_pred, E1_pred, E2_pred, E3_pred, S1_pred, S2_pred, S3_pred);


    // printVector(dt, "dt");
    // printVector(Q1_pred, "Q1_pred");
    // printVector(Q2_pred, "Q2_pred");
    // printVector(Q3_pred, "Q3_pred");
    // printVector(E1_pred, "E1_pred");
    // printVector(E2_pred, "E2_pred");
    // printVector(E3_pred, "E3_pred");
    // printVector(S1_pred, "S1_pred");
    // printVector(S2_pred, "S2_pred");
    // printVector(S3_pred, "S3_pred");

    // Corrector step
    std::vector<double> Q1_corr(params.N_+2);
    std::vector<double> Q2_corr(params.N_+2);
    std::vector<double> Q3_corr(params.N_+2);
    for (unsigned int i = 1; i < params.N_+1; ++i) {
        Q1_corr[i] = Q1[i] - dt[i] / params.dx_ * (E1_pred[i+1] - E1_pred[i]) + dt[i] * S1_pred[i];
        Q2_corr[i] = Q2[i] - dt[i] / params.dx_ * (E2_pred[i+1] - E2_pred[i]) + dt[i] * S2_pred[i];
        Q3_corr[i] = Q3[i] - dt[i] / params.dx_ * (E3_pred[i+1] - E3_pred[i]) + dt[i] * S3_pred[i];
    }

    // Update to next time step
    for (unsigned int i = 1; i < params.N_+1; ++i) {
        Q1_np1[i] = 0.5 * (Q1_pred[i] + Q1_corr[i]);
        Q2_np1[i] = 0.5 * (Q2_pred[i] + Q2_corr[i]);
        Q3_np1[i] = 0.5 * (Q3_pred[i] + Q3_corr[i]);
    }

}