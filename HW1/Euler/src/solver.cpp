#include "solver.h"
#include "helper.h"
#include <omp.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>


void macCormack(parameters& params, const std::vector<double>& x, const std::vector<double>& A,
                const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3,
                std::vector<double>& Q1_np1, std::vector<double>& Q2_np1, std::vector<double>& Q3_np1,
                const std::vector<double>& E1, const std::vector<double>& E2, const std::vector<double>& E3,
                const std::vector<double>& S1, const std::vector<double>& S2, const std::vector<double>& S3) {
    
    std::vector<double> dt(params.N_+2);
    for (unsigned int i = 2; i < params.N_; ++i) {
        auto [rho, u, e, p] = primitiveVariables(Q1[i], Q2[i], Q3[i], A[i], params.gamma_);
        double c = std::sqrt(params.gamma_ * p / rho);
        dt[i] = params.CFL_ * params.dx_ / (std::abs(u) + c);
    }
    // Predictor step
    std::vector<double> Q1_pred(params.N_+2);
    std::vector<double> Q2_pred(params.N_+2);
    std::vector<double> Q3_pred(params.N_+2);
    for (unsigned int i = 2; i < params.N_; ++i) {
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
    for (unsigned int i = 2; i < params.N_; ++i) {
        Q1_corr[i] = Q1[i] - dt[i] / params.dx_ * (E1_pred[i+1] - E1_pred[i]) + dt[i] * S1_pred[i];
        Q2_corr[i] = Q2[i] - dt[i] / params.dx_ * (E2_pred[i+1] - E2_pred[i]) + dt[i] * S2_pred[i];
        Q3_corr[i] = Q3[i] - dt[i] / params.dx_ * (E3_pred[i+1] - E3_pred[i]) + dt[i] * S3_pred[i];
    }

    // Update to next time step
    for (unsigned int i = 2; i < params.N_; ++i) {
        Q1_np1[i] = 0.5 * (Q1_pred[i] + Q1_corr[i]);
        Q2_np1[i] = 0.5 * (Q2_pred[i] + Q2_corr[i]);
        Q3_np1[i] = 0.5 * (Q3_pred[i] + Q3_corr[i]);
    }
    Q1_np1[0] = Q1[0];
    Q2_np1[0] = Q2[0];
    Q3_np1[0] = Q3[0];
    Q1_np1[params.N_] = Q1[params.N_];
    Q2_np1[params.N_] = Q2[params.N_];
    Q3_np1[params.N_] = Q3[params.N_];

}

void beamWarming(Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>>& solver,
                parameters& params, const std::vector<double>& x, const std::vector<double>& A,
                const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3,
                std::vector<double>& Q1_np1, std::vector<double>& Q2_np1, std::vector<double>& Q3_np1,
                const std::vector<double>& E1, const std::vector<double>& E2, const std::vector<double>& E3,
                const std::vector<double>& S1, const std::vector<double>& S2, const std::vector<double>& S3) {
    std::vector<double> dt(params.N_+2);
    for (unsigned int i = 2; i < params.N_; ++i) {
        auto [rho, u, e, p] = primitiveVariables(Q1[i], Q2[i], Q3[i], A[i], params.gamma_);
        double c = std::sqrt(params.gamma_ * p / rho);
        dt[i] = params.CFL_ * params.dx_ / (std::abs(u) + c);
    }
    
    std::vector<Eigen::MatrixXd> Ai(params.N_+2, Eigen::MatrixXd(3,3));
    computeAi(Ai, Q1, Q2, Q3, params.gamma_, params.N_);
    std::vector<Eigen::MatrixXd> Bi(params.N_+2, Eigen::MatrixXd(3,3));
    computeBi(Bi, Q1, Q2, Q3, x, A, params.gamma_, params.N_);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3,3);
    Eigen::MatrixXd LHS = Eigen::MatrixXd::Zero(params.N_*3, params.N_*3);
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(params.N_*3);
    // Assemble the linear system
    for (unsigned int i = 1; i < params.N_-1; ++i) {
        const int ii = i+1; // Offset by 1 due to ghost cells
        const double eps_e = params.epsilon_e_*0.125*dt[ii]*A[ii];
        const double eps_i = 2.5*eps_e;
        // std::cout << "i: " << i << ", dt: " << dt[ii] << ", eps_e: " << eps_e << ", eps_i: " << eps_i << std::endl;

        // Fill in the LHS matrix
        LHS.block(3*i, 3*i, 3, 3) = I - dt[ii]*Bi[ii] - eps_i*(-2)/(params.dx_*params.dx_)*I;
        LHS.block(3*i, 3*(i-1), 3, 3) = -dt[ii]/(2*params.dx_)*Ai[ii-1] - eps_i*(1)/(params.dx_*params.dx_)*I;
        LHS.block(3*i, 3*(i+1), 3, 3) = dt[ii]/(2*params.dx_)*Ai[ii+1] - eps_i*(1)/(params.dx_*params.dx_)*I;

        RHS(3*i) = -dt[ii]/(2*params.dx_)*(E1[ii+1] - E1[ii-1]) + dt[ii]*S1[ii] + eps_e*(Q1[ii+1] - 2*Q1[ii] + Q1[ii-1])/(params.dx_*params.dx_);
        RHS(3*i + 1) = -dt[ii]/(2*params.dx_)*(E2[ii+1] - E2[ii-1]) + dt[ii]*S2[ii] + eps_e*(Q2[ii+1] - 2*Q2[ii] + Q2[ii-1])/(params.dx_*params.dx_);
        RHS(3*i + 2) = -dt[ii]/(2*params.dx_)*(E3[ii+1] - E3[ii-1]) + dt[ii]*S3[ii] + eps_e*(Q3[ii+1] - 2*Q3[ii] + Q3[ii-1])/(params.dx_*params.dx_);
        
    }

    // Apply boundary conditions directly in the linear system
    // Inlet (i=0, corresponds to grid point 1)
    LHS.block(0, 0, 3, 3) = I;
    RHS(0) = 0.0; // Q1[1] = rhoInf * A[1]
    RHS(1) = 0.0; // Q2[1] = rhoInf * uInf * A[1]
    RHS(2) = 0.0; // Q3[1] = eInf * A[1]

    // Outlet (i=N-1, corresponds to grid point N)
    LHS.block(3*(params.N_-1), 3*(params.N_-1), 3, 3) = I;
    RHS(3*(params.N_-1)) = 0.0; // Q1[N] = Q1[N]
    RHS(3*(params.N_-1) + 1) = 0.0; // Q2[N] = Q2[N]
    RHS(3*(params.N_-1) + 2) = 0.0; // Q3[N] = Q3[N]

    // std::cout << "LHS matrix:" << std::endl;
    // std::cout << LHS << std::endl;
    // std::cout << "RHS vector:" << std::endl;
    // std::cout << RHS << std::endl;

    // Solve the linear system
    Eigen::SparseMatrix<double> LHS_sparse = LHS.sparseView();
    solver.compute(LHS_sparse);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }
    Eigen::VectorXd solution = solver.solve(RHS);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Solving failed" << std::endl;
        return;
    }
    // Update the solution vectors
    for (unsigned int i = 0; i < params.N_; ++i) {
        const int ii = i+1; // Offset by 1 due to ghost cells
        Q1_np1[ii] = Q1[ii] + solution(3*i);
        Q2_np1[ii] = Q2[ii] + solution(3*i + 1);
        Q3_np1[ii] = Q3[ii] + solution(3*i + 2);
    }
}

void computeAi(std::vector<Eigen::MatrixXd>& Ai, const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3, double gamma, int N) {
    for (int i = 1; i < N+1; ++i) {
        
        Ai[i](0,0) = 0.0;
        Ai[i](0,1) = 1.0;
        Ai[i](0,2) = 0.0;
        Ai[i](1,0) = 0.5*(gamma - 3) * Q2[i] * Q2[i] / (Q1[i] * Q1[i]);
        Ai[i](1,1) = (3 - gamma) * Q2[i] / Q1[i];
        Ai[i](1,2) = gamma - 1;
        Ai[i](2,0) = (gamma-1)*Q2[i]*Q2[i]*Q2[i]/(Q1[i]*Q1[i]*Q1[i]) - gamma*Q2[i]*Q3[i]/(Q1[i]*Q1[i]);
        Ai[i](2,1) = gamma*Q3[i]/Q1[i] - 1.5*(gamma-1)*Q2[i]*Q2[i]/(Q1[i]*Q1[i]);
        Ai[i](2,2) = gamma*Q2[i]/Q1[i];
    }
}

void computeBi(std::vector<Eigen::MatrixXd>& Bi, const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3, const std::vector<double>& x, const std::vector<double>& A, double gamma, int N) {
    for (int i = 1; i < N+1; ++i) {
        Bi[i](0,0) = 0.0;
        Bi[i](0,1) = 0.0;
        Bi[i](0,2) = 0.0;
        Bi[i](1,0) = (gamma-1)*dA_dx(x[i])/(2*A[i]) * Q2[i]*Q2[i]/(Q1[i]*Q1[i]);
        Bi[i](1,1) = (1-gamma)*dA_dx(x[i])/(A[i]) * Q2[i]/Q1[i];
        Bi[i](1,2) = (gamma-1)*dA_dx(x[i])/(A[i]);
        Bi[i](2,0) = 0.0;
        Bi[i](2,1) = 0.0;
        Bi[i](2,2) = 0.0;
    }
}