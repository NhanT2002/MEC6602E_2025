#include "solver.h"
#include <omp.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>


void explicitBackward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 1; i < u.size(); ++i) {
        u_np1[i] = u[i] - params.alpha_ * (u[i] - u[i - 1]);
    }
    u_np1[0] = u[0] - params.alpha_ * (u[0] - u[u.size() - 1]);
}

void explicitForward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 0; i < u.size() - 1; ++i) {
        u_np1[i] = u[i] - params.alpha_ * (u[i + 1] - u[i]);
    }
    u_np1[u.size() - 1] = u[u.size() - 1] - params.alpha_ * (u[0] - u[u.size() - 1]);
}

void forwardTimeCenteredSpace(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 1; i < u.size() - 1; ++i) {
        u_np1[i] = u[i] - params.alpha_ / 2.0 * (u[i + 1] - u[i - 1]);
    }
    u_np1[0] = u[0] - params.alpha_ / 2.0 * (u[1] - u[u.size() - 1]);
    u_np1[u.size() - 1] = u[u.size() - 1] - params.alpha_ / 2.0 * (u[0] - u[u.size() - 2]);
}

void leapFrog(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1, std::vector<double>& u_nm1) {
    for (long unsigned int i = 1; i < u.size() - 1; ++i) {
        u_np1[i] = u_nm1[i] - params.alpha_ * (u[i + 1] - u[i - 1]);
    }
    u_np1[0] = u_nm1[0] - params.alpha_ * (u[1] - u[u.size() - 1]);
    u_np1[u.size() - 1] = u_nm1[u.size() - 1] - params.alpha_ * (u[0] - u[u.size() - 2]);
}

void laxWendroff(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 1; i < u.size() - 1; ++i) {
        u_np1[i] = u[i] - params.alpha_ / 2.0 * (u[i + 1] - u[i - 1])
                        + (params.alpha_ * params.alpha_) / 2.0 * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
    }
    u_np1[0] = u[0] - params.alpha_ / 2.0 * (u[1] - u[u.size() - 1])
                    + (params.alpha_ * params.alpha_) / 2.0 * (u[1] - 2.0 * u[0] + u[u.size() - 1]);
    u_np1[u.size() - 1] = u[u.size() - 1] - params.alpha_ / 2.0 * (u[0] - u[u.size() - 2])
                    + (params.alpha_ * params.alpha_) / 2.0 * (u[0] - 2.0 * u[u.size() - 1] + u[u.size() - 2]);
    
}

void lax(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 1; i < u.size() - 1; ++i) {
        u_np1[i] = (u[i + 1] + u[i - 1]) / 2.0 - params.alpha_ / 2.0 * (u[i + 1] - u[i - 1]);
    }
    u_np1[0] = (u[1] + u[u.size() - 1]) / 2.0 - params.alpha_ / 2.0 * (u[1] - u[u.size() - 1]);
    u_np1[u.size() - 1] = (u[0] + u[u.size() - 2]) / 2.0 - params.alpha_ / 2.0 * (u[0] - u[u.size() - 2]);
}

void hybridExplicitImplicit(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(params.N_, params.N_);
    Eigen::VectorXd b(params.N_);
    for (unsigned int i = 0; i < params.N_; ++i) {
        A(i, i) = 1.0;
        A(i, (i + 1) % params.N_) = params.alpha_ * params.theta_ / 2.0;
        A(i, (i - 1 + params.N_) % params.N_) = params.alpha_ * params.theta_ / -2.0;
        b(i) = u[i] - params.alpha_ * (1 - params.theta_) / 2.0 * (u[(i + 1) % params.N_] - u[(i - 1 + params.N_) % params.N_]);
    }
    Eigen::VectorXd u_np1_eigen = A.lu().solve(b);
    for (unsigned int i = 0; i < params.N_; ++i) {
        u_np1[i] = u_np1_eigen(i);
    }
}

void rungeKutta4(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    std::vector<double> k1(params.N_), k2(params.N_), k3(params.N_), k4(params.N_);
    std::vector<double> u_temp(params.N_);

    for (unsigned int i = 0; i < params.N_; ++i) {
        k1[i] = -params.c_ * (u[(i + 1) % params.N_] - u[(i - 1 + params.N_) % params.N_]) / (2.0 * params.dx_);
        u_temp[i] = u[i] + 0.5 * params.dt_ * k1[i];
    } for (unsigned int i = 0; i < params.N_; ++i) {
        k2[i] = -params.c_ * (u_temp[(i + 1) % params.N_] - u_temp[(i - 1 + params.N_) % params.N_]) / (2.0 * params.dx_);
        u_temp[i] = u[i] + 0.5 * params.dt_ * k2[i];
    } for (unsigned int i = 0; i < params.N_; ++i) {
        k3[i] = -params.c_ * (u_temp[(i + 1) % params.N_] - u_temp[(i - 1 + params.N_) % params.N_]) / (2.0 * params.dx_);
        u_temp[i] = u[i] + params.dt_ * k3[i];
    } for (unsigned int i = 0; i < params.N_; ++i) {
        k4[i] = -params.c_ * (u_temp[(i + 1) % params.N_] - u_temp[(i - 1 + params.N_) % params.N_]) / (2.0 * params.dx_);
    }

    for (unsigned int i = 0; i < params.N_; ++i) {
        u_np1[i] = u[i] + (params.dt_ / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}