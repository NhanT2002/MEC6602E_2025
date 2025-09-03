#include "solver.h"
#include <omp.h>
#include <vector>
#include <iostream>

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