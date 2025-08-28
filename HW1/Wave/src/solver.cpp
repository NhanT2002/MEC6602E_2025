#include "solver.h"
#include <omp.h>
#include <vector>
#include <iostream>

void explicitBackward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 1; i < u.size(); ++i) {
        u_np1[i] = u[i] - params.alpha_ * (u[i] - u[i-1]);
        // std::cout << "i: " << i << ", u[i]: " << u[i] << ", u[i-1]: " << u[i-1] << ", u_np1[i]: " << u_np1[i] << std::endl;
    }
}

void explicitForward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1) {
    for (long unsigned int i = 0; i < u.size() - 1; ++i) {
        u_np1[i] = u[i] - params.alpha_ * (u[i + 1] - u[i]);
    }
}
