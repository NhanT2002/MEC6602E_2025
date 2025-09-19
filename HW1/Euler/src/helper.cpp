#include "helper.h"
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cmath>

void printVector(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << ": ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

double nozzleArea(double x) {
    return 1.398 + 0.347 * std::tanh(0.8*x - 4.0);
}

double dA_dx(double x) {
    double sech2 = 1.0 / (std::cosh(0.8*x - 4.0) * std::cosh(0.8*x - 4.0));
    return 0.347 * 0.8 * sech2;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> initializeQ(parameters& params) {
    std::vector<double> x(params.N_+2); // Including ghost cells
    std::vector<double> A(params.N_+2);
    std::vector<double> Q1(params.N_+2);
    std::vector<double> Q2(params.N_+2);
    std::vector<double> Q3(params.N_+2);

    for (unsigned int i = 1; i < params.N_+1; ++i) {
        x[i] = 0 + (i-1) * params.dx_;
        A[i] = nozzleArea(x[i]);
        Q1[i] = params.rhoInf_ * A[i];
        Q2[i] = params.rhoInf_ * params.uInf_ * A[i];
        Q3[i] = params.eInf_ * A[i];
    }
    return {x, A, Q1, Q2, Q3};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> initializeE(parameters& params, const std::vector<double>& A, const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3) {
    std::vector<double> E1(params.N_+2);
    std::vector<double> E2(params.N_+2);
    std::vector<double> E3(params.N_+2);
    for (unsigned int i = 1; i < params.N_+1; ++i) {
        auto [rho, u, e, p] = primitiveVariables(Q1[i], Q2[i], Q3[i], A[i], params.gamma_);
        E1[i] = rho*u*A[i];
        E2[i] = (rho*u*u + p)*A[i];
        E3[i] = u*(e + p)*A[i];
    }
    return {E1, E2, E3};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> initializeS(parameters& params, const std::vector<double>& x, const std::vector<double>& A, const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3) {
    std::vector<double> S1(params.N_+2);
    std::vector<double> S2(params.N_+2);
    std::vector<double> S3(params.N_+2);
    for (unsigned int i = 1; i < params.N_+1; ++i) {
        auto [rho, u, e, p] = primitiveVariables(Q1[i], Q2[i], Q3[i], A[i], params.gamma_);
        S1[i] = 0.0;
        S2[i] = p * (dA_dx(x[i]) );
        S3[i] = 0.0;
    }
    return {S1, S2, S3};
}

std::tuple<double, double, double, double> primitiveVariables(double Q1, double Q2, double Q3, double A, double gamma) {
    double rho = Q1 / A;
    double u = Q2 / Q1;
    double e = Q3 / A;
    double p = (gamma - 1) * (e - 0.5 * rho * u * u);
    return {rho, u, e, p};
}

void updateBoundaryConditions(parameters& params, const std::vector<double>& x, const std::vector<double>& A,
    std::vector<double>& Q1, std::vector<double>& Q2, std::vector<double>& Q3,
    std::vector<double>& E1, std::vector<double>& E2, std::vector<double>& E3, 
    std::vector<double>& S1, std::vector<double>& S2, std::vector<double>& S3) {
    // Inlet (i=0): Supersonic inflow, all variables specified
    Q1[0] = params.rhoInf_ * nozzleArea(0);
    Q2[0] = params.rhoInf_ * params.uInf_ * nozzleArea(0);
    Q3[0] = params.eInf_ * nozzleArea(0);

    // Outlet (i=N-1)
    if (params.outletBoundaryCondition_ == 1) {
        // Supersonic outflow: all variables extrapolated
        Q1[params.N_+1] = Q1[params.N_];
        Q2[params.N_+1] = Q2[params.N_];
        Q3[params.N_+1] = Q3[params.N_];
    } else if (params.outletBoundaryCondition_ == 2) {
        // Subsonic outflow with back pressure: pressure specified, others extrapolated
        auto [rho, u, e, p] = primitiveVariables(Q1[params.N_-2], Q2[params.N_-2], Q3[params.N_-2], nozzleArea((params.N_-2) * params.dx_), params.gamma_);
        p = params.pInf_; // Set back pressure
        rho = rho; // Extrapolate density
        u = u; // Extrapolate velocity
        e = p / (params.gamma_ - 1) + 0.5 * rho * u * u; // Recalculate energy

        Q1[params.N_ - 1] = rho * nozzleArea((params.N_ - 1) * params.dx_);
        Q2[params.N_ - 1] = rho * u * nozzleArea((params.N_ - 1) * params.dx_);
        Q3[params.N_ - 1] = e * nozzleArea((params.N_ - 1) * params.dx_);
    }

    // Recompute E and S after updating Q
    E1[0] = params.rhoInf_ * params.uInf_ * nozzleArea(0);
    E2[0] = (params.rhoInf_ * params.uInf_ * params.uInf_ + params.pInf_) * nozzleArea(0);
    E3[0] = params.uInf_ * (params.eInf_ + params.pInf_) * nozzleArea(0);
    S2[0] = params.pInf_ * (dA_dx(x[1]) );

    E1[params.N_+1] = E1[params.N_];
    E2[params.N_+1] = E2[params.N_];
    E3[params.N_+1] = E3[params.N_];
    S2[params.N_+1] = S2[params.N_];
}

double l2Norm(const std::vector<double>& u_np1, const std::vector<double>& u_n) {
    double sum = 0.0;
    for (size_t i = 1; i < u_np1.size()-1; ++i) {
        sum += (u_np1[i] - u_n[i]) * (u_np1[i] - u_n[i]);
    }
    return std::sqrt(sum);
}
