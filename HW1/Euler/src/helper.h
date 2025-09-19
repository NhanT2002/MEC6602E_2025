#ifndef HELPER_H
#define HELPER_H
#include <vector>
#include <string>
#include <tuple>
#include "parameters.h"

void printVector(const std::vector<double>& vec, const std::string& name);

double nozzleArea(double x);

double dA_dx(double x);

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> initializeQ(parameters& params);

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> initializeE(parameters& params, const std::vector<double>& A, const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3);

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> initializeS(parameters& params, const std::vector<double>& x, const std::vector<double>& A, const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3);

std::tuple<double, double, double, double> primitiveVariables(double Q1, double Q2, double Q3, double A, double gamma);

void updateBoundaryConditions(parameters& params, const std::vector<double>& x, const std::vector<double>& A,
    std::vector<double>& Q1, std::vector<double>& Q2, std::vector<double>& Q3,
    std::vector<double>& E1, std::vector<double>& E2, std::vector<double>& E3, 
    std::vector<double>& S1, std::vector<double>& S2, std::vector<double>& S3);

double l2Norm(const std::vector<double>& u_np1, const std::vector<double>& u_n);

#endif // HELPER_H