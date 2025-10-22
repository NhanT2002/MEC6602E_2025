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

void printVector(const std::vector<int>& vec, const std::string& name) {
    std::cout << name << ": ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

double l2Norm(const std::vector<double>& u) {
    double sum = 0.0;
    for (size_t i = 1; i < u.size()-1; ++i) {
        sum += u[i] * u[i];
    }
    return std::sqrt(sum);
}

double pressure(double gamma, double rho, double u, double v, double E) {
    return (gamma - 1.0) * rho * (E - 0.5 * (u * u + v * v));
}

double mach(double gamma, double rho, double p, double u, double v) {
    double c = std::sqrt(gamma * p / rho);
    double M = std::sqrt(u * u + v * v) / c;
    return M;
}
