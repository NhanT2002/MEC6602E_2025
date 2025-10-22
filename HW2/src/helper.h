#ifndef HELPER_H
#define HELPER_H
#include <vector>
#include <string>
#include <tuple>

void printVector(const std::vector<double>& vec, const std::string& name);

void printVector(const std::vector<int>& vec, const std::string& name);

double l2Norm(const std::vector<double>& u);

double pressure(double gamma, double rho, double u, double v, double E);

double mach(double gamma, double rho, double p, double u, double v);

#endif // HELPER_H