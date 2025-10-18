#ifndef HELPER_H
#define HELPER_H
#include <vector>
#include <string>
#include <tuple>

void printVector(const std::vector<double>& vec, const std::string& name);

void printVector(const std::vector<int>& vec, const std::string& name);

double l2Norm(const std::vector<double>& u_np1, const std::vector<double>& u_n);

#endif // HELPER_H