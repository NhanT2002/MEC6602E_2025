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

double l2Norm(const std::vector<double>& u_np1, const std::vector<double>& u_n) {
    double sum = 0.0;
    for (size_t i = 1; i < u_np1.size()-1; ++i) {
        sum += (u_np1[i] - u_n[i]) * (u_np1[i] - u_n[i]);
    }
    return std::sqrt(sum);
}
