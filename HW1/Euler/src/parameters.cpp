#include "parameters.h"
#include <iostream>
#include <cmath>

parameters::parameters(unsigned int N, double outletBoundaryCondition, double CFL, double dx, const std::string& output_filename, double theta)
    : N_(N), outletBoundaryCondition_(outletBoundaryCondition), CFL_(CFL), output_filename_(output_filename), dx_(dx), theta_(theta) {
        Mach_ = 1.25;
        rhoInf_ = 1.0;
        pInf_ = 1.0;
        cInf_ = std::sqrt(gamma_ * pInf_ / rhoInf_);
        uInf_ = Mach_ * cInf_;
        eInf_ = pInf_ / (gamma_ - 1) + rhoInf_ * uInf_ * uInf_ / 2.0;
    }

void parameters::print() const {
    std::cout << "N: " << N_ << std::endl;
    std::cout << "outletBoundaryCondition: " << outletBoundaryCondition_ << std::endl;
    std::cout << "CFL: " << CFL_ << std::endl;
    std::cout << "dx: " << dx_ << std::endl;
    std::cout << "theta: " << theta_ << std::endl;
    std::cout << "output_filename: " << output_filename_ << std::endl;
    std::cout << "Mach: " << Mach_ << std::endl;
    std::cout << "rhoInf: " << rhoInf_ << std::endl;
    std::cout << "pInf: " << pInf_ << std::endl;
    std::cout << "cInf: " << cInf_ << std::endl;
    std::cout << "uInf: " << uInf_ << std::endl;
    std::cout << "eInf: " << eInf_ << std::endl;
}