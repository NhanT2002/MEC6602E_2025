#include "parameters.h"
#include <iostream>

parameters::parameters(int N, double c, double CFL, double t_final, double dx, double dt, const std::string& output_filename)
    : N_(N), c_(c), CFL_(CFL), t_final_(t_final), output_filename_(output_filename), dx_(dx), dt_(dt) {
        alpha_ = c_ * dt_ / dx_;
    }

void parameters::print() const {
    std::cout << "N: " << N_ << std::endl;
    std::cout << "c: " << c_ << std::endl;
    std::cout << "CFL: " << CFL_ << std::endl;
    std::cout << "t_final: " << t_final_ << std::endl;
    std::cout << "dx: " << dx_ << std::endl;
    std::cout << "dt: " << dt_ << std::endl;
}