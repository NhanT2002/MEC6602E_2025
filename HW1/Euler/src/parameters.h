#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <tuple>
#include <string>

class parameters {
    public:
    unsigned int N_;
    double outletBoundaryCondition_;
    double CFL_;
    double epsilon_e_;
    double epsilon_i_;
    std::string output_filename_;
    double dx_;
    double alpha_;
    double theta_;
    double Mach_;
    double pBackRatio_ = 1.9;
    double cInf_;
    double gamma_ = 1.4;
    double rhoInf_;
    double pInf_;
    double uInf_;
    double eInf_;

    parameters(unsigned int N, double outletBoundaryCondition, double CFL, double epsilon_e, double dx, const std::string& output_filename, double theta);

    void print() const;
};

#endif // PARAMETERS_H