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
    std::string output_filename_;
    double dx_;
    double alpha_;
    double theta_;
    double Mach_;
    double cInf_;
    double gamma_ = 1.4;
    double rhoInf_;
    double pInf_;
    double uInf_;
    double eInf_;

    parameters(unsigned int N, double outletBoundaryCondition, double CFL, double dx, const std::string& output_filename, double theta);

    void print() const;
};

#endif // PARAMETERS_H