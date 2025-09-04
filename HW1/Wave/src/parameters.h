#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <tuple>
#include <string>

class parameters {
    public:
    unsigned int N_;
    double c_;
    double CFL_;
    double t_final_;
    std::string output_filename_;
    double dx_;
    double dt_;
    double alpha_;
    double theta_;

    parameters(unsigned int N, double c, double CFL, double t_final, double dx, double dt, const std::string& output_filename, double theta);

    void print() const;
};

#endif // PARAMETERS_H