#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <tuple>
#include "parameters.h"


void macCormack(parameters& params, const std::vector<double>& x, const std::vector<double>& A,
                const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3,
                std::vector<double>& Q1_np1, std::vector<double>& Q2_np1, std::vector<double>& Q3_np1,
                const std::vector<double>& E1, const std::vector<double>& E2, const std::vector<double>& E3,
                const std::vector<double>& S1, const std::vector<double>& S2, const std::vector<double>& S3);

#endif // SOLVER_H