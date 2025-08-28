#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <tuple>
#include "parameters.h"


void explicitBackward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void explicitForward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

#endif // SOLVER_H