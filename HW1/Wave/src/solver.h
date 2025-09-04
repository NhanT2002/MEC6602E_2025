#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <tuple>
#include "parameters.h"


void explicitBackward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void explicitForward(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void forwardTimeCenteredSpace(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void leapFrog(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1, std::vector<double>& u_nm1);

void laxWendroff(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void lax(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void hybridExplicitImplicit(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

void rungeKutta4(parameters& params, const std::vector<double>& u, std::vector<double>& u_np1);

#endif // SOLVER_H