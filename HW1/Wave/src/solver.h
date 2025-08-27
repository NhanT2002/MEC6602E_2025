#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <tuple>


void explicitBackward(const std::vector<double>& x, std::vector<double>& u);
void explicitForward(const std::vector<double>& x, std::vector<double>& u);

#endif // SOLVER_H