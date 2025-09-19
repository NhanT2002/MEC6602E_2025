#ifndef WRITE_SOLUTION_H
#define WRITE_SOLUTION_H

#include <vector>
#include <string>

void writeSolution(const std::vector<double>& x, 
                   const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3, 
                   const std::vector<double>& E1, const std::vector<double>& E2, const std::vector<double>& E3, 
                   const std::vector<double>& S1, const std::vector<double>& S2, const std::vector<double>& S3, 
                   const std::vector<double>& rho, const std::vector<double>& u, const std::vector<double>& p, 
                   const std::vector<double>& e, const std::vector<double>& mach,
                   const std::string& filename);

void writeConvergenceHistory(const std::vector<double>& res1, const std::vector<double>& res2, const std::vector<double>& res3, const std::string& filename);

#endif // WRITE_SOLUTION_H