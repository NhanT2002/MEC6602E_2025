#include "writeSolution.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#include "helper.h"

void writeSolution(const std::vector<double>& x, 
                   const std::vector<double>& Q1, const std::vector<double>& Q2, const std::vector<double>& Q3, 
                   const std::vector<double>& E1, const std::vector<double>& E2, const std::vector<double>& E3, 
                   const std::vector<double>& S1, const std::vector<double>& S2, const std::vector<double>& S3, 
                   const std::vector<double>& rho, const std::vector<double>& u, const std::vector<double>& p, 
                   const std::vector<double>& e, const std::vector<double>& mach,
                   const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file << "x,Q1,Q2,Q3,E1,E2,E3,S1,S2,S3,rho,u,p,e,mach" << std::endl;
    file << std::scientific << std::setprecision(12);
    for (size_t i = 1; i < x.size() - 1; ++i) { // Skip ghost cells
        file << x[i] << "," << Q1[i] << "," << Q2[i] << "," << Q3[i] 
        << "," << E1[i] << "," << E2[i] << "," << E3[i] << "," << S1[i] << "," << S2[i] << "," << S3[i]
        << "," << rho[i] << "," << u[i] << "," << p[i] << "," << e[i] << "," << mach[i] << std::endl;
    }

    file.close();
    std::cout << "Solution written to " << filename << std::endl;
}

void writeConvergenceHistory(const std::vector<double>& res1, const std::vector<double>& res2, const std::vector<double>& res3, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file << "it,res1,res2,res3" << std::endl;
    for (size_t i = 0; i < res1.size(); ++i) {
        file << i+1 << "," << res1[i] << "," << res2[i] << "," << res3[i] << std::endl;
    }

    file.close();
    std::cout << "Convergence history written to " << filename << std::endl;
}