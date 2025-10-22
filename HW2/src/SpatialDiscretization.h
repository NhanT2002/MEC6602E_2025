#ifndef SPATIAL_DISCRETIZATION_H
#define SPATIAL_DISCRETIZATION_H
#include <vector>
#include <cmath>

// Forward declare Mesh to avoid circular include between mesh.h and this header.
class Mesh;

class SpatialDiscretization {
public:
    Mesh &mesh_;

    double gamma_ = 1.4; // ratio of specific heats
    double Mach_;
    double alpha_;
    double k2_;
    double k4_;

    double rhoInfty_ = 1.0; // freestream density
    double pInfty_ = 1.0; // freestream pressure
    double cInfty_ = std::sqrt(gamma_ * pInfty_ / rhoInfty_); // freestream speed of sound
    double uInfty_;
    double vInfty_;
    double EInfty_;

    int ncells_;
    int nfaces_;

    std::vector<double> W0; // rho
    std::vector<double> W1; // rho*u
    std::vector<double> W2; // rho*v
    std::vector<double> W3; // rho*E

    std::vector<double> rhorho;
    std::vector<double> uu;
    std::vector<double> vv;
    std::vector<double> EE;
    std::vector<double> pp;

    std::vector<double> Rc0; // convective residual: rho
    std::vector<double> Rc1; // convective residual: rho*u
    std::vector<double> Rc2; // convective residual: rho*v
    std::vector<double> Rc3; // convective residual: rho*E

    std::vector<double> Rd0; // diffusive residual: rho
    std::vector<double> Rd1; // diffusive residual: rho*u
    std::vector<double> Rd2; // diffusive residual: rho*v
    std::vector<double> Rd3; // diffusive residual: rho*E

    std::vector<double> LambdaI; // spectral radius in x-direction
    std::vector<double> LambdaJ; // spectral radius in y-direction

    std::vector<double> F0; // flux rho
    std::vector<double> F1; // flux rho*u
    std::vector<double> F2; // flux rho*v
    std::vector<double> F3; // flux rho*E

    std::vector<double> D0; // diffusive flux rho
    std::vector<double> D1; // diffusive flux rho*u
    std::vector<double> D2; // diffusive flux rho*v
    std::vector<double> D3; // diffusive flux rho*E



    // Constructor declared here; definition is in SpatialDiscretization.cpp
    SpatialDiscretization(Mesh &mesh, double Mach, double alpha, double k2, double k4);

    void initializeVariables(); // initialize flow variables
    void updateGhostCells(); // update ghost cells for IB
    void compute_convective_fluxes();
    std::tuple<double, double, double, double> compute_conservative_fluxes_IB(int fluidCell, int fluidCell_p1, int fluidCell_p2, 
                                        double area, double ib_nx, double ib_ny);
    void compute_convective_residuals();
    void compute_diffusive_residuals();
    void updatePrimitivesVariables();

    void compute_lambdas();
    void compute_diffusive_fluxes();
    std::tuple<double, double> epsilon(double p_Im1, double p_I, double p_Ip1, double p_Ip2);

    void run_even();
    void run_odd();
};

#endif // SPATIAL_DISCRETIZATION_H