#include "SpatialDiscretization.h"
#include "vector_helper.h"
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>
#include <Eigen/Dense>
#include "Multigrid.h"

void halo(Eigen::ArrayXXd& array) {
    int im1 = array.cols() - 3;
    int im2 = array.cols() - 4;
    int ip1 = 2;
    int ip2 = 3;

    array.col(1) = array.col(im1);
    array.col(0) = array.col(im2);
    array.col(array.cols()-2) = array.col(ip1);
    array.col(array.cols()-1) = array.col(ip2);
}

SpatialDiscretization::SpatialDiscretization(const Eigen::ArrayXXd& x,
                                             const Eigen::ArrayXXd& y,
                                             double rho,
                                             double u,
                                             double v,
                                             double E,
                                             double T,
                                             double p,
                                             double k2_coeff,
                                             double k4_coeff,
                                             double Mach,
                                             double U_ref)
    : x(x), y(y), rho(rho), u(u), v(v), E(E), T(T), p(p), k2_coeff(k2_coeff), k4_coeff(k4_coeff), Mach(Mach), U_ref(U_ref) {
    nvertex_y = x.rows();
    nvertex_x = x.cols();
    ncells_y = nvertex_y + 3; // nvertex_y - 1 + 4 for dummy cells
    ncells_x = nvertex_x + 3; // nvertex_x - 1 + 4 for halo cells
    ncells_domain_y = nvertex_y - 1; // number of cells in real domain (without dummys) 
    ncells_domain_x = nvertex_x - 1; // number of cells in real domain (without dummys)
    alpha = std::atan2(v, u);

    OMEGA.resize(ncells_y, ncells_x);
    sx_x.resize(ncells_y, ncells_x);
    sx_y.resize(ncells_y, ncells_x);
    sy_x.resize(ncells_y, ncells_x);
    sy_y.resize(ncells_y, ncells_x);
    Ds_x.resize(ncells_y, ncells_x); 
    Ds_y.resize(ncells_y, ncells_x); 
    nx_x.resize(ncells_y, ncells_x);
    nx_y.resize(ncells_y, ncells_x); 
    ny_x.resize(ncells_y, ncells_x); 
    ny_y.resize(ncells_y, ncells_x); 

    Ds_x_avg.resize(ncells_y, ncells_x);
    Ds_y_avg.resize(ncells_y, ncells_x);
    nx_x_avg.resize(ncells_y, ncells_x);
    nx_y_avg.resize(ncells_y, ncells_x);
    ny_x_avg.resize(ncells_y, ncells_x);
    ny_y_avg.resize(ncells_y, ncells_x);
    sx_x_avg.resize(ncells_y, ncells_x);
    sx_y_avg.resize(ncells_y, ncells_x);
    sy_x_avg.resize(ncells_y, ncells_x);
    sy_y_avg.resize(ncells_y, ncells_x);
    

    rho_cells.resize(ncells_y, ncells_x); 
    u_cells.resize(ncells_y, ncells_x); 
    v_cells.resize(ncells_y, ncells_x); 
    E_cells.resize(ncells_y, ncells_x); 
    p_cells.resize(ncells_y, ncells_x); 
    W_0.resize(ncells_y, ncells_x);
    W_1.resize(ncells_y, ncells_x); 
    W_2.resize(ncells_y, ncells_x);
    W_3.resize(ncells_y, ncells_x);

    Rc_0.resize(ncells_domain_y, ncells_domain_x); 
    Rc_1.resize(ncells_domain_y, ncells_domain_x); 
    Rc_2.resize(ncells_domain_y, ncells_domain_x); 
    Rc_3.resize(ncells_domain_y, ncells_domain_x); 
    Rd_0.resize(ncells_domain_y, ncells_domain_x); 
    Rd_1.resize(ncells_domain_y, ncells_domain_x); 
    Rd_2.resize(ncells_domain_y, ncells_domain_x); 
    Rd_3.resize(ncells_domain_y, ncells_domain_x); 
    Rd0_0.resize(ncells_domain_y, ncells_domain_x); 
    Rd0_1.resize(ncells_domain_y, ncells_domain_x);
    Rd0_2.resize(ncells_domain_y, ncells_domain_x);
    Rd0_3.resize(ncells_domain_y, ncells_domain_x);

    fluxx_0.resize(ncells_domain_y+1, ncells_domain_x+1);
    fluxx_1.resize(ncells_domain_y+1, ncells_domain_x+1); 
    fluxx_2.resize(ncells_domain_y+1, ncells_domain_x+1);
    fluxx_3.resize(ncells_domain_y+1, ncells_domain_x+1);
    fluxy_0.resize(ncells_domain_y+1, ncells_domain_x+1); 
    fluxy_1.resize(ncells_domain_y+1, ncells_domain_x+1); 
    fluxy_2.resize(ncells_domain_y+1, ncells_domain_x+1); 
    fluxy_3.resize(ncells_domain_y+1, ncells_domain_x+1); 
    
    dissipx_0.resize(ncells_domain_y+1, ncells_domain_x+1);
    dissipx_1.resize(ncells_domain_y+1, ncells_domain_x+1); 
    dissipx_2.resize(ncells_domain_y+1, ncells_domain_x+1); 
    dissipx_3.resize(ncells_domain_y+1, ncells_domain_x+1);
    dissipy_0.resize(ncells_domain_y+1, ncells_domain_x+1); 
    dissipy_1.resize(ncells_domain_y+1, ncells_domain_x+1); 
    dissipy_2.resize(ncells_domain_y+1, ncells_domain_x+1); 
    dissipy_3.resize(ncells_domain_y+1, ncells_domain_x+1); 
    // eps2_x.resize(ncells_domain_y, ncells_domain_x);
    // eps2_y.resize(ncells_domain_y, ncells_domain_x);
    // eps4_x.resize(ncells_domain_y, ncells_domain_x);
    // eps4_y.resize(ncells_domain_y, ncells_domain_x);
    Lambda_I.resize(ncells_y, ncells_x);
    Lambda_J.resize(ncells_y, ncells_x);

    restriction_operator_0.resize(ncells_domain_y, ncells_domain_x);
    restriction_operator_0.setZero();
    restriction_operator_1.resize(ncells_domain_y, ncells_domain_x);
    restriction_operator_1.setZero();
    restriction_operator_2.resize(ncells_domain_y, ncells_domain_x);
    restriction_operator_2.setZero();
    restriction_operator_3.resize(ncells_domain_y, ncells_domain_x);
    restriction_operator_3.setZero();
    forcing_function_0.resize(ncells_domain_y, ncells_domain_x);
    forcing_function_0.setZero();
    forcing_function_1.resize(ncells_domain_y, ncells_domain_x);
    forcing_function_1.setZero();
    forcing_function_2.resize(ncells_domain_y, ncells_domain_x);
    forcing_function_2.setZero();
    forcing_function_3.resize(ncells_domain_y, ncells_domain_x);
    forcing_function_3.setZero();
    prolongation_operator_0.resize(ncells_y, ncells_x);
    prolongation_operator_0.setZero();
    prolongation_operator_1.resize(ncells_y, ncells_x);
    prolongation_operator_1.setZero();
    prolongation_operator_2.resize(ncells_y, ncells_x);
    prolongation_operator_2.setZero();
    prolongation_operator_3.resize(ncells_y, ncells_x);
    prolongation_operator_3.setZero();
    deltaW2h_0.resize(ncells_y, ncells_x);
    deltaW2h_0.setZero();
    deltaW2h_1.resize(ncells_y, ncells_x);
    deltaW2h_1.setZero();
    deltaW2h_2.resize(ncells_y, ncells_x);
    deltaW2h_2.setZero();
    deltaW2h_3.resize(ncells_y, ncells_x);
    deltaW2h_3.setZero();
    W2h_0.resize(ncells_y, ncells_x);
    W2h_0.setZero();
    W2h_1.resize(ncells_y, ncells_x);
    W2h_1.setZero();
    W2h_2.resize(ncells_y, ncells_x);
    W2h_2.setZero();
    W2h_3.resize(ncells_y, ncells_x);
    W2h_3.setZero();

    for (int j=2; j<ncells_y-2; j++) {
        for (int i=2; i<ncells_x-2; i++) {
            int jj = j - 2;
            int ii = i - 2;
            int jjp1 = jj + 1;
            int iip1 = ii + 1;    

            const double &x1 = x(jj, ii);
            const double &x2 = x(jj, iip1);
            const double &x3 = x(jjp1, iip1);
            const double &x4 = x(jjp1, ii);
            const double &y1 = y(jj, ii);
            const double &y2 = y(jj, iip1);
            const double &y3 = y(jjp1, iip1);
            const double &y4 = y(jjp1, ii);

            OMEGA(j, i) = 0.5 * ((x1-x3)*(y2-y4) + (x4-x2)*(y1-y3));

            double Sx_x = y2 - y1;
            double Sx_y = x1 - x2;
            double Sy_x = y1 - y4;
            double Sy_y = x4 - x1;

            Ds_x(j, i) = std::sqrt(Sx_x*Sx_x + Sx_y*Sx_y);
            Ds_y(j, i) = std::sqrt(Sy_x*Sy_x + Sy_y*Sy_y);

            nx_x(j, i) = Sx_x / Ds_x(j, i);
            nx_y(j, i) = Sx_y / Ds_x(j, i);
            ny_x(j, i) = Sy_x / Ds_y(j, i);
            ny_y(j, i) = Sy_y / Ds_y(j, i);

            sx_x(j, i) = Sx_x;
            sx_y(j, i) = Sx_y;
            sy_x(j, i) = Sy_x;
            sy_y(j, i) = Sy_y;

            rho_cells(j, i) = rho;
            u_cells(j, i) = u;
            v_cells(j, i) = v;
            E_cells(j, i) = E;
            p_cells(j, i) = p;
        }
    }
    // Dummys same geometry as closest cell
    OMEGA.row(0) = OMEGA.row(2);
    OMEGA.row(1) = OMEGA.row(2);
    OMEGA.row(ncells_y-2) = OMEGA.row(ncells_y-3);
    OMEGA.row(ncells_y-1) = OMEGA.row(ncells_y-3);

    nx_x.row(0) = nx_x.row(2);
    nx_x.row(1) = nx_x.row(2);
    nx_y.row(0) = nx_y.row(2);
    nx_y.row(1) = nx_y.row(2);
    // nx_x.row(ncells_y-2) compute below

    ny_x.row(0) = ny_x.row(2);
    ny_x.row(1) = ny_x.row(2);
    ny_x.row(ncells_y-2) = ny_x.row(ncells_y-3);
    ny_x.row(ncells_y-1) = ny_x.row(ncells_y-3);
    ny_y.row(0) = ny_y.row(2);
    ny_y.row(1) = ny_y.row(2);
    ny_y.row(ncells_y-2) = ny_y.row(ncells_y-3);
    ny_y.row(ncells_y-1) = ny_y.row(ncells_y-3);

    sx_x.row(0) = sx_x.row(2);
    sx_x.row(1) = sx_x.row(2);
    sx_y.row(0) = sx_y.row(2);
    sx_y.row(1) = sx_y.row(2);

    sy_x.row(0) = sy_x.row(2);
    sy_x.row(1) = sy_x.row(2);
    sy_x.row(ncells_y-2) = sy_x.row(ncells_y-3);
    sy_x.row(ncells_y-1) = sy_x.row(ncells_y-3);
    sy_y.row(0) = sy_y.row(2);
    sy_y.row(1) = sy_y.row(2);
    sy_y.row(ncells_y-2) = sy_y.row(ncells_y-3);
    sy_y.row(ncells_y-1) = sy_y.row(ncells_y-3);


    Ds_x.row(0) = Ds_x.row(2);
    Ds_x.row(1) = Ds_x.row(2);
    Ds_x.row(ncells_y-2) = Ds_x.row(ncells_y-3);
    Ds_x.row(ncells_y-1) = Ds_x.row(ncells_y-3);
    Ds_y.row(0) = Ds_y.row(2);
    Ds_y.row(1) = Ds_y.row(2);
    Ds_y.row(ncells_y-2) = Ds_y.row(ncells_y-3);
    Ds_y.row(ncells_y-1) = Ds_y.row(ncells_y-3);

    // Compute normal vector of x face in first farfield dummy cell
    int j = ncells_y - 2;
    int jj = j - 2;
    for (int i=2; i<ncells_x-2; i++) {
        const double &x1 = x(jj, i-2);
        const double &x2 = x(jj, i-1); // i-2+1
        const double &y1 = y(jj, i-2);
        const double &y2 = y(jj, i-1); // i-2+1

        double Sx_x = y2 - y1;
        double Sx_y = x1 - x2;

        sx_x(j, i) = Sx_x;
        sx_x(j+1, i) = Sx_x;
        sx_y(j, i) = Sx_y;
        sx_y(j+1, i) = Sx_y;

        Ds_x(j, i) = std::sqrt(Sx_x*Sx_x + Sx_y*Sx_y);
        Ds_x(j+1, i) = std::sqrt(Sx_x*Sx_x + Sx_y*Sx_y);

        nx_x(j, i) = Sx_x / Ds_x(j, i);
        nx_y(j, i) = Sx_y / Ds_x(j, i);
        nx_x(j+1, i) = Sx_x / Ds_x(j, i);
        nx_y(j+1, i) = Sx_y / Ds_x(j, i);
    }
    // Halo cells
    halo(OMEGA);
    halo(Ds_x);
    halo(Ds_y);
    halo(sx_x);
    halo(sx_y);
    halo(sy_x);
    halo(sy_y);
    halo(nx_x);
    halo(nx_y);
    halo(ny_x);
    halo(ny_y);
       

    // Compute average cell geometry
    auto seq_y = Eigen::seq(1, ncells_y-2);
    auto seq_yp1 = Eigen::seq(2, ncells_y-1);
    auto seq_x = Eigen::seq(1, ncells_x-2);
    auto seq_xp1 = Eigen::seq(2, ncells_x-1);
    Ds_x_avg(seq_y, seq_x) = 0.5*(Ds_x(seq_yp1, seq_x) + Ds_x(seq_y, seq_x));
    Ds_y_avg(seq_y, seq_x) = 0.5*(Ds_y(seq_y, seq_xp1) + Ds_y(seq_y, seq_x));
    nx_x_avg(seq_y, seq_x) = 0.5*(nx_x(seq_yp1, seq_x) + nx_x(seq_y, seq_x));
    nx_y_avg(seq_y, seq_x) = 0.5*(nx_y(seq_yp1, seq_x) + nx_y(seq_y, seq_x));
    ny_x_avg(seq_y, seq_x) = 0.5*(ny_x(seq_y, seq_xp1) + ny_x(seq_y, seq_x));
    ny_y_avg(seq_y, seq_x) = 0.5*(ny_y(seq_y, seq_xp1) + ny_y(seq_y, seq_x));
    sx_x_avg(seq_y, seq_x) = 0.5*(sx_x(seq_yp1, seq_x) + sx_x(seq_y, seq_x));
    sx_y_avg(seq_y, seq_x) = 0.5*(sx_y(seq_yp1, seq_x) + sx_y(seq_y, seq_x));
    sy_x_avg(seq_y, seq_x) = 0.5*(sy_x(seq_y, seq_xp1) + sy_x(seq_y, seq_x));
    sy_y_avg(seq_y, seq_x) = 0.5*(sy_y(seq_y, seq_xp1) + sy_y(seq_y, seq_x));

    // std::cout << "sx_x_avg\n" << sx_x_avg << std::endl;
    // std::cout << "sx_y_avg\n" << sx_y_avg << std::endl;
    // std::cout << "sy_x_avg\n" << sy_x_avg << std::endl;
    // std::cout << "sy_y_avg\n" << sy_y_avg << std::endl;



}

void SpatialDiscretization::update_W() {
    W_0 = rho_cells;
    W_1 = rho_cells * u_cells;
    W_2 = rho_cells * v_cells;
    W_3 = rho_cells * E_cells;
}

void SpatialDiscretization::update_conservative_variables() {
    rho_cells = W_0;
    u_cells = W_1 / W_0;
    v_cells = W_2 / W_0;
    E_cells = W_3 / W_0;
    p_cells = (1.4-1)*rho_cells*(E_cells - 0.5*(u_cells*u_cells + v_cells*v_cells));
}

void SpatialDiscretization::update_halo() {
    int im1 = ncells_x - 3;
    int im2 = ncells_x - 4;
    int ip1 = 2;
    int ip2 = 3;

    rho_cells.col(1) = rho_cells.col(im1);
    u_cells.col(1) = u_cells.col(im1);
    v_cells.col(1) = v_cells.col(im1);
    E_cells.col(1) = E_cells.col(im1);
    p_cells.col(1) = p_cells.col(im1);

    rho_cells.col(0) = rho_cells.col(im2);
    u_cells.col(0) = u_cells.col(im2);
    v_cells.col(0) = v_cells.col(im2);
    E_cells.col(0) = E_cells.col(im2);
    p_cells.col(0) = p_cells.col(im2);

    rho_cells.col(ncells_x-2) = rho_cells.col(ip1);
    u_cells.col(ncells_x-2) = u_cells.col(ip1);
    v_cells.col(ncells_x-2) = v_cells.col(ip1);
    E_cells.col(ncells_x-2) = E_cells.col(ip1);
    p_cells.col(ncells_x-2) = p_cells.col(ip1);

    rho_cells.col(ncells_x-1) = rho_cells.col(ip2);
    u_cells.col(ncells_x-1) = u_cells.col(ip2);
    v_cells.col(ncells_x-1) = v_cells.col(ip2);
    E_cells.col(ncells_x-1) = E_cells.col(ip2);
    p_cells.col(ncells_x-1) = p_cells.col(ip2);

}

std::tuple<double, double, double> SpatialDiscretization::compute_coeff() {
    double x_ref = 0.25;
    double y_ref = 0.0;
    double c = 1.0;

    auto seqx = Eigen::seq(2, this->ncells_x-3);    
    Eigen::ArrayXXd p_wall = 0.5*(3*this->p_cells(2, seqx) - this->p_cells(3, seqx));
    double Fx = (p_wall*this->nx_x(2, seqx)*this->Ds_x(2, seqx)).sum();
    double Fy = (p_wall*this->nx_y(2, seqx)*this->Ds_x(2, seqx)).sum();

    Eigen::ArrayXXd x_mid = 0.5*(this->x(0, Eigen::seq(0, x.cols()-2)) + this->x(0, Eigen::seq(1, x.cols()-1)));
    Eigen::ArrayXXd y_mid = 0.5*(this->y(0, Eigen::seq(0, x.cols()-2)) + this->y(0, Eigen::seq(1, x.cols()-1)));
    double M = (this->p_cells(2, seqx)*(-(x_mid-x_ref)*this->nx_y(2, seqx) + (y_mid-y_ref)*this->nx_x(2, seqx))*this->Ds_x(2, seqx)).sum();

    double L = Fy*std::cos(this->alpha) - Fx*std::sin(this->alpha);
    double D = Fy*std::sin(this->alpha) + Fx*std::cos(this->alpha);

    double C_l = L/(0.5*rho*(u*u+v*v)*c);
    double C_d = D/(0.5*rho*(u*u+v*v)*c);
    double C_m = M/(0.5*rho*(u*u+v*v)*c*c);

    return {C_l, C_d, C_m};
}

void SpatialDiscretization::compute_dummy_cells() {
    // Solid wall
    // Eigen::Array<double, 1, Eigen::Dynamic> V = nx_x(2, Eigen::all)*u_cells(2, Eigen::all) + nx_y(2, Eigen::all)*v_cells(2, Eigen::all);
    // Eigen::Array<double, 1, Eigen::Dynamic> u_dummy = u_cells(2, Eigen::all) - 2*V*nx_x(2, Eigen::all);
    // Eigen::Array<double, 1, Eigen::Dynamic> v_dummy = v_cells(2, Eigen::all) - 2*V*nx_y(2, Eigen::all);
    // Eigen::Array<double, 1, Eigen::Dynamic> rho_dummy = rho_cells(2, Eigen::all);
    
    // Eigen::Array<double, 1, Eigen::Dynamic> E_dummy = E_cells(2, Eigen::all);
    // Eigen::Array<double, 1, Eigen::Dynamic> p_dummy  = (1.4-1)*rho_dummy*(E_dummy - 0.5*(u_dummy*u_dummy + v_dummy*v_dummy));

    // rho_cells.row(1) = rho_dummy;
    // u_cells.row(1) = u_dummy;
    // v_cells.row(1) = v_dummy;
    // E_cells.row(1) = E_dummy;
    // p_cells.row(1) = p_dummy;

    // rho_cells.row(0) = rho_dummy;
    // u_cells.row(0) = u_dummy;
    // v_cells.row(0) = v_dummy;
    // E_cells.row(0) = E_dummy;
    // p_cells.row(0) = p_dummy;

    rho_cells.row(1) = 2*rho_cells.row(2) - rho_cells.row(3);
    u_cells.row(1) = (2*u_cells.row(2)*rho_cells.row(2) - u_cells.row(3)*rho_cells.row(3))/rho_cells.row(1);
    v_cells.row(1) = (2*v_cells.row(2)*rho_cells.row(2) - v_cells.row(3)*rho_cells.row(3))/rho_cells.row(1);
    E_cells.row(1) = (2*E_cells.row(2)*rho_cells.row(2) - E_cells.row(3)*rho_cells.row(3))/rho_cells.row(1);
    p_cells.row(1) = (1.4-1)*rho_cells.row(1)*(E_cells.row(1) - 0.5*(u_cells.row(1)*u_cells.row(1) + v_cells.row(1)*v_cells.row(1)));

    rho_cells.row(0) = 3*rho_cells.row(2) - 2*rho_cells.row(3);
    u_cells.row(0) = (3*u_cells.row(2)*rho_cells.row(2) - 2*u_cells.row(3)*rho_cells.row(3))/rho_cells.row(0);
    v_cells.row(0) = (3*v_cells.row(2)*rho_cells.row(2) - 2*v_cells.row(3)*rho_cells.row(3))/rho_cells.row(0);
    E_cells.row(0) = (3*E_cells.row(2)*rho_cells.row(2) - 2*E_cells.row(3)*rho_cells.row(3))/rho_cells.row(0);
    p_cells.row(0) = (1.4-1)*rho_cells.row(0)*(E_cells.row(0) - 0.5*(u_cells.row(0)*u_cells.row(0) + v_cells.row(0)*v_cells.row(0)));

    // Far field
    int j_last_cells = ncells_y - 3;
    int j = j_last_cells + 1;
    int jj = j + 1;
    Eigen::Array<double, 1, Eigen::Dynamic> c_cells = (1.4*p_cells(j_last_cells, Eigen::all)/rho_cells(j_last_cells, Eigen::all)).sqrt(); // speed of sound
    Eigen::Array<double, 1, Eigen::Dynamic> M_cells = (u_cells(j_last_cells, Eigen::all).square() + v_cells(j_last_cells, Eigen::all).square()).sqrt()/c_cells; // Mach number
    Eigen::Array<double, 1, Eigen::Dynamic> nx_cells = -1*nx_x(j, Eigen::all);
    Eigen::Array<double, 1, Eigen::Dynamic> ny_cells = -1*nx_y(j, Eigen::all);

    for (int i=2; i<ncells_x-2; i++) {
        double& rho_d = rho_cells(j_last_cells, i);
        double& u_d = u_cells(j_last_cells, i);
        double& v_d = v_cells(j_last_cells, i);
        double& E_d = E_cells(j_last_cells, i);
        double& p_d = p_cells(j_last_cells, i);
        double& c = c_cells(0, i);
        double& M = M_cells(0, i);
        double& nx = nx_cells(0, i);
        double& ny = ny_cells(0, i);

        if (u_d*nx + v_d*ny > 0) { // out of cell
            if (M >= 1) { // supersonic
                rho_cells(j, i) = rho_cells(j_last_cells, i);
                u_cells(j, i) = u_cells(j_last_cells, i);
                v_cells(j, i) = v_cells(j_last_cells, i);
                E_cells(j, i) = E_cells(j_last_cells, i);
                p_cells(j, i) = p_cells(j_last_cells, i);

                rho_cells(jj, i) = rho_cells(j_last_cells, i);
                u_cells(jj, i) = u_cells(j_last_cells, i);
                v_cells(jj, i) = v_cells(j_last_cells, i);
                E_cells(jj, i) = E_cells(j_last_cells, i);
                p_cells(jj, i) = p_cells(j_last_cells, i);
            } 
            else { // subsonic
                // Vortex correction
                int j_mesh = j_last_cells - 2;
                int i_mesh = i - 2;
                double x_cell = 0.5*(this->x(j_mesh, i_mesh) + this->x(j_mesh+1, i_mesh+1));
                double y_cell = 0.5*(this->y(j_mesh, i_mesh) + this->y(j_mesh+1, i_mesh+1));
                double d = std::sqrt((x_cell - 0.25)*(x_cell - 0.25) + y_cell*y_cell);
                double theta = std::atan2(y_cell, (x_cell - 0.25));
                // std::cout << "d: " << d << " theta: " << theta << std::endl;
                auto [C_l, C_d, C_m] = compute_coeff();
                double gamma = 0.5*std::sqrt(this->u*this->u + this->v*this->v)*C_l;
                double u_inf = this->u + (gamma*std::sqrt(1-this->Mach*this->Mach)/(2*M_PI*d))*std::sin(theta)/(1-this->Mach*this->Mach*std::sin(theta-this->alpha)*std::sin(theta-this->alpha));
                double v_inf = this->v - (gamma*std::sqrt(1-this->Mach*this->Mach)/(2*M_PI*d))*std::cos(theta)/(1-this->Mach*this->Mach*std::sin(theta-this->alpha)*std::sin(theta-this->alpha));
                double p_inf = std::pow(std::pow(this->p, (1.4-1)/1.4) + (1.4-1)/1.4*(this->p*((this->u*this->u + this->v*this->v)-(u_inf*u_inf+v_inf*v_inf))/(2*std::pow(this->p, 1/1.4))), 1.4/(1.4-1));
                // double rho_inf = this->rho*std::pow(p_inf/this->p, 1/1.4);

                // double p_inf = this->p;
                // double rho_inf = this->rho;
                // double u_inf = this->u;
                // double v_inf = this->v;

                double p_b = p_inf;
                double rho_b = rho_d + (p_b - p_d)/(c*c);
                double u_b = u_d + nx*(p_d - p_b)/(rho_d*c);
                double v_b = v_d + ny*(p_d - p_b)/(rho_d*c);
                double E_b = p_b/(rho_b*(1.4-1)) + 0.5*(u_b*u_b + v_b*v_b);

                // rho_cells(j, i) = 2*rho_b - rho_d;
                // u_cells(j, i) = (2*(u_b*rho_b) - (u_d*rho_d))/rho_cells(j, i);
                // v_cells(j, i) = (2*(v_b*rho_b) - (v_d*rho_d))/rho_cells(j, i);
                // E_cells(j, i) = (2*(E_b*rho_b) - (E_d*rho_d))/rho_cells(j, i);
                // p_cells(j, i) = (1.4-1)*rho_cells(j, i)*(E_cells(j, i) - 0.5*(u_cells(j, i)*u_cells(j, i) + v_cells(j, i)*v_cells(j, i)));

                // rho_cells(jj, i) = rho_cells(j, i);
                // u_cells(jj, i) = u_cells(j, i);
                // v_cells(jj, i) = v_cells(j, i);
                // E_cells(jj, i) = E_cells(j, i);
                // p_cells(jj, i) = p_cells(j, i);

                rho_cells(j, i) = rho_b;
                u_cells(j, i) = u_b;
                v_cells(j, i) = v_b;
                E_cells(j, i) = E_b;
                p_cells(j, i) = p_b;

                rho_cells(jj, i) = 2*rho_cells(j, i) - rho_cells(j-1, i);
                u_cells(jj, i) = 2*u_cells(j, i) - u_cells(j-1, i);
                v_cells(jj, i) = 2*v_cells(j, i) - v_cells(j-1, i);
                E_cells(jj, i) = 2*E_cells(j, i) - E_cells(j-1, i);
                p_cells(jj, i) = 2*p_cells(j, i) - p_cells(j-1, i);

                }
        }
        else { // in cell
            if (M >=1) { // supersonic
                rho_cells(j, i) = this->rho;
                u_cells(j, i) = this->u;
                v_cells(j, i) = this->v;
                E_cells(j, i) = this->E;
                p_cells(j, i) = this->p;

                rho_cells(jj, i) = rho_cells(j, i);
                u_cells(jj, i) = u_cells(j, i);
                v_cells(jj, i) = v_cells(j, i);
                E_cells(jj, i) = E_cells(j, i);
                p_cells(jj, i) = p_cells(j, i);
            }
            else { // subsonic
                // Vortex correction
                int j_mesh = j_last_cells - 2;
                int i_mesh = i - 2;
                double x_cell = 0.5*(this->x(j_mesh, i_mesh) + this->x(j_mesh+1, i_mesh+1));
                double y_cell = 0.5*(this->y(j_mesh, i_mesh) + this->y(j_mesh+1, i_mesh+1));
                double d = std::sqrt((x_cell - 0.25)*(x_cell - 0.25) + y_cell*y_cell);
                double theta = std::atan2(y_cell, (x_cell - 0.25));
                // std::cout << "d: " << d << " theta: " << theta << std::endl;
                auto [C_l, C_d, C_m] = compute_coeff();
                double gamma = 0.5*std::sqrt(this->u*this->u + this->v*this->v)*C_l;
                double u_inf = this->u + (gamma*std::sqrt(1-this->Mach*this->Mach)/(2*M_PI*d))*std::sin(theta)/(1-this->Mach*this->Mach*std::sin(theta-this->alpha)*std::sin(theta-this->alpha));
                double v_inf = this->v - (gamma*std::sqrt(1-this->Mach*this->Mach)/(2*M_PI*d))*std::cos(theta)/(1-this->Mach*this->Mach*std::sin(theta-this->alpha)*std::sin(theta-this->alpha));
                double p_inf = std::pow(std::pow(this->p, (1.4-1)/1.4) + (1.4-1)/1.4*(this->p*((this->u*this->u + this->v*this->v)-(u_inf*u_inf+v_inf*v_inf))/(2*std::pow(this->p, 1/1.4))), 1.4/(1.4-1));
                double rho_inf = this->rho*std::pow(p_inf/this->p, 1/1.4);

                // double p_inf = this->p;
                // double rho_inf = this->rho;
                // double u_inf = this->u;
                // double v_inf = this->v;

                double p_b = 0.5*(p_inf + p_d - rho_d*c*(nx*(u_inf - u_d) + ny*(v_inf - v_d)));
                double rho_b = rho_inf + (p_b - p_inf)/(c*c);
                double u_b = u_inf - nx*(p_inf - p_b)/(rho_d*c);
                double v_b = v_inf - ny*(p_inf - p_b)/(rho_d*c);
                double E_b = p_b/(rho_b*(1.4-1)) + 0.5*(u_b*u_b + v_b*v_b);

                // rho_cells(j, i) = 2*rho_b - rho_d;
                // u_cells(j, i) = (2*(u_b*rho_b) - (u_d*rho_d))/rho_cells(j, i);
                // v_cells(j, i) = (2*(v_b*rho_b) - (v_d*rho_d))/rho_cells(j, i);
                // E_cells(j, i) = (2*(E_b*rho_b) - (E_d*rho_d))/rho_cells(j, i);
                // p_cells(j, i) = (1.4-1)*rho_cells(j, i)*(E_cells(j, i) - 0.5*(u_cells(j, i)*u_cells(j, i) + v_cells(j, i)*v_cells(j, i)));

                // rho_cells(jj, i) = rho_cells(j, i);
                // u_cells(jj, i) = u_cells(j, i);
                // v_cells(jj, i) = v_cells(j, i);
                // E_cells(jj, i) = E_cells(j, i);
                // p_cells(jj, i) = p_cells(j, i);

                rho_cells(j, i) = rho_b;
                u_cells(j, i) = u_b;
                v_cells(j, i) = v_b;
                E_cells(j, i) = E_b;
                p_cells(j, i) = p_b;

                rho_cells(jj, i) = 2*rho_cells(j, i) - rho_cells(j-1, i);
                u_cells(jj, i) = 2*u_cells(j, i) - u_cells(j-1, i);
                v_cells(jj, i) = 2*v_cells(j, i) - v_cells(j-1, i);
                E_cells(jj, i) = 2*E_cells(j, i) - E_cells(j-1, i);
                p_cells(jj, i) = 2*p_cells(j, i) - p_cells(j-1, i);
            }
        }
    }
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> SpatialDiscretization::FcDs(const Eigen::ArrayXXd& rhoo, const Eigen::ArrayXXd& uu, const Eigen::ArrayXXd& vv, const Eigen::ArrayXXd& EE, const Eigen::ArrayXXd& pp, 
                                            const Eigen::ArrayXXd& nx, const Eigen::ArrayXXd& ny, const Eigen::ArrayXXd& Ds) {
    Eigen::ArrayXXd V = nx*uu + ny*vv;
    Eigen::ArrayXXd H = EE + pp/rhoo;

    return {rhoo*V*Ds, (rhoo*uu*V + nx*pp)*Ds, (rhoo*vv*V + ny*pp)*Ds, rhoo*H*V*Ds};
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd, Eigen::ArrayXXd> SpatialDiscretization::FcDs_wall(const Eigen::ArrayXXd& pp, const Eigen::ArrayXXd& nx, const Eigen::ArrayXXd& ny, const Eigen::ArrayXXd& Ds) {
    Eigen::ArrayXXd V = Eigen::ArrayXXd::Zero(nx.rows(), nx.cols());

    return {V, nx*pp*Ds, ny*pp*Ds, V};
}

void SpatialDiscretization::compute_flux() {
    // The wall flux will be computed later
    auto seqy = Eigen::seq(3, ncells_y-2);
    auto seqx = Eigen::seq(2, ncells_x-2);
    auto seqy_m1 = Eigen::seq(2, ncells_y-3);
    auto seqx_m1 = Eigen::seq(1, ncells_x-3);

    // y direction
    Eigen::ArrayXXd avg_W_0 = 0.5*(W_0(seqy, seqx) + W_0(seqy_m1, seqx));
    Eigen::ArrayXXd avg_W_1 = 0.5*(W_1(seqy, seqx) + W_1(seqy_m1, seqx));
    Eigen::ArrayXXd avg_W_2 = 0.5*(W_2(seqy, seqx) + W_2(seqy_m1, seqx));
    Eigen::ArrayXXd avg_W_3 = 0.5*(W_3(seqy, seqx) + W_3(seqy_m1, seqx));

    Eigen::ArrayXXd avg_u = avg_W_1 / avg_W_0;
    Eigen::ArrayXXd avg_v = avg_W_2 / avg_W_0;
    Eigen::ArrayXXd avg_E = avg_W_3 / avg_W_0;
    Eigen::ArrayXXd avg_p = (1.4-1)*avg_W_0*(avg_E - 0.5*(avg_u*avg_u + avg_v*avg_v));

    Eigen::ArrayXXd p_wall = 0.125*(15*p_cells(2, seqx) - 10*p_cells(3, seqx) + 3*p_cells(4, seqx));

    auto [Fcy0, Fcy1, Fcy2, Fcy3] = FcDs(avg_W_0, avg_u, avg_v, avg_E, avg_p, nx_x(seqy, seqx), nx_y(seqy, seqx), Ds_x(seqy, seqx));
    auto [Fcy0_wall, Fcy1_wall, Fcy2_wall, Fcy3_wall] = FcDs_wall(p_wall, nx_x(2, seqx), nx_y(2, seqx), Ds_x(2, seqx));

    fluxy_0(Eigen::seq(1, ncells_domain_y), Eigen::all) = Fcy0;
    fluxy_1(Eigen::seq(1, ncells_domain_y), Eigen::all) = Fcy1;
    fluxy_2(Eigen::seq(1, ncells_domain_y), Eigen::all) = Fcy2;
    fluxy_3(Eigen::seq(1, ncells_domain_y), Eigen::all) = Fcy3;

    fluxy_0.row(0) = Fcy0_wall;
    fluxy_1.row(0) = Fcy1_wall;
    fluxy_2.row(0) = Fcy2_wall;
    fluxy_3.row(0) = Fcy3_wall;

    // x direction

    seqy = Eigen::seq(2, ncells_y-2);
    seqx = Eigen::seq(2, ncells_x-2);
    seqy_m1 = Eigen::seq(1, ncells_y-3);
    seqx_m1 = Eigen::seq(1, ncells_x-3);

    avg_W_0 = 0.5*(W_0(seqy, seqx) + W_0(seqy, seqx_m1));
    avg_W_1 = 0.5*(W_1(seqy, seqx) + W_1(seqy, seqx_m1));
    avg_W_2 = 0.5*(W_2(seqy, seqx) + W_2(seqy, seqx_m1));
    avg_W_3 = 0.5*(W_3(seqy, seqx) + W_3(seqy, seqx_m1));

    avg_u = avg_W_1 / avg_W_0;
    avg_v = avg_W_2 / avg_W_0;
    avg_E = avg_W_3 / avg_W_0;
    avg_p = (1.4-1)*avg_W_0*(avg_E - 0.5*(avg_u*avg_u + avg_v*avg_v));

    auto [Fcx0, Fcx1, Fcx2, Fcx3] = FcDs(avg_W_0, avg_u, avg_v, avg_E, avg_p, ny_x(seqy, seqx), ny_y(seqy, seqx), Ds_y(seqy, seqx));
    fluxx_0 = Fcx0;
    fluxx_1 = Fcx1;
    fluxx_2 = Fcx2;
    fluxx_3 = Fcx3;

}

void SpatialDiscretization::compute_lambda() {
    // Eigen::ArrayXXd cc_cells = (1.4*p_cells/rho_cells);
    // Lambda_I = (u_cells*sy_x_avg + v_cells*sy_y_avg).abs() + (cc_cells*(sy_x_avg*sy_x_avg+sy_y_avg*sy_y_avg)).sqrt();
    // Lambda_J = (u_cells*sx_x_avg + v_cells*sx_y_avg).abs() + (cc_cells*(sx_x_avg*sx_x_avg+sx_y_avg*sx_y_avg)).sqrt();

    Eigen::ArrayXXd c_cells = (1.4*p_cells/rho_cells).sqrt();
    Eigen::ArrayXXd V_I = u_cells*ny_x_avg + v_cells*ny_y_avg;
    Eigen::ArrayXXd V_J = u_cells*nx_x_avg + v_cells*nx_y_avg;
    Lambda_I = (V_I.abs() + c_cells)*Ds_y_avg;
    Lambda_J = (V_J.abs() + c_cells)*Ds_x_avg;
    
    halo(Lambda_I);
    halo(Lambda_J);
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> SpatialDiscretization::compute_epsilon(const Eigen::ArrayXXd& p_Im1, const Eigen::ArrayXXd& p_I, const Eigen::ArrayXXd& p_Ip1, const Eigen::ArrayXXd& p_Ip2,  double k2, double k4) {
    Eigen::ArrayXXd gamma_I = ((p_Ip1 - 2.0*p_I + p_Im1)/(p_Ip1 + 2.0*p_I + p_Im1)).abs();
    Eigen::ArrayXXd gamma_Ip1 = ((p_Ip2 - 2.0*p_Ip1 + p_I)/(p_Ip2 + 2.0*p_Ip1 + p_I)).abs();

    Eigen::ArrayXXd eps2 = k2*gamma_I.max(gamma_Ip1);
    Eigen::ArrayXXd eps4 = (k4-eps2).max(0.0);

    return {eps2, eps4};
}

void SpatialDiscretization::compute_dissipation() {
    auto seqy_p1 = Eigen::seq(3, ncells_y-1);
    auto seqx_p1 = Eigen::seq(3, ncells_x-1);
    auto seqy = Eigen::seq(2, ncells_y-2);
    auto seqx = Eigen::seq(2, ncells_x-2);
    auto seqy_m1 = Eigen::seq(1, ncells_y-3);
    auto seqx_m1 = Eigen::seq(1, ncells_x-3);
    auto seqy_m2 = Eigen::seq(0, ncells_y-4);
    auto seqx_m2 = Eigen::seq(0, ncells_x-4);

    auto [eps2_x, eps4_x] = compute_epsilon(p_cells(seqy, seqx_p1), p_cells(seqy, seqx), p_cells(seqy, seqx_m1), p_cells(seqy, seqx_m2), k2_coeff*0.25, k4_coeff*0.015625); // k2=1/4, k4=1/64
    auto [eps2_y, eps4_y] = compute_epsilon(p_cells(seqy_p1, seqx), p_cells(seqy, seqx), p_cells(seqy_m1, seqx), p_cells(seqy_m2, seqx), k2_coeff*0.25, k4_coeff*0.015625); // k2=1/4, k4=1/64

    Eigen::ArrayXXd Lambda_x_I = 0.5*(Lambda_I(seqy, seqx) + Lambda_I(seqy, seqx_m1));
    Eigen::ArrayXXd Lambda_x_J = 0.5*(Lambda_J(seqy, seqx) + Lambda_J(seqy, seqx_m1));
    Eigen::ArrayXXd Lambda_x_S = Lambda_x_I + Lambda_x_J;

    Eigen::ArrayXXd Lambda_y_I = 0.5*(Lambda_I(seqy, seqx) + Lambda_I(seqy_m1, seqx));
    Eigen::ArrayXXd Lambda_y_J = 0.5*(Lambda_J(seqy, seqx) + Lambda_J(seqy_m1, seqx));
    Eigen::ArrayXXd Lambda_y_S = Lambda_y_I + Lambda_y_J;

    // Dissipation
    dissipx_0 = Lambda_x_S*(eps2_x*(W_0(seqy, seqx_m1)-W_0(seqy, seqx)) - eps4_x*(W_0(seqy, seqx_m2)-3*W_0(seqy, seqx_m1)+3*W_0(seqy, seqx)-W_0(seqy, seqx_p1)));
    dissipx_1 = Lambda_x_S*(eps2_x*(W_1(seqy, seqx_m1)-W_1(seqy, seqx)) - eps4_x*(W_1(seqy, seqx_m2)-3*W_1(seqy, seqx_m1)+3*W_1(seqy, seqx)-W_1(seqy, seqx_p1)));
    dissipx_2 = Lambda_x_S*(eps2_x*(W_2(seqy, seqx_m1)-W_2(seqy, seqx)) - eps4_x*(W_2(seqy, seqx_m2)-3*W_2(seqy, seqx_m1)+3*W_2(seqy, seqx)-W_2(seqy, seqx_p1)));
    dissipx_3 = Lambda_x_S*(eps2_x*(W_3(seqy, seqx_m1)-W_3(seqy, seqx)) - eps4_x*(W_3(seqy, seqx_m2)-3*W_3(seqy, seqx_m1)+3*W_3(seqy, seqx)-W_3(seqy, seqx_p1)));

    dissipy_0 = Lambda_y_S*(eps2_y*(W_0(seqy_m1, seqx)-W_0(seqy, seqx)) - eps4_y*(W_0(seqy_m2, seqx)-3*W_0(seqy_m1, seqx)+3*W_0(seqy, seqx)-W_0(seqy_p1, seqx)));
    dissipy_1 = Lambda_y_S*(eps2_y*(W_1(seqy_m1, seqx)-W_1(seqy, seqx)) - eps4_y*(W_1(seqy_m2, seqx)-3*W_1(seqy_m1, seqx)+3*W_1(seqy, seqx)-W_1(seqy_p1, seqx)));
    dissipy_2 = Lambda_y_S*(eps2_y*(W_2(seqy_m1, seqx)-W_2(seqy, seqx)) - eps4_y*(W_2(seqy_m2, seqx)-3*W_2(seqy_m1, seqx)+3*W_2(seqy, seqx)-W_2(seqy_p1, seqx)));
    dissipy_3 = Lambda_y_S*(eps2_y*(W_3(seqy_m1, seqx)-W_3(seqy, seqx)) - eps4_y*(W_3(seqy_m2, seqx)-3*W_3(seqy_m1, seqx)+3*W_3(seqy, seqx)-W_3(seqy_p1, seqx)));
  

    // // Boundary conditions -----------------------------------------------to test if effective -----------------------------------------------
    // // Calculate dissipx for cells (3, i)
    // dissipy_0.row(1) = Lambda_y_S.row(1)*(eps2_y.row(1)*(W_0(2,seqx)-W_0(3,seqx)) - eps4_y.row(1)*(2*W_0(3,seqx) - W_0(2,seqx) - W_0(4,seqx)));
    // dissipy_1.row(1) = Lambda_y_S.row(1)*(eps2_y.row(1)*(W_1(2,seqx)-W_1(3,seqx)) - eps4_y.row(1)*(2*W_1(3,seqx) - W_1(2,seqx) - W_1(4,seqx)));
    // dissipy_2.row(1) = Lambda_y_S.row(1)*(eps2_y.row(1)*(W_2(2,seqx)-W_2(3,seqx)) - eps4_y.row(1)*(2*W_2(3,seqx) - W_2(2,seqx) - W_2(4,seqx)));
    // dissipy_3.row(1) = Lambda_y_S.row(1)*(eps2_y.row(1)*(W_3(2,seqx)-W_3(3,seqx)) - eps4_y.row(1)*(2*W_3(3,seqx) - W_3(2,seqx) - W_3(4,seqx)));

    // // Calculate dissipx for cells (2, i)
    // dissipy_0.row(0) = Lambda_y_S.row(0)*(eps2_y.row(0)*(W_0(2,seqx)-W_0(3,seqx)) - eps4_y.row(0)*(2*W_0(3,seqx) - W_0(2,seqx) - W_0(4,seqx)));
    // dissipy_1.row(0) = Lambda_y_S.row(0)*(eps2_y.row(0)*(W_1(2,seqx)-W_1(3,seqx)) - eps4_y.row(0)*(2*W_1(3,seqx) - W_1(2,seqx) - W_1(4,seqx)));
    // dissipy_2.row(0) = Lambda_y_S.row(0)*(eps2_y.row(0)*(W_2(2,seqx)-W_2(3,seqx)) - eps4_y.row(0)*(2*W_2(3,seqx) - W_2(2,seqx) - W_2(4,seqx)));
    // dissipy_3.row(0) = Lambda_y_S.row(0)*(eps2_y.row(0)*(W_3(2,seqx)-W_3(3,seqx)) - eps4_y.row(0)*(2*W_3(3,seqx) - W_3(2,seqx) - W_3(4,seqx)));
}

void SpatialDiscretization::compute_Rc() {
    int im1 = fluxx_0.cols()-2;
    int jm1 = fluxx_0.rows()-2;
    auto seqy = Eigen::seq(0, jm1);
    auto seqx = Eigen::seq(0, im1);
    auto seqy_p1 = Eigen::seq(1, jm1+1);
    auto seqx_p1 = Eigen::seq(1, im1+1);

    Rc_0 = fluxx_0(seqy, seqx) - fluxx_0(seqy, seqx_p1) + fluxy_0(seqy, seqx) - fluxy_0(seqy_p1, seqx);
    Rc_1 = fluxx_1(seqy, seqx) - fluxx_1(seqy, seqx_p1) + fluxy_1(seqy, seqx) - fluxy_1(seqy_p1, seqx);
    Rc_2 = fluxx_2(seqy, seqx) - fluxx_2(seqy, seqx_p1) + fluxy_2(seqy, seqx) - fluxy_2(seqy_p1, seqx);
    Rc_3 = fluxx_3(seqy, seqx) - fluxx_3(seqy, seqx_p1) + fluxy_3(seqy, seqx) - fluxy_3(seqy_p1, seqx);
}

void SpatialDiscretization::compute_Rd() {
    int im1 = dissipx_0.cols()-2;
    int jm1 = dissipx_0.rows()-2;
    auto seqy = Eigen::seq(0, jm1);
    auto seqx = Eigen::seq(0, im1);
    auto seqy_p1 = Eigen::seq(1, jm1+1);
    auto seqx_p1 = Eigen::seq(1, im1+1);

    Rd_0 = dissipx_0(seqy, seqx) - dissipx_0(seqy, seqx_p1) + dissipy_0(seqy, seqx) - dissipy_0(seqy_p1, seqx);
    Rd_1 = dissipx_1(seqy, seqx) - dissipx_1(seqy, seqx_p1) + dissipy_1(seqy, seqx) - dissipy_1(seqy_p1, seqx);
    Rd_2 = dissipx_2(seqy, seqx) - dissipx_2(seqy, seqx_p1) + dissipy_2(seqy, seqx) - dissipy_2(seqy_p1, seqx);
    Rd_3 = dissipx_3(seqy, seqx) - dissipx_3(seqy, seqx_p1) + dissipy_3(seqy, seqx) - dissipy_3(seqy_p1, seqx);
}

void SpatialDiscretization::update_Rd0() {
    Rd0_0 = Rd_0;
    Rd0_1 = Rd_1;
    Rd0_2 = Rd_2;
    Rd0_3 = Rd_3;
}

void SpatialDiscretization::run_odd() {
    SpatialDiscretization::compute_dummy_cells();
    SpatialDiscretization::update_halo();
    SpatialDiscretization::update_W();
    // SpatialDiscretization::compute_lambda();
    SpatialDiscretization::compute_flux();
    SpatialDiscretization::compute_Rc();
}

void SpatialDiscretization::run_even() {
    SpatialDiscretization::compute_dummy_cells();
    SpatialDiscretization::update_halo();
    SpatialDiscretization::update_W();
    SpatialDiscretization::compute_lambda();
    SpatialDiscretization::compute_flux();
    SpatialDiscretization::compute_dissipation();
    SpatialDiscretization::compute_Rc();
    SpatialDiscretization::compute_Rd();
}

