#include "SpatialDiscretization.h"
#include "mesh.h"
#include "helper.h"
#include <iostream>
#include <cmath>
#include <vector>

SpatialDiscretization::SpatialDiscretization(Mesh &mesh, double Mach, double alpha, double k2, double k4)
    : mesh_(mesh), Mach_(Mach), alpha_(alpha), k2_(k2), k4_(k4) {

    ncells_ = mesh_.ncells;
    nfaces_ = static_cast<int>(mesh_.faces.size());

    uInfty_ = Mach_ * cInfty_ * std::cos(alpha_ * M_PI / 180.0);
    vInfty_ = Mach_ * cInfty_ * std::sin(alpha_ * M_PI / 180.0);
    EInfty_ = pInfty_ / ((gamma_ - 1.0) * rhoInfty_) + 0.5 * (uInfty_ * uInfty_ + vInfty_ * vInfty_);

    // Resize all vectors to number of cells or faces
    W0.resize(ncells_); W1.resize(ncells_); W2.resize(ncells_); W3.resize(ncells_);
    rhorho.resize(ncells_); uu.resize(ncells_); vv.resize(ncells_); EE.resize(ncells_); pp.resize(ncells_);
    Rc0.resize(ncells_); Rc1.resize(ncells_); Rc2.resize(ncells_); Rc3.resize(ncells_);
    Rd0.resize(ncells_); Rd1.resize(ncells_); Rd2.resize(ncells_); Rd3.resize(ncells_);
    LambdaI.resize(ncells_); LambdaJ.resize(ncells_);

    F0.resize(nfaces_); F1.resize(nfaces_); F2.resize(nfaces_); F3.resize(nfaces_);
    D0.resize(nfaces_); D1.resize(nfaces_); D2.resize(nfaces_); D3.resize(nfaces_);
}

void SpatialDiscretization::initializeVariables() {
    for (int cell = 0; cell < ncells_; ++cell) {
        W0[cell] = rhoInfty_;
        W1[cell] = rhoInfty_ * uInfty_;
        W2[cell] = rhoInfty_ * vInfty_;
        W3[cell] = rhoInfty_ * EInfty_;

        rhorho[cell] = rhoInfty_;
        uu[cell] = uInfty_;
        vv[cell] = vInfty_;
        EE[cell] = EInfty_;
        pp[cell] = pInfty_;
    }

}

void SpatialDiscretization::updateGhostCells() {
    std::cout << "face, ghostCell, u_mirror, v_mirror, u_ghost, v_ghost, cx, cy\n";
    // copy flow variables to ghost cells using immersed boundary conditions
    std::vector<double> rho_copy = rhorho;
    std::vector<double> u_copy = uu;
    std::vector<double> v_copy = vv;
    std::vector<double> E_copy = EE;
    std::vector<double> p_copy = pp;
    for (auto face : mesh_.immersedBoundaryFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int leftCellType = mesh_.cell_types[leftCell];
        int ghostCell = (leftCellType == 0) ? leftCell : rightCell;
        double rho_mirror = 0.0, u_mirror = 0.0, v_mirror = 0.0, E_mirror = 0.0, p_mirror = 0.0;
        double total_weight = 0.0;
        // std::cout << "Updating ghost cell " << ghostCell << " for face " << face << ":\n";
        for (size_t k = 0; k < mesh_.faces[face].adjacentCells.size(); ++k) {
            int adjCell = mesh_.faces[face].adjacentCells[k];
            double dist = mesh_.faces[face].adjacentDistances[k];
            double weight = 1.0 / dist;
            rho_mirror += rho_copy[adjCell] * weight;
            u_mirror += u_copy[adjCell] * weight;
            v_mirror += v_copy[adjCell] * weight;
            E_mirror += E_copy[adjCell] * weight;
            p_mirror += p_copy[adjCell] * weight;
            total_weight += weight;
            // std::cout << "  adjacent cell: " << adjCell << ", dist: " << dist << ", weight: " << weight << " used values: rho=" << rho_copy[adjCell]
            //           << ", u=" << u_copy[adjCell] << ", v=" << v_copy[adjCell]
            //           << ", E=" << E_copy[adjCell] << ", p=" << p_copy[adjCell] << "\n";
        }
        rho_mirror /= total_weight;
        u_mirror /= total_weight;
        v_mirror /= total_weight;
        E_mirror /= total_weight;
        p_mirror /= total_weight;
        // std::cout << "  mirrored values before BCs: rho=" << rho_mirror << ", u=" << u_mirror 
        //           << ", v=" << v_mirror << ", E=" << E_mirror << ", p=" << p_mirror << "\n";

        uu[ghostCell] = u_mirror - 2.0 * (u_mirror * mesh_.faces[face].ib_nx + v_mirror * mesh_.faces[face].ib_ny) * mesh_.faces[face].ib_nx;
        vv[ghostCell] = v_mirror - 2.0 * (u_mirror * mesh_.faces[face].ib_nx + v_mirror * mesh_.faces[face].ib_ny) * mesh_.faces[face].ib_ny;
        rhorho[ghostCell] = rho_mirror;
        pp[ghostCell] = p_mirror;
        EE[ghostCell] = p_mirror / ((gamma_ - 1.0) * rho_mirror) + 0.5 * (uu[ghostCell] * uu[ghostCell] + vv[ghostCell] * vv[ghostCell]);
        
        W0[ghostCell] = rhorho[ghostCell];
        W1[ghostCell] = rhorho[ghostCell] * uu[ghostCell];
        W2[ghostCell] = rhorho[ghostCell] * vv[ghostCell];
        W3[ghostCell] = rhorho[ghostCell] * EE[ghostCell];

        std::cout << face << ", " << ghostCell << ", " << u_mirror << ", " << v_mirror 
                << ", " << uu[ghostCell] << ", " << vv[ghostCell] 
                << ", " << mesh_.cx[ghostCell] << ", " << mesh_.cy[ghostCell] << "\n";
    }
}

std::tuple<double, double, double, double> SpatialDiscretization::compute_conservative_fluxes_IB(int fluidCell, int fluidCell_p1, int fluidCell_p2, 
                                        double area, double ib_nx, double ib_ny) {
    double uwall = 0.125*(15*this->uu[fluidCell] - 10*this->uu[fluidCell_p1] + 3*this->uu[fluidCell_p2]); // 2nd order extrapolation
    double vwall = 0.125*(15*this->vv[fluidCell] - 10*this->vv[fluidCell_p1] + 3*this->vv[fluidCell_p2]); // 2nd order extrapolation
    double rhowall = 0.125*(15*this->rhorho[fluidCell] - 10*this->rhorho[fluidCell_p1] + 3*this->rhorho[fluidCell_p2]); // 2nd order extrapolation
    double Ewall = 0.125*(15*this->EE[fluidCell] - 10*this->EE[fluidCell_p1] + 3*this->EE[fluidCell_p2]); // 2nd order extrapolation
    double pwall = 0.125*(15*this->pp[fluidCell] - 10*this->pp[fluidCell_p1] + 3*this->pp[fluidCell_p2]); // 2nd order extrapolation

    double V = uwall * ib_nx + vwall * ib_ny;
    double H = Ewall + pwall / rhowall;
    double F0 = rhowall * V * area;
    double F1 = (rhowall*uwall*V + pwall*ib_nx) * area;
    double F2 = (rhowall*vwall*V + pwall*ib_ny) * area;
    double F3 = (rhowall*H*V) * area;

    return {F0, F1, F2, F3};
}

void SpatialDiscretization::compute_convective_fluxes() {
    for (auto face : mesh_.fluidFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        double nx = mesh_.faces[face].nx;
        double ny = mesh_.faces[face].ny;
        double area = mesh_.faces[face].area;

        double avg_W0 = 0.5 * (W0[leftCell] + W0[rightCell]);
        double avg_W1 = 0.5 * (W1[leftCell] + W1[rightCell]);
        double avg_W2 = 0.5 * (W2[leftCell] + W2[rightCell]);
        double avg_W3 = 0.5 * (W3[leftCell] + W3[rightCell]);

        double avg_u = avg_W1 / avg_W0;
        double avg_v = avg_W2 / avg_W0;
        double avg_E = avg_W3 / avg_W0;
        double avg_p = pressure(gamma_, avg_W0, avg_u, avg_v, avg_E);

        // Compute fluxes
        double V = avg_u * nx + avg_v * ny;
        F0[face] = avg_W0 * V * area;
        F1[face] = (avg_W1 * V + avg_p * nx) * area;
        F2[face] = (avg_W2 * V + avg_p * ny) * area;
        F3[face] = (avg_W3 * V + avg_p * V) * area;
    }
    // std::cout << "face, leftCell, rightCell, cx, cy, u_ib, v_ib\n";
    for (auto face : mesh_.immersedBoundaryFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        double nx = mesh_.faces[face].nx;
        double ny = mesh_.faces[face].ny;
        double area = mesh_.faces[face].area;

        double avg_W0 = 0.5 * (W0[leftCell] + W0[rightCell]);
        double avg_W1 = 0.5 * (W1[leftCell] + W1[rightCell]);
        double avg_W2 = 0.5 * (W2[leftCell] + W2[rightCell]);
        double avg_W3 = 0.5 * (W3[leftCell] + W3[rightCell]);

        double avg_u = avg_W1 / avg_W0;
        double avg_v = avg_W2 / avg_W0;
        double avg_E = avg_W3 / avg_W0;
        double avg_p = pressure(gamma_, avg_W0, avg_u, avg_v, avg_E);

        // std::cout << face << ", " << leftCell << ", " << rightCell << ", "
        //           << mesh_.faces[face].cx << ", " << mesh_.faces[face].cy << ", "
        //           << avg_u << ", " << avg_v << "\n";

        // Compute fluxes
        double V = avg_u * nx + avg_v * ny;
        F0[face] = avg_W0 * V * area;
        F1[face] = (avg_W1 * V + avg_p * nx) * area;
        F2[face] = (avg_W2 * V + avg_p * ny) * area;
        F3[face] = (avg_W3 * V + avg_p * V) * area;
    }

    // for (auto face : mesh_.ibFacesX_m1) {
    //     int leftCell = mesh_.faces[face].leftCell;
    //     int rightCell = mesh_.faces[face].rightCell;
    //     int leftCellType = mesh_.cell_types[leftCell];
    //     double ib_nx = mesh_.faces[face].ib_nx; // use immersed boundary normal
    //     double ib_ny = mesh_.faces[face].ib_ny;

    //     int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
    //     int fluidCell_p1 = fluidCell + 1;
    //     int fluidCell_p2 = fluidCell + 2;

    //     auto [F0_face, F1_face, F2_face, F3_face] = compute_conservative_fluxes_IB(fluidCell, fluidCell_p1, fluidCell_p2,
    //                                                                                  mesh_.faces[face].area, ib_nx, ib_ny);
    //     F0[face] = F0_face;
    //     F1[face] = F1_face;
    //     F2[face] = F2_face;
    //     F3[face] = F3_face;
    // }

    // for (auto face : mesh_.ibFacesX_p1) {
    //     int leftCell = mesh_.faces[face].leftCell;
    //     int rightCell = mesh_.faces[face].rightCell;
    //     int leftCellType = mesh_.cell_types[leftCell];
    //     double ib_nx = mesh_.faces[face].ib_nx; // use immersed boundary normal
    //     double ib_ny = mesh_.faces[face].ib_ny;

    //     int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
    //     int fluidCell_p1 = fluidCell - 1;
    //     int fluidCell_p2 = fluidCell - 2;

    //     auto [F0_face, F1_face, F2_face, F3_face] = compute_conservative_fluxes_IB(fluidCell, fluidCell_p1, fluidCell_p2,
    //                                                                                  mesh_.faces[face].area, ib_nx, ib_ny);
    //     F0[face] = F0_face;
    //     F1[face] = F1_face;
    //     F2[face] = F2_face;
    //     F3[face] = F3_face;
    // }

    // for (auto face : mesh_.ibFacesY_m1) {
    //     int leftCell = mesh_.faces[face].leftCell;
    //     int rightCell = mesh_.faces[face].rightCell;
    //     int leftCellType = mesh_.cell_types[leftCell];
    //     double ib_nx = mesh_.faces[face].ib_nx; // use immersed boundary normal
    //     double ib_ny = mesh_.faces[face].ib_ny;

    //     int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
    //     int fluidCell_p1 = fluidCell + (mesh_.ni-1);
    //     int fluidCell_p2 = fluidCell + 2*(mesh_.ni-1);

    //     auto [F0_face, F1_face, F2_face, F3_face] = compute_conservative_fluxes_IB(fluidCell, fluidCell_p1, fluidCell_p2,
    //                                                                                  mesh_.faces[face].area, ib_nx, ib_ny);
    //     F0[face] = F0_face;
    //     F1[face] = F1_face;
    //     F2[face] = F2_face;
    //     F3[face] = F3_face;
    // }

    // for (auto face : mesh_.ibFacesY_p1) {
    //     int leftCell = mesh_.faces[face].leftCell;
    //     int rightCell = mesh_.faces[face].rightCell;
    //     int leftCellType = mesh_.cell_types[leftCell];
    //     double ib_nx = mesh_.faces[face].ib_nx; // use immersed boundary normal
    //     double ib_ny = mesh_.faces[face].ib_ny;

    //     int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
    //     int fluidCell_p1 = fluidCell - (mesh_.ni-1);
    //     int fluidCell_p2 = fluidCell - 2*(mesh_.ni-1);

    //     auto [F0_face, F1_face, F2_face, F3_face] = compute_conservative_fluxes_IB(fluidCell, fluidCell_p1, fluidCell_p2,
    //                                                                                  mesh_.faces[face].area, ib_nx, ib_ny);
    //     F0[face] = F0_face;
    //     F1[face] = F1_face;
    //     F2[face] = F2_face;
    //     F3[face] = F3_face;
    // }

    for (auto face : mesh_.farfieldFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int leftCellType = mesh_.cell_types[leftCell];
        double nx = mesh_.faces[face].nx;
        double ny = mesh_.faces[face].ny;
        double area = mesh_.faces[face].area;
        int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
        double machNumber = mach(gamma_, rhorho[fluidCell], pp[fluidCell], uu[fluidCell], vv[fluidCell]);
        if (machNumber >=1.0) {
            if (uu[fluidCell]*nx + vv[fluidCell]*ny >= 0) {
                // outflow
                double avg_W0 = W0[leftCell];
                double avg_W1 = W1[leftCell];
                double avg_W2 = W2[leftCell];
                double avg_W3 = W3[leftCell];

                double avg_u = avg_W1 / avg_W0;
                double avg_v = avg_W2 / avg_W0;
                double avg_E = avg_W3 / avg_W0;
                double avg_p = pressure(gamma_, avg_W0, avg_u, avg_v, avg_E);

                // Compute fluxes
                double V = avg_u * nx + avg_v * ny;
                F0[face] = avg_W0 * V * area;
                F1[face] = (avg_W1 * V + avg_p * nx) * area;
                F2[face] = (avg_W2 * V + avg_p * ny) * area;
                F3[face] = (avg_W3 * V + avg_p * V) * area;
                // std::cout << "  Outflow: V=" << V << std::endl;
            } else {
                // inflow
                double V = uInfty_ * nx + vInfty_ * ny;
                double H = EInfty_ + pInfty_ / rhoInfty_;
                F0[face] = rhoInfty_ * V * area;
                F1[face] = (rhoInfty_*uInfty_*V + pInfty_*nx) * area;
                F2[face] = (rhoInfty_*vInfty_*V + pInfty_*ny) * area;
                F3[face] = (rhoInfty_*H*V) * area;
                // std::cout << "  Inflow: V=" << V << ", H=" << H << std::endl;
            }
        }
        else if (machNumber < 1.0) {
            double c = std::sqrt(gamma_ * pp[fluidCell] / rhorho[fluidCell]);
            if (uu[fluidCell]*nx + vv[fluidCell]*ny >= 0) {
                // outflow
                double pb = pInfty_;
                double rhob = rhorho[fluidCell] + (pb - pp[fluidCell]) / (c*c);
                double ub = uu[fluidCell] + nx*(pp[fluidCell] - pb)/(rhorho[fluidCell]*c);
                double vb = vv[fluidCell] + ny*(pp[fluidCell] - pb)/(rhorho[fluidCell]*c);
                double Eb = pb / ((gamma_ - 1.0) * rhob) + 0.5 * (ub * ub + vb * vb);

                double V = ub * nx + vb * ny;
                double H = Eb + pb / rhob;
                F0[face] = rhob * V * area;
                F1[face] = (rhob*ub*V + pb*nx) * area;
                F2[face] = (rhob*vb*V + pb*ny) * area;
                F3[face] = (rhob*H*V) * area;
            } else {
                // inflow
                double pb = 0.5*(pInfty_ + pp[fluidCell] - rhorho[fluidCell]*c*(nx*(uInfty_-uu[fluidCell]) + ny*(vInfty_-vv[fluidCell])));
                double rhob = rhoInfty_ + (pb - pInfty_) / (c*c);
                double ub = uInfty_ - nx*(pInfty_ - pb)/(rhorho[fluidCell]*c);
                double vb = vInfty_ - ny*(pInfty_ - pb)/(rhorho[fluidCell]*c);
                double Eb = pb / ((gamma_ - 1.0) * rhob) + 0.5 * (ub * ub + vb * vb);

                double V = ub * nx + vb * ny;
                double H = Eb + pb / rhob;
                F0[face] = rhob * V * area;
                F1[face] = (rhob*ub*V + pb*nx) * area;
                F2[face] = (rhob*vb*V + pb*ny) * area;
                F3[face] = (rhob*H*V) * area;
            }
        }
    }


}

void SpatialDiscretization::compute_lambdas() {
    for (auto cell : mesh_.fluidCells) {
        double u = uu[cell];
        double v = vv[cell];
        double p = pp[cell];
        double rho = rhorho[cell];
        double a = std::sqrt(gamma_ * p / rho);

        LambdaI[cell] = (fabs(u) + a)*mesh_.avg_face_area_x[cell];
        LambdaJ[cell] = (fabs(v) + a)*mesh_.avg_face_area_y[cell];
    }
}

void SpatialDiscretization::compute_diffusive_fluxes() {
    for (auto face : mesh_.fluidFacesX) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - 1;
        int rightCell = mesh_.faces[face].rightCell;
        int rightCell_p1 = rightCell + 1;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], pp[rightCell], pp[rightCell_p1]);
        double lambdaS = 0.5*(LambdaI[leftCell] + LambdaI[rightCell]) + 0.5*(LambdaJ[leftCell] + LambdaJ[rightCell]);
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0[leftCell]) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1[leftCell]) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2[leftCell]) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3[leftCell]) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.fluidFacesX_m1) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int rightCell_p1 = rightCell + 1;
        auto [eps2, eps4] = epsilon(pp[leftCell] - (pp[rightCell]-pp[leftCell]), pp[leftCell], pp[rightCell], pp[rightCell_p1]);
        double lambdaS = 0.5*(LambdaI[leftCell] + LambdaI[rightCell]) + 0.5*(LambdaJ[leftCell] + LambdaJ[rightCell]);
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0[leftCell]) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0[leftCell]-(W0[leftCell] - (W0[rightCell]-W0[leftCell]))));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1[leftCell]) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1[leftCell]-(W1[leftCell] - (W1[rightCell]-W1[leftCell]))));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2[leftCell]) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2[leftCell]-(W2[leftCell] - (W2[rightCell]-W2[leftCell]))));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3[leftCell]) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3[leftCell]-(W3[leftCell] - (W3[rightCell]-W3[leftCell]))));
    }

    for (auto face : mesh_.fluidFacesX_p1) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - 1;
        int rightCell = mesh_.faces[face].rightCell;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], pp[rightCell], pp[rightCell] + (pp[rightCell]-pp[leftCell]));
        double lambdaS = 0.5*(LambdaI[leftCell] + LambdaI[rightCell]) + 0.5*(LambdaJ[leftCell] + LambdaJ[rightCell]);
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0[leftCell]) - eps4*((W0[rightCell] + (W0[rightCell]-W0[leftCell]))-3.0*W0[rightCell]+3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1[leftCell]) - eps4*((W1[rightCell] + (W1[rightCell]-W1[leftCell]))-3.0*W1[rightCell]+3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2[leftCell]) - eps4*((W2[rightCell] + (W2[rightCell]-W2[leftCell]))-3.0*W2[rightCell]+3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3[leftCell]) - eps4*((W3[rightCell] + (W3[rightCell]-W3[leftCell]))-3.0*W3[rightCell]+3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.fluidFacesY) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - (mesh_.ni-1);
        int rightCell = mesh_.faces[face].rightCell;
        int rightCell_p1 = rightCell + (mesh_.ni-1);
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], pp[rightCell], pp[rightCell_p1]);
        double lambdaS = 0.5*(LambdaI[leftCell] + LambdaI[rightCell]) + 0.5*(LambdaJ[leftCell] + LambdaJ[rightCell]);
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0[leftCell]) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1[leftCell]) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2[leftCell]) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3[leftCell]) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.fluidFacesY_m1) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int rightCell_p1 = rightCell + (mesh_.ni-1);
        auto [eps2, eps4] = epsilon(pp[leftCell] - (pp[rightCell]-pp[leftCell]), pp[leftCell], pp[rightCell], pp[rightCell_p1]);
        double lambdaS = 0.5*(LambdaI[leftCell] + LambdaI[rightCell]) + 0.5*(LambdaJ[leftCell] + LambdaJ[rightCell]);
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0[leftCell]) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0[leftCell]-(W0[leftCell] - (W0[rightCell]-W0[leftCell]))));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1[leftCell]) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1[leftCell]-(W1[leftCell] - (W1[rightCell]-W1[leftCell]))));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2[leftCell]) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2[leftCell]-(W2[leftCell] - (W2[rightCell]-W2[leftCell]))));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3[leftCell]) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3[leftCell]-(W3[leftCell] - (W3[rightCell]-W3[leftCell]))));
    }

    for (auto face : mesh_.fluidFacesY_p1) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - (mesh_.ni-1);
        int rightCell = mesh_.faces[face].rightCell;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], pp[rightCell], pp[rightCell] + (pp[rightCell]-pp[leftCell]));
        double lambdaS = 0.5*(LambdaI[leftCell] + LambdaI[rightCell]) + 0.5*(LambdaJ[leftCell] + LambdaJ[rightCell]);
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0[leftCell]) - eps4*((W0[rightCell] + (W0[rightCell]-W0[leftCell]))-3.0*W0[rightCell]+3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1[leftCell]) - eps4*((W1[rightCell] + (W1[rightCell]-W1[leftCell]))-3.0*W1[rightCell]+3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2[leftCell]) - eps4*((W2[rightCell] + (W2[rightCell]-W2[leftCell]))-3.0*W2[rightCell]+3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3[leftCell]) - eps4*((W3[rightCell] + (W3[rightCell]-W3[leftCell]))-3.0*W3[rightCell]+3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.farfieldFacesX_m1) {
        int rightCell = mesh_.faces[face].leftCell;
        int rightCell_p1 = rightCell + 1;
        double delta_p = pp[rightCell_p1] - pp[rightCell];
        double p_left = pp[rightCell] - delta_p;
        double p_left_m1 = p_left - delta_p;
        auto [eps2, eps4] = epsilon(p_left_m1, p_left, pp[rightCell], pp[rightCell_p1]);
        double lambdaS = LambdaI[rightCell] + LambdaJ[rightCell];
        double delta_W0 = W0[rightCell_p1] - W0[rightCell];
        double delta_W1 = W1[rightCell_p1] - W1[rightCell];
        double delta_W2 = W2[rightCell_p1] - W2[rightCell];
        double delta_W3 = W3[rightCell_p1] - W3[rightCell];
        double W0_left = W0[rightCell] - delta_W0;
        double W1_left = W1[rightCell] - delta_W1;
        double W2_left = W2[rightCell] - delta_W2;
        double W3_left = W3[rightCell] - delta_W3;
        double W0_left_m1 = W0_left - delta_W0;
        double W1_left_m1 = W1_left - delta_W1;
        double W2_left_m1 = W2_left - delta_W2;
        double W3_left_m1 = W3_left - delta_W3;
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0_left) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0_left-W0_left_m1));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1_left) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1_left-W1_left_m1));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2_left) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2_left-W2_left_m1));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3_left) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3_left-W3_left_m1));
    }

    for (auto face : mesh_.farfieldFacesX_p1) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - 1;
        double delta_p = pp[leftCell] - pp[leftCell_m1];
        double p_right = pp[leftCell] + delta_p;
        double p_right_p1 = p_right + delta_p;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], p_right, p_right_p1);
        double lambdaS = LambdaI[leftCell] + LambdaJ[leftCell];
        double delta_W0 = W0[leftCell] - W0[leftCell_m1];
        double delta_W1 = W1[leftCell] - W1[leftCell_m1];
        double delta_W2 = W2[leftCell] - W2[leftCell_m1];
        double delta_W3 = W3[leftCell] - W3[leftCell_m1];
        double W0_right = W0[leftCell] + delta_W0;
        double W1_right = W1[leftCell] + delta_W1;
        double W2_right = W2[leftCell] + delta_W2;
        double W3_right = W3[leftCell] + delta_W3;
        double W0_right_p1 = W0_right + delta_W0;
        double W1_right_p1 = W1_right + delta_W1;
        double W2_right_p1 = W2_right + delta_W2;
        double W3_right_p1 = W3_right + delta_W3;
        D0[face] = lambdaS*(eps2*(W0_right - W0[leftCell]) - eps4*(W0_right_p1 - 3.0*W0_right + 3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1_right - W1[leftCell]) - eps4*(W1_right_p1 - 3.0*W1_right + 3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2_right - W2[leftCell]) - eps4*(W2_right_p1 - 3.0*W2_right + 3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3_right - W3[leftCell]) - eps4*(W3_right_p1 - 3.0*W3_right + 3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.farfieldFacesY_m1) {
        int rightCell = mesh_.faces[face].leftCell;
        int rightCell_p1 = rightCell + (mesh_.ni-1);
        double delta_p = pp[rightCell_p1] - pp[rightCell];
        double p_left = pp[rightCell] - delta_p;
        double p_left_m1 = p_left - delta_p;
        auto [eps2, eps4] = epsilon(p_left_m1, p_left, pp[rightCell], pp[rightCell_p1]);
        double lambdaS = LambdaI[rightCell] + LambdaJ[rightCell];
        double delta_W0 = W0[rightCell_p1] - W0[rightCell];
        double delta_W1 = W1[rightCell_p1] - W1[rightCell];
        double delta_W2 = W2[rightCell_p1] - W2[rightCell];
        double delta_W3 = W3[rightCell_p1] - W3[rightCell];
        double W0_left = W0[rightCell] - delta_W0;
        double W1_left = W1[rightCell] - delta_W1;
        double W2_left = W2[rightCell] - delta_W2;
        double W3_left = W3[rightCell] - delta_W3;
        double W0_left_m1 = W0_left - delta_W0;
        double W1_left_m1 = W1_left - delta_W1;
        double W2_left_m1 = W2_left - delta_W2;
        double W3_left_m1 = W3_left - delta_W3;
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0_left) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0_left-W0_left_m1));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1_left) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1_left-W1_left_m1));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2_left) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2_left-W2_left_m1));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3_left) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3_left-W3_left_m1));
    }

    for (auto face : mesh_.farfieldFacesY_p1) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - (mesh_.ni-1);
        double delta_p = pp[leftCell] - pp[leftCell_m1];
        double p_right = pp[leftCell] + delta_p;
        double p_right_p1 = p_right + delta_p;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], p_right, p_right_p1);
        double lambdaS = LambdaI[leftCell] + LambdaJ[leftCell];
        double delta_W0 = W0[leftCell] - W0[leftCell_m1];
        double delta_W1 = W1[leftCell] - W1[leftCell_m1];
        double delta_W2 = W2[leftCell] - W2[leftCell_m1];
        double delta_W3 = W3[leftCell] - W3[leftCell_m1];
        double W0_right = W0[leftCell] + delta_W0;
        double W1_right = W1[leftCell] + delta_W1;
        double W2_right = W2[leftCell] + delta_W2;
        double W3_right = W3[leftCell] + delta_W3;
        double W0_right_p1 = W0_right + delta_W0;
        double W1_right_p1 = W1_right + delta_W1;
        double W2_right_p1 = W2_right + delta_W2;
        double W3_right_p1 = W3_right + delta_W3;
        D0[face] = lambdaS*(eps2*(W0_right - W0[leftCell]) - eps4*(W0_right_p1 - 3.0*W0_right + 3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1_right - W1[leftCell]) - eps4*(W1_right_p1 - 3.0*W1_right + 3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2_right - W2[leftCell]) - eps4*(W2_right_p1 - 3.0*W2_right + 3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3_right - W3[leftCell]) - eps4*(W3_right_p1 - 3.0*W3_right + 3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.ibFacesX_m1) {
        int rightCell = mesh_.faces[face].rightCell;
        int rightCell_p1 = rightCell + 1;
        double delta_p = pp[rightCell_p1] - pp[rightCell];
        double p_left = pp[rightCell] - delta_p;
        double p_left_m1 = p_left - delta_p;
        auto [eps2, eps4] = epsilon(p_left_m1, p_left, pp[rightCell], pp[rightCell_p1]);
        double lambdaS = LambdaI[rightCell] + LambdaJ[rightCell];
        double delta_W0 = W0[rightCell_p1] - W0[rightCell];
        double delta_W1 = W1[rightCell_p1] - W1[rightCell];
        double delta_W2 = W2[rightCell_p1] - W2[rightCell];
        double delta_W3 = W3[rightCell_p1] - W3[rightCell];
        double W0_left = W0[rightCell] - delta_W0;
        double W1_left = W1[rightCell] - delta_W1;
        double W2_left = W2[rightCell] - delta_W2;
        double W3_left = W3[rightCell] - delta_W3;
        double W0_left_m1 = W0_left - delta_W0;
        double W1_left_m1 = W1_left - delta_W1;
        double W2_left_m1 = W2_left - delta_W2;
        double W3_left_m1 = W3_left - delta_W3;
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0_left) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0_left-W0_left_m1));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1_left) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1_left-W1_left_m1));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2_left) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2_left-W2_left_m1));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3_left) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3_left-W3_left_m1));
    }

    for (auto face : mesh_.ibFacesX_p1) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - 1;
        double delta_p = pp[leftCell] - pp[leftCell_m1];
        double p_right = pp[leftCell] + delta_p;
        double p_right_p1 = p_right + delta_p;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], p_right, p_right_p1);
        double lambdaS = LambdaI[leftCell] + LambdaJ[leftCell];
        double delta_W0 = W0[leftCell] - W0[leftCell_m1];
        double delta_W1 = W1[leftCell] - W1[leftCell_m1];
        double delta_W2 = W2[leftCell] - W2[leftCell_m1];
        double delta_W3 = W3[leftCell] - W3[leftCell_m1];
        double W0_right = W0[leftCell] + delta_W0;
        double W1_right = W1[leftCell] + delta_W1;
        double W2_right = W2[leftCell] + delta_W2;
        double W3_right = W3[leftCell] + delta_W3;
        double W0_right_p1 = W0_right + delta_W0;
        double W1_right_p1 = W1_right + delta_W1;
        double W2_right_p1 = W2_right + delta_W2;
        double W3_right_p1 = W3_right + delta_W3;
        D0[face] = lambdaS*(eps2*(W0_right - W0[leftCell]) - eps4*(W0_right_p1 - 3.0*W0_right + 3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1_right - W1[leftCell]) - eps4*(W1_right_p1 - 3.0*W1_right + 3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2_right - W2[leftCell]) - eps4*(W2_right_p1 - 3.0*W2_right + 3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3_right - W3[leftCell]) - eps4*(W3_right_p1 - 3.0*W3_right + 3.0*W3[leftCell]-W3[leftCell_m1]));
    }

    for (auto face : mesh_.ibFacesY_m1) {
        int rightCell = mesh_.faces[face].rightCell;
        int rightCell_p1 = rightCell + (mesh_.ni-1);
        double delta_p = pp[rightCell_p1] - pp[rightCell];
        double p_left = pp[rightCell] - delta_p;
        double p_left_m1 = p_left - delta_p;
        auto [eps2, eps4] = epsilon(p_left_m1, p_left, pp[rightCell], pp[rightCell_p1]);
        double lambdaS = LambdaI[rightCell] + LambdaJ[rightCell];
        double delta_W0 = W0[rightCell_p1] - W0[rightCell];
        double delta_W1 = W1[rightCell_p1] - W1[rightCell];
        double delta_W2 = W2[rightCell_p1] - W2[rightCell];
        double delta_W3 = W3[rightCell_p1] - W3[rightCell];
        double W0_left = W0[rightCell] - delta_W0;
        double W1_left = W1[rightCell] - delta_W1;
        double W2_left = W2[rightCell] - delta_W2;
        double W3_left = W3[rightCell] - delta_W3;
        double W0_left_m1 = W0_left - delta_W0;
        double W1_left_m1 = W1_left - delta_W1;
        double W2_left_m1 = W2_left - delta_W2;
        double W3_left_m1 = W3_left - delta_W3;
        D0[face] = lambdaS*(eps2*(W0[rightCell]-W0_left) - eps4*(W0[rightCell_p1]-3.0*W0[rightCell]+3.0*W0_left-W0_left_m1));
        D1[face] = lambdaS*(eps2*(W1[rightCell]-W1_left) - eps4*(W1[rightCell_p1]-3.0*W1[rightCell]+3.0*W1_left-W1_left_m1));
        D2[face] = lambdaS*(eps2*(W2[rightCell]-W2_left) - eps4*(W2[rightCell_p1]-3.0*W2[rightCell]+3.0*W2_left-W2_left_m1));
        D3[face] = lambdaS*(eps2*(W3[rightCell]-W3_left) - eps4*(W3[rightCell_p1]-3.0*W3[rightCell]+3.0*W3_left-W3_left_m1));
    }

    for (auto face : mesh_.ibFacesY_p1) {
        int leftCell = mesh_.faces[face].leftCell;
        int leftCell_m1 = leftCell - (mesh_.ni-1);
        double delta_p = pp[leftCell] - pp[leftCell_m1];
        double p_right = pp[leftCell] + delta_p;
        double p_right_p1 = p_right + delta_p;
        auto [eps2, eps4] = epsilon(pp[leftCell_m1], pp[leftCell], p_right, p_right_p1);
        double lambdaS = LambdaI[leftCell] + LambdaJ[leftCell];
        double delta_W0 = W0[leftCell] - W0[leftCell_m1];
        double delta_W1 = W1[leftCell] - W1[leftCell_m1];
        double delta_W2 = W2[leftCell] - W2[leftCell_m1];
        double delta_W3 = W3[leftCell] - W3[leftCell_m1];
        double W0_right = W0[leftCell] + delta_W0;
        double W1_right = W1[leftCell] + delta_W1;
        double W2_right = W2[leftCell] + delta_W2;
        double W3_right = W3[leftCell] + delta_W3;
        double W0_right_p1 = W0_right + delta_W0;
        double W1_right_p1 = W1_right + delta_W1;
        double W2_right_p1 = W2_right + delta_W2;
        double W3_right_p1 = W3_right + delta_W3;
        D0[face] = lambdaS*(eps2*(W0_right - W0[leftCell]) - eps4*(W0_right_p1 - 3.0*W0_right + 3.0*W0[leftCell]-W0[leftCell_m1]));
        D1[face] = lambdaS*(eps2*(W1_right - W1[leftCell]) - eps4*(W1_right_p1 - 3.0*W1_right + 3.0*W1[leftCell]-W1[leftCell_m1]));
        D2[face] = lambdaS*(eps2*(W2_right - W2[leftCell]) - eps4*(W2_right_p1 - 3.0*W2_right + 3.0*W2[leftCell]-W2[leftCell_m1]));
        D3[face] = lambdaS*(eps2*(W3_right - W3[leftCell]) - eps4*(W3_right_p1 - 3.0*W3_right + 3.0*W3[leftCell]-W3[leftCell_m1]));
    }

}

std::tuple<double, double> SpatialDiscretization::epsilon(double p_Im1, double p_I, double p_Ip1, double p_Ip2) {
    double gamma_I = fabs(p_Ip1 - 2.0*p_I + p_Im1) / (p_Ip1 + 2.0*p_I + p_Im1);
    double gamma_Ip1 = fabs(p_Ip2 - 2.0*p_Ip1 + p_I) / (p_Ip2 + 2.0*p_Ip1 + p_I);
    double eps2 = k2_ * std::max(gamma_I, gamma_Ip1);
    double eps4 = std::max(0.0, k4_ - eps2);
    return {eps2, eps4};
}

void SpatialDiscretization::compute_convective_residuals() {
    for (auto face : mesh_.fluidFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;

        // Update residuals for left cell
        Rc0[leftCell] += F0[face];
        Rc1[leftCell] += F1[face];
        Rc2[leftCell] += F2[face];
        Rc3[leftCell] += F3[face];

        // Update residuals for right cell
        Rc0[rightCell] -= F0[face];
        Rc1[rightCell] -= F1[face];
        Rc2[rightCell] -= F2[face];
        Rc3[rightCell] -= F3[face];
    }

    for (auto face : mesh_.immersedBoundaryFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int leftCellType = mesh_.cell_types[leftCell];
        int fluidCell = (leftCellType == 1) ? leftCell : rightCell;

        // Update residuals for fluid cell only
        if (fluidCell == leftCell) {
            Rc0[fluidCell] += F0[face];
            Rc1[fluidCell] += F1[face];
            Rc2[fluidCell] += F2[face];
            Rc3[fluidCell] += F3[face];
        } else {
            Rc0[fluidCell] -= F0[face];
            Rc1[fluidCell] -= F1[face];
            Rc2[fluidCell] -= F2[face];
            Rc3[fluidCell] -= F3[face];
        }
    }

    for (auto face : mesh_.farfieldFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int leftCellType = mesh_.cell_types[leftCell];
        int fluidCell = (leftCellType == 1) ? leftCell : rightCell;

        // Update residuals for fluid cell only
        if (fluidCell == leftCell) {
            Rc0[fluidCell] += F0[face];
            Rc1[fluidCell] += F1[face];
            Rc2[fluidCell] += F2[face];
            Rc3[fluidCell] += F3[face];
        } else {
            Rc0[fluidCell] -= F0[face];
            Rc1[fluidCell] -= F1[face];
            Rc2[fluidCell] -= F2[face];
            Rc3[fluidCell] -= F3[face];
        }
    }
}

void SpatialDiscretization::compute_diffusive_residuals() {
    for (auto face : mesh_.fluidFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;

        // Update residuals for left cell
        Rd0[leftCell] += D0[face];
        Rd1[leftCell] += D1[face];
        Rd2[leftCell] += D2[face];
        Rd3[leftCell] += D3[face];

        // Update residuals for right cell
        Rd0[rightCell] -= D0[face];
        Rd1[rightCell] -= D1[face];
        Rd2[rightCell] -= D2[face];
        Rd3[rightCell] -= D3[face];
    }

    // for (auto face : mesh_.immersedBoundaryFaces) {
    //     int leftCell = mesh_.faces[face].leftCell;
    //     int rightCell = mesh_.faces[face].rightCell;
    //     int leftCellType = mesh_.cell_types[leftCell];
    //     int fluidCell = (leftCellType == 1) ? leftCell : rightCell;

    //     // Update residuals for fluid cell only
    //     if (fluidCell == leftCell) {
    //         Rd0[fluidCell] += D0[face];
    //         Rd1[fluidCell] += D1[face];
    //         Rd2[fluidCell] += D2[face];
    //         Rd3[fluidCell] += D3[face];
    //     } else {
    //         Rd0[fluidCell] -= D0[face];
    //         Rd1[fluidCell] -= D1[face];
    //         Rd2[fluidCell] -= D2[face];
    //         Rd3[fluidCell] -= D3[face];
    //     }
    // }

    // for (auto face : mesh_.farfieldFaces) {
    //     int leftCell = mesh_.faces[face].leftCell;
    //     int rightCell = mesh_.faces[face].rightCell;
    //     int leftCellType = mesh_.cell_types[leftCell];
    //     int fluidCell = (leftCellType == 1) ? leftCell : rightCell;

    //     // Update residuals for fluid cell only
    //     if (fluidCell == leftCell) {
    //         Rd0[fluidCell] += D0[face];
    //         Rd1[fluidCell] += D1[face];
    //         Rd2[fluidCell] += D2[face];
    //         Rd3[fluidCell] += D3[face];
    //     } else {
    //         Rd0[fluidCell] -= D0[face];
    //         Rd1[fluidCell] -= D1[face];
    //         Rd2[fluidCell] -= D2[face];
    //         Rd3[fluidCell] -= D3[face];
    //     }
    // }
}


void SpatialDiscretization::updatePrimitivesVariables() {
    for (auto cell : mesh_.fluidCells) {
        rhorho[cell] = W0[cell];
        uu[cell] = W1[cell] / W0[cell];
        vv[cell] = W2[cell] / W0[cell];
        EE[cell] = W3[cell] / W0[cell];
        pp[cell] = pressure(gamma_, rhorho[cell], uu[cell], vv[cell], EE[cell]);
    }
}

void SpatialDiscretization::run_even() {
    updatePrimitivesVariables();
    updateGhostCells();
    compute_convective_fluxes();
    compute_convective_residuals();
}

void SpatialDiscretization::run_odd() {
    updatePrimitivesVariables();
    updateGhostCells();
    compute_lambdas();
    compute_convective_fluxes();
    compute_diffusive_fluxes();
    compute_convective_residuals();
    compute_diffusive_residuals();
}

std::tuple<double, double, double> SpatialDiscretization::compute_aerodynamics_coefficients() {
    double Fx = 0.0;
    double Fy = 0.0;
    double M = 0.0;
    for (auto face : mesh_.immersedBoundaryFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int leftCellType = mesh_.cell_types[leftCell];
        int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
        double p_face = 0.5 * (pp[leftCell] + pp[rightCell]);
        double nx = mesh_.faces[face].nx;
        double ny = mesh_.faces[face].ny;
        if (fluidCell != leftCell) {
            nx = -nx;
            ny = -ny;
        }
        double area = mesh_.faces[face].area;
        double cx = mesh_.faces[face].cx;
        double cy = mesh_.faces[face].cy;
        Fx += p_face * nx * area;
        Fy += p_face * ny * area;
        M += p_face * ((0.25-cx) * ny + p_face * (cy-0.0) * nx) * area;
    }
    double L = Fy*std::cos(alpha_) - Fx*std::sin(alpha_);
    double D = Fx*std::cos(alpha_) + Fy*std::sin(alpha_);
    double q_inf = 0.5 * rhoInfty_ * Mach_*cInfty_ * Mach_*cInfty_;
    double S = 1.0; // Reference area
    double C_L = L / (q_inf * S);
    double C_D = D / (q_inf * S);
    double c = 1.0; // Reference length
    double C_M = M / (q_inf * S * c);
    return {C_L, C_D, C_M};
    

}