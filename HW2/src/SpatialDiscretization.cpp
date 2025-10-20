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
    for (auto cell : mesh_.fluidCells) {
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

void SpatialDiscretization::compute_fluxes() {
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

        double avg_u = avg_W1 / avg_W0;;
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

    for (auto face : mesh_.immersedBoundaryFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell; // should be -1 for IB face
        int leftCellType = mesh_.cell_types[leftCell];
        double ib_nx = mesh_.faces[face].ib_nx; // use immersed boundary normal
        double ib_ny = mesh_.faces[face].ib_ny;
        double nx = (fabs(ib_nx) > fabs(ib_ny)) ? ib_nx : 0.0;
        double ny = (fabs(ib_ny) > fabs(ib_nx)) ? ib_ny : 0.0;

        int fluidCell = (leftCellType == 1) ? leftCell : rightCell;
        int fluidCell_p1;
        int fluidCell_p2;
        if (nx < -1e-12) {
            fluidCell_p1 = fluidCell + 1;
            fluidCell_p2 = fluidCell + 2;
        } else if (nx > 1e-12) {
            fluidCell_p1 = fluidCell - 1;
            fluidCell_p2 = fluidCell - 2;
        } else if (ny < -1e-12) {
            fluidCell_p1 = fluidCell + (mesh_.ni-1);
            fluidCell_p2 = fluidCell + 2*(mesh_.ni-1);
        } else { // ny > 0
            fluidCell_p1 = fluidCell - (mesh_.ni-1);
            fluidCell_p2 = fluidCell - 2*(mesh_.ni-1);
        }
        // std::cout << "IB Face " << face  << " ib_nx=" << ib_nx << ", ib_ny=" << ib_ny << " nx =" << nx << ", ny=" << ny
        //           << ": fluidCell=" << fluidCell << ", p1=" << fluidCell_p1 << ", p2=" << fluidCell_p2 << std::endl;
        double area = mesh_.faces[face].area;
        double uwall = 0.125*(15*uu[fluidCell] - 10*uu[fluidCell_p1] + 3*uu[fluidCell_p2]); // 2nd order extrapolation
        double vwall = 0.125*(15*vv[fluidCell] - 10*vv[fluidCell_p1] + 3*vv[fluidCell_p2]); // 2nd order extrapolation
        double rhowall = 0.125*(15*rhorho[fluidCell] - 10*rhorho[fluidCell_p1] + 3*rhorho[fluidCell_p2]); // 2nd order extrapolation
        double Ewall = 0.125*(15*EE[fluidCell] - 10*EE[fluidCell_p1] + 3*EE[fluidCell_p2]); // 2nd order extrapolation
        double pwall = 0.125*(15*pp[fluidCell] - 10*pp[fluidCell_p1] + 3*pp[fluidCell_p2]); // 2nd order extrapolation

        double V = uwall * ib_nx + vwall * ib_ny;
        double H = Ewall + pwall / rhowall;
        F0[face] = rhowall * V * area;
        F1[face] = (rhowall*uwall*V + pwall*ib_nx) * area;
        F2[face] = (rhowall*vwall*V + pwall*ib_ny) * area;
        F3[face] = (rhowall*H*V) * area;
    }

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

void SpatialDiscretization::compute_residuals() {
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
        Rc0[fluidCell] += F0[face];
        Rc1[fluidCell] += F1[face];
        Rc2[fluidCell] += F2[face];
        Rc3[fluidCell] += F3[face];
    }

    for (auto face : mesh_.farfieldFaces) {
        int leftCell = mesh_.faces[face].leftCell;
        int rightCell = mesh_.faces[face].rightCell;
        int leftCellType = mesh_.cell_types[leftCell];
        int fluidCell = (leftCellType == 1) ? leftCell : rightCell;

        // Update residuals for fluid cell only
        Rc0[fluidCell] += F0[face];
        Rc1[fluidCell] += F1[face];
        Rc2[fluidCell] += F2[face];
        Rc3[fluidCell] += F3[face];
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

void SpatialDiscretization::updatePrimitivesVariables() {
    for (auto cell : mesh_.fluidCells) {
        rhorho[cell] = W0[cell];
        uu[cell] = W1[cell] / W0[cell];
        vv[cell] = W2[cell] / W0[cell];
        EE[cell] = W3[cell] / W0[cell];
        pp[cell] = pressure(gamma_, rhorho[cell], uu[cell], vv[cell], EE[cell]);
    }
}