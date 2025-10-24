#include "kExactLeastSquare.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

kExactLeastSquare::kExactLeastSquare()
    : stencilCellsCx_(), stencilCellsCy_(),
      centerCx_(0.0), centerCy_(0.0),
      A_(Eigen::MatrixXd::Zero(3,3)),
      value_(0.0), gradX_(0.0), gradY_(0.0),
      x_(), y_(), c0_(), cx_(), cy_()
{
    // intentionally empty: default object is inert until properly initialized
}

kExactLeastSquare::kExactLeastSquare(std::vector<double> stencilCellsCx,
                                     std::vector<double> stencilCellsCy,
                                     double centerCx,
                                     double centerCy)
    : stencilCellsCx_(stencilCellsCx),
      stencilCellsCy_(stencilCellsCy),
      centerCx_(centerCx),
      centerCy_(centerCy) {

    x_.resize(stencilCellsCx_.size());
    y_.resize(stencilCellsCy_.size());
    c0_.resize(stencilCellsCx_.size());
    cx_.resize(stencilCellsCx_.size());
    cy_.resize(stencilCellsCy_.size());

    // Initialize the field values
    rho_fieldValues_.resize(stencilCellsCx_.size());
    u_fieldValues_.resize(stencilCellsCx_.size());
    v_fieldValues_.resize(stencilCellsCx_.size());
    E_fieldValues_.resize(stencilCellsCx_.size());
    p_fieldValues_.resize(stencilCellsCx_.size());

    // Initialize the least squares problem
    for (size_t i = 0; i < stencilCellsCx_.size(); ++i) {
        double dx = stencilCellsCx_[i] - centerCx_;
        double dy = stencilCellsCy_[i] - centerCy_;
        A_(0, 0) += 1.0;
        A_(0, 1) += dx;
        A_(0, 2) += dy;
        A_(1, 0) += dx;
        A_(1, 1) += dx * dx;
        A_(1, 2) += dx * dy;
        A_(2, 0) += dy;
        A_(2, 1) += dx * dy;
        A_(2, 2) += dy * dy;

        x_[i] = dx;
        y_[i] = dy;
    }

    // Solve the least squares problem
    Eigen::MatrixXd A_inv = A_.inverse();

    for (size_t i = 0; i < stencilCellsCx_.size(); ++i) {
        c0_[i] = A_inv(0, 0) + A_inv(0, 1) * x_[i] + A_inv(0, 2) * y_[i];
        cx_[i] = A_inv(1, 0) + A_inv(1, 1) * x_[i] + A_inv(1, 2) * y_[i];
        cy_[i] = A_inv(2, 0) + A_inv(2, 1) * x_[i] + A_inv(2, 2) * y_[i];
    }
}

double kExactLeastSquare::interpolate(std::vector<double> fieldValues) {
    double c0 = 0.0;
    double cx = 0.0;
    double cy = 0.0;

    for (size_t i = 0; i < stencilCellsCx_.size(); ++i) {
        double value = fieldValues[i];
        c0 += c0_[i] * value;
        cx += cx_[i] * value;
        cy += cy_[i] * value;
    }

    gradX_ = cx;
    gradY_ = cy;
    value_ = c0;

    return value_;
}