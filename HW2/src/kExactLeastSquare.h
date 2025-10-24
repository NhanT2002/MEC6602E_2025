#ifndef KEXACTLEASTSQUARE_H
#define KEXACTLEASTSQUARE_H
#include <vector>
#include <Eigen/Dense>

class kExactLeastSquare {
public:
    std::vector<double> stencilCellsCx_;
    std::vector<double> stencilCellsCy_;
    double centerCx_;
    double centerCy_;
    Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero(3, 3); // 3x3 matrix for 2D least squares
    double value_;
    double gradX_;
    double gradY_;
    kExactLeastSquare(std::vector<double> stencilCellsCx,
                      std::vector<double> stencilCellsCy,
                      double centerCx,
                      double centerCy);

    std::vector<double> x_;
    std::vector<double> y_;
    std::vector<double> c0_;
    std::vector<double> cx_;
    std::vector<double> cy_;

    double interpolate(std::vector<double> fieldValues);

};

#endif // KEXACTLEASTSQUARE_H