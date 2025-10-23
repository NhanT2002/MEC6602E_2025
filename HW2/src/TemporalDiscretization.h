#ifndef TEMPORAL_DISCRETIZATION_H
#define TEMPORAL_DISCRETIZATION_H
#include <vector>
#include <cmath>
#include "SpatialDiscretization.h"

class TemporalDiscretization {
public:
    SpatialDiscretization &spatialDiscretization_;
    double CFL_;
    int it_max_;
    std::vector<double> dt_cells;

    double a1 = 0.25; double b1 = 1.0;
    double a2 = 0.1667; double b2 = 0.0;
    double a3 = 0.3750; double b3 = 0.56;
    double a4 = 0.5; double b4 = 0.0;
    double a5 = 1.0; double b5 = 0.44;

    TemporalDiscretization(SpatialDiscretization &spatialDiscretization, double CFL, int it_max);

    void compute_dt();

    void eulerStep();
    void RungeKuttaStep();

    void solve();
};

#endif // TEMPORAL_DISCRETIZATION_H