// orthogonal_lasso_call.cpp

// ---- Fix R macro conflicts (same as before) ----

#include <cmath>
#include <algorithm>


// Pure C++ core (shared by R and Python)
#include "hapc_core.hpp"
#include <cmath>

VectorXd fast_pchal_call(const MatrixXd& U, const VectorXd& D2, const VectorXd& Y, double lambda) {
    const int n = U.rows();
    const int p = U.cols();

    if (n <= 0 || p <= 0) throw std::runtime_error("Invalid matrix dimensions");
    if (Y.size() != n) throw std::runtime_error("Dimension mismatch");
    if (D2.size() != p) throw std::runtime_error("Dimension mismatch");
    if (lambda < 0) throw std::runtime_error("lambda must be non-negative");

    VectorXd UTy = U.transpose() * Y;
    VectorXd sqrtD = D2.array().sqrt();

    VectorXd beta(p);
    for (int j = 0; j < p; ++j) {
        double threshold = lambda * n / sqrtD[j];
        double val = std::abs(UTy[j]) - threshold;
        beta[j] = (val > 0 ? std::copysign(val / sqrtD[j], UTy[j]) : 0.0);
    }

    return beta;
}