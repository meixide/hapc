// har_kernel_cross_call.cpp
#define R_NO_REMAP
#include <Rinternals.h>

#ifdef length
#undef length
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <R_ext/Print.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <algorithm>

using Eigen::Map;
using Eigen::MatrixXd;

// -----------------------------------------------------------------------------
// Cross Haar-like kernel
// K_test (m × n) between X_test (m × p) and X_train (n × p):
// K[t,b] = sum_i ( 2^{g_{i,tb}} - 1 ),
// where g_{i,tb} = #{ j : X_train[i,j] <= min(X_test[t,j], X_train[b,j]) }.
// -----------------------------------------------------------------------------
extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_) {
    const int n = Rf_nrows(Xtr_);
    const int p = Rf_ncols(Xtr_);
    const int m = Rf_nrows(Xte_);

    if (n <= 0 || p <= 0 || m <= 0)
        Rf_error("Invalid matrix dimensions");

    const double* Xtrp = REAL(Xtr_);
    const double* Xtep = REAL(Xte_);

    // Map R memory to Eigen matrices
    Map<const MatrixXd> Xtr(Xtrp, n, p);
    Map<const MatrixXd> Xte(Xtep, m, p);

    // Precompute 2^k for k = 0,...,p
    std::vector<double> pow2(p + 1);
    pow2[0] = 1.0;
    for (int k = 1; k <= p; ++k)
        pow2[k] = 2.0 * pow2[k - 1];

    // Allocate output matrix (m × n)
    SEXP K_ = PROTECT(Rf_allocMatrix(REALSXP, m, n));
    double* Kp = REAL(K_);
    Map<MatrixXd> K(Kp, m, n);
    K.setZero();

    // Main computation
    for (int t = 0; t < m; ++t) {       // test rows
        for (int b = 0; b < n; ++b) {   // train rows

            double s = 0.0;

            for (int i = 0; i < n; ++i) {
                int g = 0;
                for (int j = 0; j < p; ++j) {
                    double thr = std::min(Xte(t, j), Xtr(b, j));
                    if (Xtr(i, j) <= thr)
                        ++g;
                }
                s += pow2[g] - 1.0;
            }

            K(t, b) = s;
        }
    }

    UNPROTECT(1);
    return K_;
}