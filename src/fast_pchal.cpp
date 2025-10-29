// orthogonal_lasso_call.cpp
#define R_NO_REMAP
#include <Rinternals.h>

// ---- Fix R macro conflicts (same as before) ----
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
#include <RcppEigen.h>
#include <cmath>
#include <algorithm>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

extern "C" SEXP fast_pchal_call(SEXP U_, SEXP D2_, SEXP Y_, SEXP lambda_) {
    const int n = Rf_nrows(U_);
    const int p = Rf_ncols(U_);

    if (n <= 0 || p <= 0)
        Rf_error("Invalid matrix dimensions (n=%d, p=%d)", n, p);

    const double* Up = REAL(U_);
    const double* Dp = REAL(D2_);
    const double* Yp = REAL(Y_);
    const double lam = REAL(lambda_)[0];

    if (lam < 0)
        Rf_error("lambda must be non-negative");

    // Map R objects to Eigen
    Map<const MatrixXd> U(Up, n, p);
    Map<const VectorXd> D(Dp, p);
    Map<const VectorXd> y(Yp, n);

    VectorXd UTy = U.transpose() * y;

    VectorXd sqrtD = D.array().sqrt();


    VectorXd beta(p);
    for (int j = 0; j < p; ++j) {
        double threshold = lam * n / sqrtD[j];
        double val = std::abs(UTy[j]) - threshold;
        beta[j] = (val > 0 ? std::copysign(val / sqrtD[j], UTy[j]) : 0.0);
    }

    // Return as numeric vector to R
    SEXP out = PROTECT(Rf_allocVector(REALSXP, p));
    for (int j = 0; j < p; ++j)
        REAL(out)[j] = beta[j];

    UNPROTECT(1);
    return out;
}