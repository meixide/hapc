// ridge_call.cpp
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

extern "C" SEXP ridge_call(SEXP Y_, SEXP X_, SEXP lambda_) {
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);

    if (n <= 0 || p <= 0)
        Rf_error("Invalid matrix dimensions (n=%d, p=%d)", n, p);

    const double* Yp = REAL(Y_);
    const double* Xp = REAL(X_);
    const double lam = REAL(lambda_)[0];

    if (lam < 0)
        Rf_error("lambda must be non-negative");

    // Map R memory to Eigen (column-major)
    Map<const MatrixXd> X(Xp, n, p);
    Map<const VectorXd> y(Yp, n);

    // Compute ridge estimator: (X'X + nλI)^(-1) X'y
    MatrixXd XtX = X.transpose() * X;
    XtX.diagonal().array() += lam * n;  // add nλ to diagonal
    VectorXd Xty = X.transpose() * y;

    // Solve using LDLT for numerical stability
    VectorXd beta = XtX.ldlt().solve(Xty);

    // Return as numeric vector to R
    SEXP out = PROTECT(Rf_allocVector(REALSXP, p));
    for (int j = 0; j < p; ++j)
        REAL(out)[j] = beta[j];

    UNPROTECT(1);
    return out;
}

