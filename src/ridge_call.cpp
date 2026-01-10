// ridge_call.cpp
#define R_NO_REMAP
#include <Rinternals.h>

// ---- Fix R macro conflicts ----
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

// Updated signature to accept U and D2 instead of X
extern "C" SEXP ridge_call(SEXP Y_, SEXP U_, SEXP D2_, SEXP lambda_) {
    // 1. Check dimensions and input types
    if (!Rf_isMatrix(U_) || !Rf_isReal(U_))
        Rf_error("U must be a numeric matrix");
    
    const int n = Rf_nrows(U_);
    const int k = Rf_ncols(U_);
    
    if (Rf_length(Y_) != n)
        Rf_error("Y must have same length as nrow(U)");
    if (Rf_length(D2_) != k)
        Rf_error("D2 must have same length as ncol(U)");

    // 2. Map Inputs
    Map<const VectorXd> y(REAL(Y_), n);
    Map<const MatrixXd> U(REAL(U_), n, k);
    Map<const VectorXd> D2(REAL(D2_), k);
    const double lam = REAL(lambda_)[0];
    const double n_dbl = (double)n;

    // 3. Allocate Output Vector (length k)
    // We allocate this first so we can compute directly into it
    SEXP out = PROTECT(Rf_allocVector(REALSXP, k));
    Map<VectorXd> res(REAL(out), k);

    // 4. Compute w = U^T * Y
    // We compute this directly into 'res' to avoid a temporary vector allocation.
    // .noalias() tells Eigen the result doesn't overlap with inputs (optimization hint).
    res.noalias() = U.transpose() * y;

    // 5. Compute scaling vector z and apply element-wise
    // z = sqrt(D2) / (D2 + n*lambda)
    // res = res * z
    // We do this in-place on 'res'
    res.array() *= D2.array().sqrt() / (D2.array() + n_dbl * lam);
    Rprintf("New O(n) version");


    UNPROTECT(1);
    return out;
}

