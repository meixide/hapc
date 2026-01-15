// logistic_call.cpp

// ---- Fix R macro conflicts ----

#include <RcppEigen.h>
#include <cmath>
#include <algorithm>


extern "C" SEXP logistic_call(SEXP Y_, SEXP X_, SEXP lambda_) {
    // 1. Check dimensions and input types
    if (!Rf_isMatrix(X_) || !Rf_isReal(X_))
        Rf_error("X must be a numeric matrix");
    
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    
    if (Rf_length(Y_) != n)
        Rf_error("Y must have same length as nrow(X)");

    // 2. Map Inputs
    Map<const VectorXd> y(REAL(Y_), n);
    Map<const MatrixXd> X(REAL(X_), n, p);
    double lam = REAL(lambda_)[0];
    
    // Transform lambda to lambda * n
    lam = lam * n;

    // 3. Initialize coefficients to zero
    VectorXd beta = VectorXd::Zero(p);
    
    // 4. Newton-Raphson optimization parameters
    const int max_iter = 100;
    const double tol = 1e-8;
    
    // 5. Newton-Raphson iterations
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute linear predictor: eta = X * beta
        VectorXd eta = X * beta;
        
        // Compute probabilities: mu = 1 / (1 + exp(-eta))
        VectorXd mu = (1.0 + (-eta.array()).exp()).inverse();
        
        // Compute weights: w = mu * (1 - mu)
        VectorXd w = mu.array() * (1.0 - mu.array());
        
        // Avoid numerical issues with very small weights
        w = w.array().max(1e-8);
        
        // Compute working response: z = eta + (y - mu) / w
        VectorXd z = eta.array() + (y.array() - mu.array()) / w.array();
        
        // Compute weighted X: sqrt(w) * X
        MatrixXd XtW = X.transpose() * w.asDiagonal();
        
        // Compute Hessian: X^T W X + lambda * I
        MatrixXd H = XtW * X;
        H.diagonal().array() += lam;
        
        // Compute gradient: X^T W z - lambda * beta
        VectorXd grad = XtW * z - lam * beta;
        
        // Solve for update: delta_beta = H^{-1} * grad
        VectorXd delta_beta = H.ldlt().solve(grad);
        
        // Update coefficients
        VectorXd beta_old = beta;
        beta = delta_beta;
        
        // Check convergence
        double change = (beta - beta_old).norm();
        if (change < tol) {
            Rprintf("Converged in %d iterations\n", iter + 1);
            break;
        }
        
        if (iter == max_iter - 1) {
            Rprintf("Warning: Maximum iterations reached without convergence\n");
        }
    }
    
    // 6. Allocate and populate output
    SEXP out = PROTECT(Rf_allocVector(REALSXP, p));
    Map<VectorXd> res(REAL(out), p);
    res = beta;
    
    UNPROTECT(1);
    return out;
}