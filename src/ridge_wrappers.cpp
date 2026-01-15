// Pure C++ ridge implementation (shared by R and Python)
#include "hapc_core.hpp"

VectorXd ridge_call(const VectorXd& Y, const MatrixXd& U, const VectorXd& D2, double lambda) {
    const int n = U.rows();
    const int p = U.cols();
    
    // Safety checks
    if (Y.size() != n) throw std::runtime_error("Dimension mismatch: Y and U");
    if (D2.size() != p) throw std::runtime_error("Dimension mismatch: D2 and U");
    
    // 1. Compute U^T * Y
    VectorXd UTy = U.transpose() * Y;
    
    VectorXd beta(p);
    
    for (int j = 0; j < p; ++j) {
        // 2. Denominator: D^2 + n * lambda
        double denom = D2[j] + lambda * n;
        
        // 3. Numerator Scaling: sqrt(D^2) corresponds to diag(D) in R
        // Use max(0, val) to prevent NaNs if small eigenvalues are slightly negative due to precision
        double d_val = std::sqrt(std::max(0.0, D2[j])); 
        
        // 4. Combine: (sqrt(D2) / (D2 + n*lambda)) * (U'Y)
        if (denom > 1e-12) {
            beta[j] = (d_val * UTy[j]) / denom;
        } else {
            beta[j] = 0.0;
        }
    }
    
    return beta;
}