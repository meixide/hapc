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
#include <RcppEigen.h>
#include <cmath>
#include <vector>
#include <algorithm>
using Eigen::Map;
using Eigen::MatrixXd;

// -----------------------------------------------------------------------------
// Cross Haar-like kernel with interactions up to order m_order
// K_test (m_test × n) between X_test (m_test × p) and X_train (n × p):
// K[t,b] = sum_i sum_{k=0}^{min(g,m_order)} C(g,k),
// where g_{i,tb} = #{ j : X_train[i,j] <= min(X_test[t,j], X_train[b,j]) }.
// -----------------------------------------------------------------------------
extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_) {
    const int n = Rf_nrows(Xtr_);
    const int p = Rf_ncols(Xtr_);
    const int m_test = Rf_nrows(Xte_);
    
    if (n <= 0 || p <= 0 || m_test <= 0)
        Rf_error("Invalid matrix dimensions");
    
    const double* Xtrp = REAL(Xtr_);
    const double* Xtep = REAL(Xte_);
    
    // Map R memory to Eigen matrices
    Map<const MatrixXd> Xtr(Xtrp, n, p);
    Map<const MatrixXd> Xte(Xtep, m_test, p);
    
    // Extract m_order (max interaction order) robustly
    int m_order;
    if (TYPEOF(m_) == INTSXP) {
        m_order = INTEGER(m_)[0];
    } else if (TYPEOF(m_) == REALSXP) {
        m_order = static_cast<int>(REAL(m_)[0]);
    } else {
        Rf_error("Argument 'm' must be numeric or integer");
    }
    if (m_order < 0) m_order = 0;
    if (m_order > p) m_order = p;
    
    // Precompute partial sums using stable incremental computation
    // psum[g] = sum_{k=0}^{min(g,m_order)} C(g,k)
    std::vector<double> psum(p + 1, 0.0);
    psum[0] = 1.0;  // C(0,0) = 1
    for (int g = 1; g <= p; ++g) {
        const int kmax = std::min(g, m_order);
        // Compute sum incrementally: sum += C(g,k) for k=0..kmax
        // Use recurrence: C(g,k) = C(g,k-1) * (g-k+1) / k
        double sum = 1.0;  // k=0: C(g,0) = 1
        double binom = 1.0;
        for (int k = 1; k <= kmax; ++k) {
            binom = binom * (g - k + 1) / k;  // C(g,k) = C(g,k-1) * (g-k+1) / k
            sum += binom;
        }
        psum[g] = sum;
    }
    
    // Allocate output matrix (m_test × n)
    SEXP K_ = PROTECT(Rf_allocMatrix(REALSXP, m_test, n));
    double* Kp = REAL(K_);
    Map<MatrixXd> K(Kp, m_test, n);
    K.setZero();
    
    // Main computation
    for (int t = 0; t < m_test; ++t) {       // test rows
        for (int b = 0; b < n; ++b) {         // train rows
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                int g = 0;
                for (int j = 0; j < p; ++j) {
                    double thr = std::min(Xte(t, j), Xtr(b, j));
                    if (Xtr(i, j) <= thr)
                        ++g;
                }
                // Contribution: sum_{k=0}^{min(g,m_order)} C(g,k)
                const int gg = (g <= p) ? g : p;  // safety
                s += psum[gg] - 1.0;
            }
            K(t, b) = s;
        }
    }
    
    // Centering if requested
    if (Rf_asLogical(center_) == TRUE) {
        // First compute the training kernel (n × n) for centering statistics
        MatrixXd Ktrain(n, n);
        Ktrain.setZero();
        
        for (int a = 0; a < n; ++a) {
            for (int b = a; b < n; ++b) {
                double s = 0.0;
                for (int i = 0; i < n; ++i) {
                    int g = 0;
                    for (int j = 0; j < p; ++j) {
                        double thr = std::min(Xtr(a, j), Xtr(b, j));
                        if (Xtr(i, j) <= thr)
                            ++g;
                    }
                    // Contribution: sum_{k=0}^{min(g,m_order)} C(g,k)
                    const int gg = (g <= p) ? g : p;  // safety
                    s += psum[gg] - 1.0;
                }
                Ktrain(a, b) = s;
                if (a != b) Ktrain(b, a) = s;
            }
        }
        
        // Compute column means of training kernel
        std::vector<double> colmean_train(n, 0.0);
        double grand_train = 0.0;
        
        for (int b = 0; b < n; ++b) {
            double cs = 0.0;
            for (int a = 0; a < n; ++a) {
                cs += Ktrain(a, b);
            }
            colmean_train[b] = cs / n;
            grand_train += cs;
        }
        grand_train /= (n * n);
        
        // Compute row means of test kernel
        std::vector<double> rowmean_test(m_test, 0.0);
        for (int t = 0; t < m_test; ++t) {
            double rs = 0.0;
            for (int b = 0; b < n; ++b) {
                rs += K(t, b);
            }
            rowmean_test[t] = rs / n;
        }
        
        // Apply centering: K[t,b] <- K[t,b] - rowmean_test[t] - colmean_train[b] + grand_train
        for (int t = 0; t < m_test; ++t) {
            for (int b = 0; b < n; ++b) {
                K(t, b) = K(t, b) - rowmean_test[t] - colmean_train[b] + grand_train;
            }
        }
    }
    
    UNPROTECT(1);
    return K_;
}