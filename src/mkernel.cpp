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
#include <vector>
#include <algorithm>
#include <cmath>

// Modified Haar-like kernel limited to subsets up to size m:
// K(x, x') = sum_i sum_{s subseteq s_i(x,x'), |s|<=m} 1
//          = sum_i sum_{k=0}^{min(g,m)} C(g,k)
// where g = |s_i(x,x')| = #{ j : X[i,j] <= min(x_j, x'_j) }
extern "C" SEXP mkernel_call(SEXP X_, SEXP m_) {
    if (TYPEOF(X_) != REALSXP || !Rf_isMatrix(X_))
        Rf_error("X must be a numeric (double) matrix");
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    const double* Xp = REAL(X_);
    
    // Extract m robustly
    int m;
    if (TYPEOF(m_) == INTSXP) {
        m = INTEGER(m_)[0];
    } else if (TYPEOF(m_) == REALSXP) {
        m = static_cast<int>(REAL(m_)[0]);
    } else {
        Rf_error("Argument 'm' must be numeric or integer");
    }
    if (m < 0) m = 0;
    if (m > p) m = p;
    
    // Precompute partial sums using stable incremental computation
    // psum[g] = sum_{k=0}^{min(g,m)} C(g,k)
    std::vector<double> psum(p + 1, 0.0);
    
    // For small values, use direct computation
    psum[0] = 1.0;  // C(0,0) = 1
    
    for (int g = 1; g <= p; ++g) {
        const int kmax = std::min(g, m);
        
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
    
    // Allocate output matrix K (n x n)
    SEXP K_ = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    double* K = REAL(K_);
    std::fill(K, K + n * n, 0.0);
    
    // Main loops (upper triangle + mirror)
    for (int a = 0; a < n; ++a) {
        for (int b = a; b < n; ++b) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                int g = 0;
                for (int j = 0; j < p; ++j) {
                    const double xa = Xp[a + j * n];
                    const double xb = Xp[b + j * n];
                    const double xi = Xp[i + j * n];
                    // Handle NaNs: comparisons with NaN are false, so they don't increment g
                    const double thr = (xa < xb) ? xa : xb;
                    if (xi <= thr) ++g;
                }
                // Contribution: sum_{k=0}^{min(g,m)} C(g,k)
                const int gg = (g <= p) ? g : p;  // safety
                s += psum[gg] - 1.0;
            }
            K[a + b * n] = s;
            if (a != b) K[b + a * n] = s;
        }
    }
    UNPROTECT(1);
    return K_;
}