// har_kernel_call.cpp
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

// K[a,b] = sum_i ( 2^{ g_{i,ab} } - 1 ),
// where g_{i,ab} = #{ j : X[i,j] <= min(X[a,j], X[b,j]) }.
extern "C" SEXP kernel_call(SEXP X_) {
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    const double* Xp = REAL(X_);

    // precompute 2^k, k=0..p
    std::vector<double> pow2(p + 1);
    pow2[0] = 1.0;
    for (int k = 1; k <= p; ++k) pow2[k] = 2.0 * pow2[k - 1];

    // allocate K (n x n)
    SEXP K_ = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    double* K = REAL(K_);
    std::fill(K, K + n * n, 0.0);

    // main loops (upper triangle, then mirror)
    for (int a = 0; a < n; ++a) {
        for (int b = a; b < n; ++b) {

            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                int g = 0;
                for (int j = 0; j < p; ++j) {
                    const double xa = Xp[a + j * n];
                    const double xb = Xp[b + j * n];
                    const double xi = Xp[i + j * n];
                    const double thr = (xa < xb) ? xa : xb;
                    if (xi <= thr) ++g;
                }
                s += pow2[g] - 1.0;
            }

            K[a + b * n] = s;
            if (a != b) K[b + a * n] = s;
        }
    }

    UNPROTECT(1);
    return K_;
}