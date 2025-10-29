#define R_NO_REMAP
#include <Rinternals.h>

// --- Fix R macro pollution ---
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
#include <Eigen/Dense>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ---- recursively generate all variable index combinations ----
static void comb_recursive(std::vector<int>& cur, int start, int depth, int k, int p,
                           std::vector<std::vector<int>>& all) {
    if (depth == k) {
        all.push_back(cur);
        return;
    }
    for (int i = start; i < p; ++i) {
        cur.push_back(i);
        comb_recursive(cur, i + 1, depth + 1, k, p, all);
        cur.pop_back();
    }
}

extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_) {
    if (!Rf_isReal(X_))
        Rf_error("X must be a numeric matrix");

    // --- parse max_degree ---
    int maxdeg;
    if (Rf_isInteger(maxdeg_)) maxdeg = INTEGER(maxdeg_)[0];
    else if (Rf_isReal(maxdeg_)) maxdeg = (int)REAL(maxdeg_)[0];
    else Rf_error("max_degree must be integer/numeric");
    if (maxdeg < 1) Rf_error("max_degree must be >= 1");

    // --- parse npc ---
    int npc;
    if (Rf_isInteger(npc_)) npc = INTEGER(npc_)[0];
    else if (Rf_isReal(npc_)) npc = (int)REAL(npc_)[0];
    else Rf_error("npc must be integer/numeric");

    // --- parse center ---
    bool center = true;
    if (Rf_isLogical(center_)) center = LOGICAL(center_)[0];
    else Rf_error("center must be logical");

    

    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    const double* Xp = REAL(X_);

    if (n <= 0 || p <= 0) Rf_error("X must have positive dimensions");

    // --- copy X into column-major std::vectors for easy access ---
    std::vector<std::vector<double>> Xcols(p, std::vector<double>(n));
    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i)
            Xcols[j][i] = Xp[i + n*j];

    // --- compute total number of columns q = n * sum_{k=1..maxdeg} C(p,k) ---
    auto choose = [](int nn, int kk) -> long long {
        if (kk < 0 || kk > nn) return 0;
        long long r = 1;
        for (int i = 1; i <= kk; ++i) r = (r * (nn - i + 1)) / i;
        return r;
    };
    long long qll = 0;
    for (int k = 1; k <= maxdeg; ++k) qll += choose(p, k) * (long long)n;
    if (qll <= 0) Rf_error("invalid column count");
    if (qll > (long long)std::numeric_limits<int>::max())
        Rf_error("design matrix too wide for this build");
    const int q = (int)qll;

    // --- allocate H (n x q) in R memory and zero-init ---
    SEXP H_ = PROTECT(Rf_allocMatrix(REALSXP, n, q));
    double* Hptr = REAL(H_);
    std::fill(Hptr, Hptr + (size_t)n * (size_t)q, 0.0);

    // --- build H in the same way as your existing hal_kway_raw_call ---
    int col_offset = 0;

    for (int deg = 1; deg <= maxdeg; ++deg) {
        std::vector<std::vector<int>> combos;
        std::vector<int> cur;
        comb_recursive(cur, 0, 0, deg, p, combos);

        for (const auto& J : combos) {
            for (int i = 0; i < n; ++i) {
                for (int t = 0; t < n; ++t) {
                    double val = 1.0;
                    for (int j : J) {
                        val *= (Xcols[j][i] >= Xcols[j][t]) ? 1.0 : 0.0;
                        if (val == 0.0) break; // short-circuit
                    }
                    Hptr[i + (size_t)n * (col_offset + t)] = val;
                }
            }
            col_offset += n; // each combo contributes n columns
        }
    }
    // make a copy of H before centering
    SEXP H_copy_ = PROTECT(Rf_allocMatrix(REALSXP, n, q));
    double* H_copy_ptr = REAL(H_copy_);
    std::copy(Hptr, Hptr + (size_t)n * (size_t)q, H_copy_ptr);

    // --- center H columns if requested ---
    if (center) {
        for (int j = 0; j < q; ++j) {
            double col_mean = 0.0;
            for (int i = 0; i < n; ++i) {
                col_mean += Hptr[i + (size_t)n * j];
            }
            col_mean /= n;
            for (int i = 0; i < n; ++i) {
                Hptr[i + (size_t)n * j] -= col_mean;
            }
        }
    }                           

    // --- thin SVD via Gram matrix: G = H H^T (n x n) ---
    // Map H without copying
    Map<const MatrixXd> Hmap(Hptr, n, q);

    MatrixXd G = Hmap * Hmap.transpose();  // O(n^2 q), good when q >> n

    // Eigendecomposition (self-adjoint). Eigen returns ascending eigenvalues.
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(G, Eigen::ComputeEigenvectors);
    if (es.info() != Eigen::Success) {
        UNPROTECT(1);
        Rf_error("Eigen decomposition failed");
    }

    VectorXd evals = es.eigenvalues();     // ascending
    MatrixXd evecs = es.eigenvectors();    // columns correspond to evals

    // Clamp npc to valid range
    int npc_clamped = std::max(1, std::min(npc, n));
    // collect top-npc (largest) eigenvalues/vectors
    MatrixXd U = MatrixXd::Zero(n, npc_clamped);
    VectorXd d = VectorXd::Zero(npc_clamped);

    const double eps = 1e-12;
    for (int kidx = 0; kidx < npc_clamped; ++kidx) {
        int src = n - 1 - kidx;              // from largest down
        double lam = std::max(0.0, evals[src]);
        d[kidx] = std::sqrt(lam);
        U.col(kidx) = evecs.col(src);        // corresponding eigenvector
    }

    // V = H^T U D^{-1}  (q x npc)
    MatrixXd V = Hmap.transpose() * U;       // q x npc
    for (int kidx = 0; kidx < npc_clamped; ++kidx) {
        double sigma = d[kidx];
        if (sigma > eps) {
            V.col(kidx) /= sigma;
        } else {
            V.col(kidx).setZero();           // rank deficiency safeguard
        }
    }

    // --- marshal outputs back to R: list(H, U, d, V) ---
    SEXP U_ = PROTECT(Rf_allocMatrix(REALSXP, n, npc_clamped));
    SEXP d_ = PROTECT(Rf_allocVector(REALSXP, npc_clamped));
    SEXP V_ = PROTECT(Rf_allocMatrix(REALSXP, q, npc_clamped));

    // copy U
    {
        double* Up = REAL(U_);
        for (int j = 0; j < npc_clamped; ++j)
            for (int i = 0; i < n; ++i)
                Up[i + (size_t)n * j] = U(i, j);
    }
    // copy d
    {
        double* dp = REAL(d_);
        for (int k = 0; k < npc_clamped; ++k) dp[k] = d[k];
    }
    // copy V
    {
        double* Vp = REAL(V_);
        for (int j = 0; j < npc_clamped; ++j)
            for (int i = 0; i < q; ++i)
                Vp[i + (size_t)q * j] = V(i, j);
    }

    // build named list
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, H_copy_);
    SET_VECTOR_ELT(out, 1, U_);
    SET_VECTOR_ELT(out, 2, d_);
    SET_VECTOR_ELT(out, 3, V_);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(names, 0, Rf_mkChar("H"));
    SET_STRING_ELT(names, 1, Rf_mkChar("U"));
    SET_STRING_ELT(names, 2, Rf_mkChar("d"));
    SET_STRING_ELT(names, 3, Rf_mkChar("V"));
    Rf_setAttrib(out, R_NamesSymbol, names);

    UNPROTECT(7); // H_, H_copy_, U_, d_, V_, out, names
    return out;
}