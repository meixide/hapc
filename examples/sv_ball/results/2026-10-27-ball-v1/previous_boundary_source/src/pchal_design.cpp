// Pure C++ core (shared by R and Python)
#include "hapc_core.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

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

DesignOutput pchal_des(const MatrixXd& X, int maxdeg, int npc, bool center) {
    const int n = X.rows();
    const int p = X.cols();
    
    if (n <= 0 || p <= 0) throw std::runtime_error("X must have positive dimensions");
    if (maxdeg < 1) throw std::runtime_error("max_degree must be >= 1");
    
    // --- copy X into column-major std::vectors for easy access ---
    std::vector<std::vector<double>> Xcols(p, std::vector<double>(n));
    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i)
            Xcols[j][i] = X(i, j);

    // --- compute total number of columns q = n * sum_{k=1..maxdeg} C(p,k) ---
    auto choose = [](int nn, int kk) -> long long {
        if (kk < 0 || kk > nn) return 0;
        long long r = 1;
        for (int i = 1; i <= kk; ++i) r = (r * (nn - i + 1)) / i;
        return r;
    };
    long long qll = 0;
    for (int k = 1; k <= maxdeg; ++k) qll += choose(p, k) * (long long)n;
    if (qll <= 0) throw std::runtime_error("invalid column count");
    if (qll > (long long)std::numeric_limits<int>::max())
        throw std::runtime_error("design matrix too wide for this build");
    const int q = (int)qll;

    MatrixXd H = MatrixXd::Zero(n, q);
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
                        if (val == 0.0) break;
                    }
                    H(i, col_offset + t) = val;
                }
            }
            col_offset += n;
        }
    }
    
    MatrixXd H_orig = H;

    if (center) {
        for (int j = 0; j < q; ++j) {
            double col_mean = H.col(j).mean();
            H.col(j).array() -= col_mean;
        }
    }

    MatrixXd G = H * H.transpose();
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(G, Eigen::ComputeEigenvectors);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigendecomposition failed");

    VectorXd evals = es.eigenvalues();
    MatrixXd evecs = es.eigenvectors();

    // Cap npc at n-1 when center=true (rank reduction due to centering)
    // Cap npc at n when center=false
    int max_npc = center ? (n - 1) : n;
    int npc_clamped = std::max(1, std::min(npc, max_npc));
    MatrixXd U = MatrixXd::Zero(n, npc_clamped);
    VectorXd d = VectorXd::Zero(npc_clamped);

    const double eps = 1e-12;
    for (int kidx = 0; kidx < npc_clamped; ++kidx) {
        int src = n - 1 - kidx;
        double lam = std::max(0.0, evals[src]);
        d[kidx] = std::sqrt(lam);
        U.col(kidx) = evecs.col(src);
    }

    MatrixXd V = H.transpose() * U;
    for (int kidx = 0; kidx < npc_clamped; ++kidx) {
        double sigma = d[kidx];
        if (sigma > eps) {
            V.col(kidx) /= sigma;
        } else {
            V.col(kidx).setZero();
        }
    }

    return DesignOutput{H_orig, U, d, V};
}