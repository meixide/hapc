#include "hapc_core.hpp"
#include <vector>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>

// Number of worker threads for the kernel loops: HAPC_NUM_THREADS if set to a
// positive integer, otherwise std::thread::hardware_concurrency().
int hapc_num_threads() {
    if (const char* s = std::getenv("HAPC_NUM_THREADS")) {
        const int v = std::atoi(s);
        if (v >= 1) return v;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : static_cast<int>(hc);
}

// Uncentred HAL kernel between the rows of Xa and the rows of Xb, with knots
// at the rows of Xtr:
//
//   K(a, b) = sum_i f(g_i),  g_i = #{ j : Xtr(i, j) <= min(Xa(a, j), Xb(b, j)) },
//   f(g)    = sum_{k = 1..min(g, m)} choose(g, k),
//
// i.e. the number of HAL basis functions (knot i, coordinate subset of size
// <= m) that are 1 at both points.  `symmetric` (Xa == Xb) fills the upper
// triangle and mirrors it.  Rows of Xa are distributed over threads; each
// entry is accumulated over i in the same order as the former single-threaded
// loop, so results are bit-identical.
MatrixXd hal_kernel_uncentered(const MatrixXd& Xtr, const MatrixXd& Xa,
                               const MatrixXd& Xb, int m, bool symmetric) {
    const int n = Xtr.rows();
    const int p = Xtr.cols();
    const int na = Xa.rows();
    const int nb = Xb.rows();
    if (n <= 0 || p <= 0 || na <= 0 || nb <= 0) throw std::runtime_error("Invalid matrix dimensions");
    if (Xa.cols() != p || Xb.cols() != p) throw std::runtime_error("Column mismatch");
    if (m < 0) m = 0;
    if (m > p) m = p;

    std::vector<double> f(p + 1, 0.0);
    for (int g = 1; g <= p; ++g) {
        const int kmax = std::min(g, m);
        double sum = 0.0;
        double binom = 1.0;
        for (int k = 1; k <= kmax; ++k) {
            binom = binom * (g - k + 1) / k;
            sum += binom;
        }
        f[g] = sum;
    }

    // Knots in row-major order so the inner loop reads p contiguous values.
    std::vector<double> knots(static_cast<size_t>(n) * p);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            knots[static_cast<size_t>(i) * p + j] = Xtr(i, j);

    MatrixXd K(na, nb);
    std::atomic<int> next_row(0);
    auto worker = [&]() {
        std::vector<double> thr(p);
        for (int a = next_row++; a < na; a = next_row++) {
            for (int b = symmetric ? a : 0; b < nb; ++b) {
                for (int j = 0; j < p; ++j) thr[j] = std::min(Xa(a, j), Xb(b, j));
                double s = 0.0;
                const double* xi = knots.data();
                for (int i = 0; i < n; ++i, xi += p) {
                    int g = 0;
                    for (int j = 0; j < p; ++j) g += (xi[j] <= thr[j]);
                    s += f[g];
                }
                K(a, b) = s;
                // Row a's thread writes K(b, a) for b > a only; row b's thread
                // writes K(b, c) for c >= b, so the writes never overlap.
                if (symmetric && b != a) K(b, a) = s;
            }
        }
    };

    const int nt = std::max(1, std::min(hapc_num_threads(), na));
    if (nt == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nt);
        for (int t = 0; t < nt; ++t) pool.emplace_back(worker);
        for (auto& th : pool) th.join();
    }
    return K;
}

MatrixXd mkernel_call(const MatrixXd& X, int m, bool center) {
    const int n = X.rows();
    const int p = X.cols();

    if (n <= 0 || p <= 0) throw std::runtime_error("Invalid matrix dimensions");

    MatrixXd K = hal_kernel_uncentered(X, X, X, m, true);

    if (center) {
        std::vector<double> rowmean(n, 0.0);
        std::vector<double> colmean(n, 0.0);
        double grand = 0.0;

        for (int a = 0; a < n; ++a) {
            double rs = K.row(a).sum();
            rowmean[a] = rs / n;
            grand += rs;
        }
        grand /= (n * n);

        for (int b = 0; b < n; ++b) {
            double cs = K.col(b).sum();
            colmean[b] = cs / n;
        }

        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                K(a, b) = K(a, b) - rowmean[a] - colmean[b] + grand;
            }
        }
    }

    return K;
}
