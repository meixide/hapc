#include "hapc_core.hpp"
#include <vector>
#include <algorithm>

MatrixXd mkernel_call(const MatrixXd& X, int m, bool center) {
    const int n = X.rows();
    const int p = X.cols();
    
    if (n <= 0 || p <= 0) throw std::runtime_error("Invalid matrix dimensions");
    if (m < 0) m = 0;
    if (m > p) m = p;
    
    std::vector<double> psum(p + 1, 0.0);
    psum[0] = 1.0;
    
    for (int g = 1; g <= p; ++g) {
        const int kmax = std::min(g, m);
        double sum = 1.0;
        double binom = 1.0;
        for (int k = 1; k <= kmax; ++k) {
            binom = binom * (g - k + 1) / k;
            sum += binom;
        }
        psum[g] = sum;
    }
    
    MatrixXd K = MatrixXd::Zero(n, n);
    
    for (int a = 0; a < n; ++a) {
        for (int b = a; b < n; ++b) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                int g = 0;
                for (int j = 0; j < p; ++j) {
                    double thr = (X(a, j) < X(b, j)) ? X(a, j) : X(b, j);
                    if (X(i, j) <= thr) ++g;
                }
                s += psum[g] - 1.0;
            }
            K(a, b) = s;
            if (a != b) K(b, a) = s;
        }
    }
    
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