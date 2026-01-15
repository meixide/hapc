#include "hapc_core.hpp"
#include <vector>
#include <algorithm>

MatrixXd kernel_cross_call(const MatrixXd& Xtr, const MatrixXd& Xte, int m, bool center) {
    const int n = Xtr.rows();
    const int p = Xtr.cols();
    const int m_test = Xte.rows();
    
    if (n <= 0 || p <= 0 || m_test <= 0) throw std::runtime_error("Invalid matrix dimensions");
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
    
    MatrixXd K = MatrixXd::Zero(m_test, n);
    
    for (int t = 0; t < m_test; ++t) {
        for (int b = 0; b < n; ++b) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                int g = 0;
                for (int j = 0; j < p; ++j) {
                    double thr = std::min(Xte(t, j), Xtr(b, j));
                    if (Xtr(i, j) <= thr) ++g;
                }
                s += psum[g] - 1.0;
            }
            K(t, b) = s;
        }
    }
    
    if (center) {
        MatrixXd Ktrain = MatrixXd::Zero(n, n);
        for (int a = 0; a < n; ++a) {
            for (int b = a; b < n; ++b) {
                double s = 0.0;
                for (int i = 0; i < n; ++i) {
                    int g = 0;
                    for (int j = 0; j < p; ++j) {
                        double thr = std::min(Xtr(a, j), Xtr(b, j));
                        if (Xtr(i, j) <= thr) ++g;
                    }
                    s += psum[g] - 1.0;
                }
                Ktrain(a, b) = s;
                if (a != b) Ktrain(b, a) = s;
            }
        }
        
        std::vector<double> colmean_train(n, 0.0);
        double grand_train = 0.0;
        for (int b = 0; b < n; ++b) {
            double cs = Ktrain.col(b).sum();
            colmean_train[b] = cs / n;
            grand_train += cs;
        }
        grand_train /= (n * n);
        
        std::vector<double> rowmean_test(m_test, 0.0);
        for (int t = 0; t < m_test; ++t) {
            double rs = K.row(t).sum();
            rowmean_test[t] = rs / n;
        }
        
        for (int t = 0; t < m_test; ++t) {
            for (int b = 0; b < n; ++b) {
                K(t, b) = K(t, b) - rowmean_test[t] - colmean_train[b] + grand_train;
            }
        }
    }
    
    return K;
}