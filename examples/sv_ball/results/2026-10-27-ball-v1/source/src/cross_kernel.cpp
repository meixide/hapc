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
    
    // Uncentred cross kernel; with centring, the training kernel is needed
    // only for its column means.
    MatrixXd K = hal_kernel_uncentered(Xtr, Xte, Xtr, m, false);
    
    if (center) {
        MatrixXd Ktrain = hal_kernel_uncentered(Xtr, Xtr, Xtr, m, true);
        
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