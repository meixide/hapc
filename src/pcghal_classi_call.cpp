// pcghal_classi_call.cpp
// Classification version with logistic loss

#include "hapc_core.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

static inline double sign_double(double x) {
    return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
}

OptimizerOutput pcghal_classi_call(const VectorXd& Y, const MatrixXd& Xtilde,
                                    const MatrixXd& ENn, const VectorXd& alpha0,
                                    int max_iter, double tol, double step_factor, 
                                    bool verbose) {
    const int n = Xtilde.rows();
    const int k = Xtilde.cols();
    const int q = ENn.rows();
    
    if (k <= 0 || n <= 0 || q <= 0) throw std::runtime_error("Invalid dimensions");
    
    const double eps = 1e-12;
    
    VectorXd alpha = alpha0;
    VectorXd mu(n), g(k), beta(q), a(k), g_tan(k);
    VectorXd sgn(q), Vt_s(k), gt_alpha(k), numer(q);
    
    auto risk = [&](const VectorXd& alph)->double {
        mu.noalias() = Xtilde * alph;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            double ymu = Y[i] * mu[i];
            if (ymu > 0) {
                sum += std::log1p(std::exp(-ymu));
            } else {
                sum += -ymu + std::log1p(std::exp(ymu));
            }
        }
        return sum / n;
    };
    
    auto grad = [&](const VectorXd& alph)->VectorXd {
        mu.noalias() = Xtilde * alph;
        VectorXd gtmp(k);
        
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n; ++i) {
                double ymu = Y[i] * mu[i];
                sum += Y[i] * Xtilde(i, j) / (1.0 + std::exp(ymu));
            }
            gtmp[j] = alph[j] * sum / n;
        }
        return gtmp;
    };
    
    MatrixXd alphaiters(max_iter + 1, k);
    alphaiters.row(0) = alpha.transpose();
    
    double R_old = risk(alpha);
    if (!std::isfinite(R_old)) throw std::runtime_error("Non-finite initial risk");
    
    if (verbose) {
        std::cout << "Classification mode (logistic loss)\n";
        std::cout << "Init | Risk = " << R_old << "  L1(beta) = " 
                  << (ENn * alpha).cwiseAbs().sum() << std::endl;
    }
    
    int iter_done = 0;
    for (int iter = 1; iter <= max_iter; ++iter) {
        g = grad(alpha);
        if (!g.allFinite()) throw std::runtime_error("Non-finite gradient");
        
        beta.noalias() = ENn * alpha;
        
        for (int i = 0; i < q; ++i) sgn[i] = sign_double(beta[i]);
        Vt_s.noalias() = ENn.transpose() * sgn;
        a = alpha.array() * Vt_s.array();
        
        double denom = a.squaredNorm();
        VectorXd proj(k);
        if (denom > eps) {
            proj = (g.dot(a) / denom) * a;
        } else {
            proj.setZero();
        }
        g_tan = g - proj;
        
        gt_alpha = g_tan.array() * alpha.array();
        numer.noalias() = ENn * gt_alpha;
        
        std::vector<double> valid;
        for (int i = 0; i < q; ++i) {
            if (std::abs(beta[i]) > eps) {
                double restr = numer[i] / beta[i];
                if (restr < 0.0) valid.push_back(-1.0 / restr);
            }
        }
        
        double step = 0.0;
        if (!valid.empty()) {
            double min_val = *std::min_element(valid.begin(), valid.end());
            step = step_factor * min_val;
        }
        if (!std::isfinite(step) || std::abs(step) > 1e6) step = 0.0;
        
        VectorXd alpha_new = alpha.array() * (1.0 + step * g_tan.array());
        double R_new = risk(alpha_new);
        
        if (verbose) {
            std::cout << "Iter " << iter << " | step=" << step << "  Risk=" << R_new 
                      << "  ||g_tan||=" << g_tan.norm() << std::endl;
        }
        
        alphaiters.row(iter) = alpha_new.transpose();
        iter_done = iter;
        
        if (!std::isfinite(R_new) || g_tan.norm() < tol) {
            alpha = alpha_new;
            R_old = R_new;
            break;
        }
        
        alpha = alpha_new;
        R_old = R_new;
    }
    
    VectorXd beta_final = ENn * alpha;
    return OptimizerOutput{
        alpha,
        alphaiters.topRows(iter_done + 1),
        beta_final,
        R_old,
        iter_done
    };
}