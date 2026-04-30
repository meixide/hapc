#include "hapc_core.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

static inline double sign_double(double x) {
    return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
}

OptimizerOutput pcghal_call(const VectorXd& Y, const MatrixXd& Xtilde, 
                            const MatrixXd& ENn, const VectorXd& alpha0,
                            int max_iter, double tol, double step_factor, 
                            bool verbose, const std::string& crit) {
    const int n = Xtilde.rows();
    const int k = Xtilde.cols();
    const int q = ENn.rows();
    
    if (k <= 0 || n <= 0 || q <= 0) throw std::runtime_error("Invalid dimensions");
    if (alpha0.size() != k) throw std::runtime_error("alpha0 size mismatch");
    
    const double eps = 1e-12;
    
    VectorXd alpha = alpha0;
    VectorXd mu(n), g(k), beta(q), a(k), g_tan(k);
    VectorXd sgn(q), Vt_s(k), gt_alpha(k), numer(q);
    
    auto risk = [&](const VectorXd& a)->double {
        mu.noalias() = Xtilde * a;
        return (Y - mu).squaredNorm() / n;
    };
    
    auto grad = [&](const VectorXd& a)->VectorXd {
        mu.noalias() = Xtilde * a;
        VectorXd r = Y - mu;
        VectorXd gtmp(k);
        for (int j = 0; j < k; ++j)
            gtmp[j] = a[j] * 2.0 * Xtilde.col(j).dot(r) / n;
        return gtmp;
    };
    
    MatrixXd alphaiters(max_iter + 1, k);
    alphaiters.row(0) = alpha.transpose();
    
    double R_old = risk(alpha);
    if (!std::isfinite(R_old)) throw std::runtime_error("Non-finite initial risk");
    
    if (verbose) {
        std::cout << "Init | Risk = " << R_old << "  L1(beta) = " 
                  << (ENn * alpha).cwiseAbs().sum() << std::endl;
    }
    
    int iter_done = 0;
    // Track last 100 iterations: store (alpha, g_tan_norm) pairs
    const int lookback_window = 100;
    std::vector<std::pair<VectorXd, double>> recent_iterations;
    recent_iterations.reserve(lookback_window);
    
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
        
        double g_tan_norm = g_tan.norm();
        
        gt_alpha = g_tan.array() * alpha.array();
        numer.noalias() = ENn * gt_alpha;
        
        // Step size selection rule: δ < 1 / max_j |h*(j)|
        // where h*(j) = Σ_m h(m)α(m)E(j,m) / β(α)(j) = numer[j] / beta[j]
        double max_abs_hstar = 0.0;
        for (int i = 0; i < q; ++i) {
            if (std::abs(beta[i]) > eps) {
                // Compute h*(j) = numer[j] / beta[j]
                double hstar_j = numer[i] / beta[i];
                double abs_hstar_j = std::abs(hstar_j);
                if (abs_hstar_j > max_abs_hstar) {
                    max_abs_hstar = abs_hstar_j;
                }
            }
        }
        
        // Step size: δ = step_factor * (1 / max_j |h*(j)|)
        // This ensures δ < 1 / max_j |h*(j)| when step_factor < 1
        double step = 0.0;
        if (max_abs_hstar > eps) {
            step = step_factor / max_abs_hstar;
        }
        if (!std::isfinite(step) || std::abs(step) > 1e6) step = 0.0;
        
        VectorXd alpha_new = alpha.array() * (1.0 + step * g_tan.array());
        double R_new = risk(alpha_new);
        
        if (verbose) {
            std::cout << "Iter " << iter << " | step=" << step << "  Risk=" << R_new 
                      << "  dRisk=" << (R_old - R_new) << "  ||g_tan||=" << g_tan_norm << std::endl;
        }
        
        // Store this iteration in the lookback window (keep only last 100)
        if (recent_iterations.size() >= lookback_window) {
            recent_iterations.erase(recent_iterations.begin());
        }
        recent_iterations.push_back(std::make_pair(alpha_new, g_tan_norm));
        
        alphaiters.row(iter) = alpha_new.transpose();
        iter_done = iter;
        
        bool should_stop = false;
        if (crit == "grad") {
            should_stop = !std::isfinite(R_new) || (g_tan_norm < tol);
        } else if (crit == "risk") {
            should_stop = !std::isfinite(R_new) || ((R_old - R_new) < tol);
        }
        
        if (should_stop) {
            alpha = alpha_new;
            R_old = R_new;
            break;
        }
        
        alpha = alpha_new;
        R_old = R_new;
    }
    
    // Select best solution from last 100 iterations (lowest ||g_tan||)
    if (!recent_iterations.empty()) {
        auto best_iter = std::min_element(recent_iterations.begin(), recent_iterations.end(),
            [](const std::pair<VectorXd, double>& a, const std::pair<VectorXd, double>& b) {
                return a.second < b.second;
            });
        alpha = best_iter->first;
        R_old = risk(alpha);
        if (verbose) {
            std::cout << "\n[Best solution] Selected from last " << recent_iterations.size() 
                      << " iterations: ||g_tan||=" << best_iter->second << std::endl;
        }
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