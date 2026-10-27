#ifndef HAPC_SV_DESCENT_HPP
#define HAPC_SV_DESCENT_HPP
#include "hapc_core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Section 5 of arXiv:2603.18204v1: multiplicative descent on ||V alpha||_1=C.
// Retraction extends the path to zero beta coordinates, where division by beta
// is undefined. This is a boundary safeguard, not a global optimality result.
template<class Risk, class NegativeGradient>
OptimizerOutput sv_descent(const MatrixXd& V, const VectorXd& initial,
                           int max_iter, double tol, double step_factor,
                           bool verbose, const std::string& crit,
                           Risk risk, NegativeGradient negative_gradient) {
    if (max_iter < 0 || !std::isfinite(tol) || tol <= 0 ||
        !std::isfinite(step_factor) || step_factor <= 0 || step_factor >= 1)
        throw std::runtime_error("Require max_iter>=0, tol>0 and 0<step_factor<1");
    if (crit != "grad" && crit != "risk")
        throw std::runtime_error("crit must be 'grad' or 'risk'");
    VectorXd alpha = initial;
    const double C = (V * alpha).lpNorm<1>();
    if (!std::isfinite(C)) throw std::runtime_error("Non-finite initial SV radius");
    const double beta_tol = 64*std::numeric_limits<double>::epsilon()*C;
    MatrixXd history(max_iter + 1, alpha.size());
    history.row(0) = alpha.transpose();
    double current = risk(alpha);
    if (!std::isfinite(current)) throw std::runtime_error("Non-finite initial risk");
    int done = 0;
    std::string status = C == 0 ? "zero_radius" : "max_iter";
    double tangent_norm = 0;
    for (int it = 0; C > 0 && it < max_iter; ++it) {
        const VectorXd ng = negative_gradient(alpha); // -dR/dalpha
        if (!ng.allFinite()) throw std::runtime_error("Non-finite gradient");
        const VectorXd beta = V * alpha;
        const VectorXd signs = beta.unaryExpr([beta_tol](double x) {
            return x > beta_tol ? 1.0 : (x < -beta_tol ? -1.0 : 0.0);
        });
        const VectorXd normal = alpha.array() * (V.transpose()*signs).array();
        VectorXd h = alpha.array() * ng.array(); // negative multiplicative gradient
        // Normalize before projection: an absolute cutoff here breaks small C.
        const double normal_norm = normal.norm();
        if (normal_norm > 0) {
            const VectorXd unit = normal / normal_norm;
            h -= unit * unit.dot(h);
        }
        tangent_norm = h.norm();
        if (tangent_norm < tol) { status = "tangent_stationary"; break; }
        const VectorXd direction = alpha.array() * h.array();
        const VectorXd dbeta = V * direction;
        // Numerical zeros must not force a vanishing step; retraction below
        // enforces the exact radius even when these coordinates become active.
        double max_ratio = 0;
        for (int j = 0; j < beta.size(); ++j)
            if (std::abs(beta[j]) > beta_tol) max_ratio = std::max(max_ratio, std::abs(dbeta[j]/beta[j]));
        double step = max_ratio > 0 ? step_factor/max_ratio : 1.0;
        if (!std::isfinite(step)) step = 1.0;
        bool accepted = false;
        VectorXd trial;
        double candidate = current;
        for (int backtrack = 0; backtrack < 80; ++backtrack, step *= 0.5) {
            trial = alpha + step * direction;
            const double trial_C = (V * trial).lpNorm<1>();
            if (!trial.allFinite() || !std::isfinite(trial_C) || trial_C <= 0) continue;
            trial *= C / trial_C;
            // Evaluate descent along the actual retracted displacement. At a
            // zero beta the old tangent formula alone need not give descent.
            const double decrease = ng.dot(trial-alpha);
            if (decrease <= 0 || (trial-alpha).norm() == 0) continue;
            candidate = risk(trial);
            if (std::isfinite(candidate) && candidate <= current - 1e-4*decrease && candidate < current) {
                accepted = true;
                break;
            }
        }
        if (!accepted) { status = "line_search_stalled"; break; }
        const double improvement = current - candidate;
        alpha = trial;
        current = candidate;
        history.row(++done) = alpha.transpose();
        if (verbose) std::cout << "Iter " << done << " risk=" << current
                               << " SV=" << (V*alpha).lpNorm<1>() << std::endl;
        if (crit == "risk" && improvement < tol) { status = "risk_tolerance"; break; }
    }
    // Recompute at the returned iterate; never attach a pre-step score to it.
    if (C > 0) {
        const VectorXd ng = negative_gradient(alpha);
        const VectorXd beta = V*alpha;
        const VectorXd signs = beta.unaryExpr([beta_tol](double x) { return x>beta_tol ? 1.0 : (x<-beta_tol ? -1.0 : 0.0); });
        VectorXd normal = alpha.array()*(V.transpose()*signs).array();
        VectorXd h = alpha.array()*ng.array();
        if (normal.norm()>0) { normal.normalize(); h -= normal*normal.dot(h); }
        tangent_norm = h.norm();
    }
    OptimizerOutput out{alpha, history.topRows(done+1), V*alpha, current, done};
    out.status = status;
    out.sv_radius = C;
    out.tangent_norm = tangent_norm;
    if (verbose) std::cout << "Stop: " << status << " (multiplicative stationarity only)" << std::endl;
    return out;
}

inline void validate_sv_data(const VectorXd& Y, const MatrixXd& Z,
                             const MatrixXd& V, const VectorXd& alpha) {
    if (Z.rows() <= 0 || Z.cols() <= 0 || V.rows() <= 0 ||
        Y.size() != Z.rows() || V.cols() != Z.cols() || alpha.size() != Z.cols())
        throw std::runtime_error("Invalid SV solver dimensions");
    if (!Y.allFinite() || !Z.allFinite() || !V.allFinite() || !alpha.allFinite())
        throw std::runtime_error("SV solver inputs must be finite");
}
#endif
