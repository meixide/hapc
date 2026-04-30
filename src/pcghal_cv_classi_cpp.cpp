// pcghal_cv_classi_cpp.cpp
// Pure-C++ binomial (logistic) HAPC: shared by both R and Python.
//
// This file provides:
//   - logistic_ridge_init  : Newton-Raphson logistic ridge regression
//                            (pure C++ counterpart of `logistic_call`).
//   - pcghal_cv_classi_python : k-fold CV for the classification model
//                               (pure C++ counterpart of R's
//                               `pchal_cv_classi_call`).
//
// Both R (`pchal_cv_classi_call` thin SEXP wrapper) and Python (pybind11
// binding `pcghal_cv_classi_fit`) call this code so fold partitioning,
// initialiser, optimisation and risk computation are bit-identical across
// languages.

#include "hapc_core.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Newton-Raphson logistic ridge regression (replicates logistic_call.cpp).
// Y must be encoded in {-1, +1}.  Reuses the same fixed iteration budget
// and tolerance as the R-only path, including the (idiosyncratic) update
// rule `beta := delta_beta` (i.e. solving the full normal equation each
// iteration, treating the IRLS working response as the regression target).
// ---------------------------------------------------------------------------
VectorXd logistic_ridge_init(const VectorXd& Y_pm1, const MatrixXd& X, double lambda) {
    const int n = X.rows();
    const int p = X.cols();
    if (Y_pm1.size() != n) {
        throw std::runtime_error("logistic_ridge_init: Y length must match nrow(X).");
    }
    // Match logistic_call: lambda is multiplied by n internally.
    const double lam = lambda * n;
    const int max_iter = 100;
    const double tol = 1e-8;

    // logistic_call expects Y in {-1,+1} but treats it via the GLM update with
    // the {0,1} working response.  We replicate that behaviour exactly: convert
    // back to a {0,1} response y01 = (Y_pm1 + 1) / 2 to compute mu/working z.
    VectorXd y01(n);
    for (int i = 0; i < n; ++i) y01[i] = (Y_pm1[i] > 0) ? 1.0 : 0.0;

    VectorXd beta = VectorXd::Zero(p);
    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd eta = X * beta;
        VectorXd mu = (1.0 + (-eta.array()).exp()).inverse();
        VectorXd w = mu.array() * (1.0 - mu.array());
        w = w.array().max(1e-8);
        VectorXd z = eta.array() + (y01.array() - mu.array()) / w.array();

        MatrixXd XtW = X.transpose() * w.asDiagonal();
        MatrixXd H = XtW * X;
        H.diagonal().array() += lam;
        VectorXd grad = XtW * z - lam * beta;
        VectorXd delta_beta = H.ldlt().solve(grad);

        VectorXd beta_old = beta;
        beta = delta_beta;  // matches the R logistic_call formula exactly
        if ((beta - beta_old).norm() < tol) break;
    }
    return beta;
}

// ---------------------------------------------------------------------------
// Build the Eigen-friendly "Xtilde = U_top * diag(d_top)" representation,
// returning final_npc (which may be capped by the design rank).
// ---------------------------------------------------------------------------
static int compute_classi_design(const MatrixXd& X, int maxdeg, int npc, bool center,
                                 MatrixXd& Xtilde_out, MatrixXd& E_Nn_out,
                                 MatrixXd& U_full_out, VectorXd& d_full_out) {
    DesignOutput des = pchal_des(X, maxdeg, npc, center);
    const int final_npc = (npc < (int)des.d.size()) ? npc : (int)des.d.size();
    Xtilde_out = des.U.leftCols(final_npc) * des.d.head(final_npc).asDiagonal();
    E_Nn_out = des.V.leftCols(final_npc);
    U_full_out = des.U.leftCols(final_npc);
    d_full_out = des.d.head(final_npc);
    return final_npc;
}

// ---------------------------------------------------------------------------
// Same fold partition strategy used everywhere in this package:
//   - block partition with the last fold absorbing the remainder
//   - shuffled with std::mt19937(12345)
// ---------------------------------------------------------------------------
static std::vector<int> make_folds(int n, int K) {
    std::vector<int> folds(n);
    const int fold_size = n / K;
    for (int i = 0; i < n; ++i) folds[i] = (i / fold_size) + 1;
    for (int i = fold_size * K; i < n; ++i) folds[i] = K;
    std::mt19937 rng(12345);
    std::shuffle(folds.begin(), folds.end(), rng);
    return folds;
}

// ---------------------------------------------------------------------------
// Public API: k-fold logistic HAPC CV.
// ---------------------------------------------------------------------------
// Helper: per-lambda full-data fit (used by the nfolds<=1 single-λ path and
// for the post-CV refit). When `with_pgd == false`, returns the logistic-ridge
// initialiser α directly with its training logistic risk; otherwise runs the
// PGD step on top of it (norm="sv").
static OptimizerOutput logistic_full_fit(const VectorXd& Y_pm1,
                                          const MatrixXd& Xtilde,
                                          const MatrixXd& E_Nn,
                                          double lambda,
                                          int max_iter, double tol,
                                          double step_factor, bool verbose,
                                          bool with_pgd) {
    VectorXd alpha0 = logistic_ridge_init(Y_pm1, Xtilde, lambda);
    if (with_pgd) {
        return pcghal_classi_call(Y_pm1, Xtilde, E_Nn, alpha0,
                                  max_iter, tol, step_factor, verbose);
    }
    // Logistic-ridge-only path: assemble the same OptimizerOutput shape with
    // logistic training risk evaluated on (Y_pm1, Xtilde, alpha0).
    const int n = Xtilde.rows();
    VectorXd eta = Xtilde * alpha0;
    double risk = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ymu = Y_pm1[i] * eta[i];
        risk += (ymu > 0) ? std::log1p(std::exp(-ymu))
                          : -ymu + std::log1p(std::exp(ymu));
    }
    risk /= n;
    OptimizerOutput out;
    out.alpha = alpha0;
    out.alphaiters = MatrixXd::Zero(0, alpha0.size());
    out.beta = E_Nn * alpha0;
    out.risk = risk;
    out.iter = 0;
    return out;
}

CVClassiOutput pcghal_cv_classi_python(const MatrixXd& X, const VectorXd& Y,
                                        int maxdeg, int npc,
                                        const std::vector<double>& lambdas,
                                        int nfolds, const MatrixXd& predict_data,
                                        int max_iter, double tol, double step_factor,
                                        bool verbose, bool center,
                                        bool with_pgd) {
    const int n = X.rows();
    const int p = X.cols();
    if (Y.size() != n) throw std::runtime_error("pcghal_cv_classi: length(Y) != nrow(X)");
    for (int i = 0; i < n; ++i) {
        if (Y[i] != 0.0 && Y[i] != 1.0) {
            throw std::runtime_error("pcghal_cv_classi: Y must be 0/1");
        }
    }
    const int L = (int)lambdas.size();
    if (L <= 0) throw std::runtime_error("pcghal_cv_classi: lambdas must be non-empty");

    // Cap npc as in cv_classi.cpp
    int npc_eff = npc;
    if (center) {
        if (npc_eff >= n) npc_eff = n - 1;
    } else {
        if (npc_eff > n) npc_eff = n;
    }

    MatrixXd Xtilde, E_Nn, U_top;
    VectorXd d_top;
    const int final_npc = compute_classi_design(X, maxdeg, npc_eff, center,
                                                 Xtilde, E_Nn, U_top, d_top);

    // Y in {-1,+1} for the optimiser
    VectorXd Y_pm1(n);
    for (int i = 0; i < n; ++i) Y_pm1[i] = (Y[i] == 1.0) ? 1.0 : -1.0;

    // Degenerate case: R `hapc(family="binomial", …)` passes nfolds=1 with a
    // single λ — there is no proper train/test split.  Fit on full data and
    // report training logistic risk for each λ (when L>1, pick best by risk).
    if (nfolds <= 1) {
        std::vector<double> deviances(L, std::numeric_limits<double>::quiet_NaN());
        VectorXd best_alpha;
        double best_lambda = lambdas[0];
        double best_val = std::numeric_limits<double>::infinity();
        for (int j = 0; j < L; ++j) {
            const double lam = lambdas[j];
            OptimizerOutput full_out = logistic_full_fit(
                Y_pm1, Xtilde, E_Nn, lam, max_iter, tol, step_factor,
                verbose, with_pgd);
            deviances[j] = full_out.risk;
            if (full_out.risk < best_val) {
                best_val = full_out.risk;
                best_lambda = lam;
                best_alpha = full_out.alpha;
            }
        }
        VectorXd predictions = VectorXd::Zero(0);
        if (predict_data.rows() > 0) {
            if (predict_data.cols() != p) {
                throw std::runtime_error("predict must have same #cols as X");
            }
            MatrixXd Ktest = kernel_cross_call(X, predict_data, maxdeg, center);
            VectorXd d_inv = d_top.cwiseInverse();
            VectorXd v = U_top * (d_inv.asDiagonal() * best_alpha);
            VectorXd eta_pred = Ktest * v;
            predictions = (1.0 + (-eta_pred.array()).exp()).inverse();
        }
        CVClassiOutput out;
        out.deviances = deviances;
        out.lambdas = lambdas;
        out.best_lambda = best_lambda;
        out.best_alpha = best_alpha;
        out.predictions = predictions;
        return out;
    }

    std::vector<int> folds = make_folds(n, nfolds);

    // fold_error[k][j] = average deviance of fold k at lambda j
    MatrixXd fold_error = MatrixXd::Constant(nfolds, L, std::numeric_limits<double>::quiet_NaN());

    for (int j = 0; j < L; ++j) {
        const double lambda = lambdas[j];
        for (int k = 1; k <= nfolds; ++k) {
            std::vector<int> tr_idx, te_idx;
            tr_idx.reserve(n); te_idx.reserve(n / nfolds + 1);
            for (int i = 0; i < n; ++i) {
                if (folds[i] == k) te_idx.push_back(i);
                else tr_idx.push_back(i);
            }
            const int ntr = (int)tr_idx.size();
            const int nte = (int)te_idx.size();
            if (ntr == 0 || nte == 0) continue;

            MatrixXd Xtr(ntr, final_npc), Xte(nte, final_npc);
            VectorXd Ytr_pm1(ntr), Yte01(nte);
            for (int i = 0; i < ntr; ++i) {
                Xtr.row(i) = Xtilde.row(tr_idx[i]);
                Ytr_pm1[i] = Y_pm1[tr_idx[i]];
            }
            for (int i = 0; i < nte; ++i) {
                Xte.row(i) = Xtilde.row(te_idx[i]);
                Yte01[i] = Y[te_idx[i]];
            }

            VectorXd alpha0 = logistic_ridge_init(Ytr_pm1, Xtr, lambda);
            VectorXd alpha_fold;
            if (with_pgd) {
                OptimizerOutput out = pcghal_classi_call(Ytr_pm1, Xtr, E_Nn, alpha0,
                                                          max_iter, tol, step_factor,
                                                          verbose);
                alpha_fold = out.alpha;
            } else {
                alpha_fold = alpha0;  // logistic ridge only (norm="2")
            }

            VectorXd eta = Xte * alpha_fold;
            VectorXd probs = (1.0 + (-eta.array()).exp()).inverse();
            double dev = 0.0;
            for (int i = 0; i < nte; ++i) {
                double pi = std::max(1e-15, std::min(1.0 - 1e-15, probs[i]));
                dev += (Yte01[i] == 1.0) ? -std::log(pi) : -std::log(1.0 - pi);
            }
            fold_error(k - 1, j) = dev / nte;
        }
    }

    // Aggregate across folds.
    std::vector<double> deviances(L, std::numeric_limits<double>::quiet_NaN());
    for (int j = 0; j < L; ++j) {
        double sum = 0.0; int cnt = 0;
        for (int k = 0; k < nfolds; ++k) {
            double v = fold_error(k, j);
            if (!std::isnan(v)) { sum += v; cnt++; }
        }
        if (cnt > 0) deviances[j] = sum / cnt;
    }

    int best_idx = 0;
    double best_val = deviances[0];
    for (int j = 1; j < L; ++j) {
        if (std::isnan(deviances[j])) continue;
        if (std::isnan(best_val) || deviances[j] < best_val) {
            best_val = deviances[j];
            best_idx = j;
        }
    }
    const double best_lambda = lambdas[best_idx];

    // Refit on full data at best_lambda (logistic ridge ± PGD).
    OptimizerOutput full_out = logistic_full_fit(
        Y_pm1, Xtilde, E_Nn, best_lambda,
        max_iter, tol, step_factor, verbose, with_pgd);

    // Predict on `predict_data` if supplied (else empty vector).
    VectorXd predictions = VectorXd::Zero(0);
    if (predict_data.rows() > 0) {
        if (predict_data.cols() != p) {
            throw std::runtime_error("predict must have same #cols as X");
        }
        MatrixXd Ktest = kernel_cross_call(X, predict_data, maxdeg, center);
        VectorXd d_inv = d_top.cwiseInverse();
        VectorXd v = U_top * (d_inv.asDiagonal() * full_out.alpha);
        VectorXd eta_pred = Ktest * v;
        predictions = (1.0 + (-eta_pred.array()).exp()).inverse();
    }

    CVClassiOutput out;
    out.deviances = deviances;
    out.lambdas = lambdas;
    out.best_lambda = best_lambda;
    out.best_alpha = full_out.alpha;
    out.predictions = predictions;
    return out;
}
