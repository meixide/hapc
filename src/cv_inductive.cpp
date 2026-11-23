// Gaussian HAPC cross-validation for norm "1" (PC-HAL) and "2" (PC-HAR) with
// an inductive basis per training part. Pure C++, shared by R and Python.
//
// For every training part (V-fold training fold, or the training part of a
// hold-out split) the HAL kernel, its centring and its eigendecomposition are
// built from the training rows only. Held-out rows enter only through the
// cross-kernel with the training knots and the training centring
// (kernel_cross_call). The PC scores Z = U S of a training part then have
// Z'Z = S^2, so the closed forms in ridge_call / fast_pchal_call are the exact
// minimisers of the training-part problems
//     (1 / 2 n_tr) ||y_c - U S alpha||^2 + (lambda / 2) ||alpha||_2^2   (ridge)
//     (1 / 2 n_tr) ||y_c - U S alpha||^2 +  lambda      ||alpha||_1     (lasso)
// with y_c the training outcomes minus their training mean (center = true).
//
// With inductive = false the fold fits use the rows of the full-sample basis,
// as fasthal_cv_call / fasthal_cv_python always did (kept for comparison).

#include "hapc_core.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();

struct Basis {
    MatrixXd U;   // n_b x r, orthonormal columns
    VectorXd D2;  // r squared singular values, descending, numerically positive
};

// Top eigenpairs of a (centred) HAL kernel. Eigenvalues at the level of
// roundoff are dropped (same tolerance as pchal_des), as is the constant
// direction removed by centring; at most npc are kept.
Basis kernel_basis(const MatrixXd& K, int npc, bool center) {
    const int n = (int)K.rows();
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(K);
    if (es.info() != Eigen::Success) throw std::runtime_error("Eigendecomposition failed");
    const VectorXd& ev = es.eigenvalues();  // ascending
    const double tol = 8.0 * std::numeric_limits<double>::epsilon() * n *
                       std::max(0.0, ev.maxCoeff());
    const int cap = std::min(npc, center ? n - 1 : n);
    int r = 0;
    while (r < cap && ev(n - 1 - r) > tol) ++r;
    if (r == 0) throw std::runtime_error("HAL kernel of a training part has numerical rank 0");
    Basis b;
    b.U.resize(n, r);
    b.D2.resize(r);
    for (int i = 0; i < r; ++i) {
        b.D2(i) = ev(n - 1 - i);
        b.U.col(i) = es.eigenvectors().col(n - 1 - i);
    }
    return b;
}

MatrixXd rows_of(const MatrixXd& A, const std::vector<int>& idx) {
    MatrixXd out(idx.size(), A.cols());
    for (size_t i = 0; i < idx.size(); ++i) out.row(i) = A.row(idx[i]);
    return out;
}

VectorXd rows_of(const VectorXd& a, const std::vector<int>& idx) {
    VectorXd out(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) out(i) = a(idx[i]);
    return out;
}

// Scores of new points in a basis built on Xb: (h(x) - hbar)' V = Kc(x, Xb) U S^{-1}.
MatrixXd new_scores(const MatrixXd& Xb, const MatrixXd& Xnew, const Basis& b, int maxdeg,
                    bool center) {
    MatrixXd Kx = kernel_cross_call(Xb, Xnew, maxdeg, center);
    return Kx * b.U * b.D2.cwiseSqrt().cwiseInverse().asDiagonal();
}

VectorXd fit_alpha(const Basis& b, const VectorXd& yc, double lambda, bool l1) {
    return l1 ? fast_pchal_call(b.U, b.D2, yc, lambda) : ridge_call(yc, b.U, b.D2, lambda);
}

void clip_to(VectorXd& v, double lo, double hi) {
    for (int i = 0; i < v.size(); ++i) v(i) = std::min(hi, std::max(lo, v(i)));
}

struct Part {
    std::vector<int> tr, te;
    Basis basis;
    MatrixXd Zte;    // scores of the held-out rows in this part's basis
    MatrixXd Zpred;  // scores of `predict` (only when needed)
    double ymean = 0.0, lo = 0.0, hi = 0.0;
    VectorXd yc;
};

}  // namespace

GeneralCVOutput fasthal_cv_general(const MatrixXd& X, const VectorXd& Y, int npc,
                                   const std::vector<double>& lambdas_in, int nlambda,
                                   int nfolds, const MatrixXd& predict, int maxdeg,
                                   bool center, bool l1, bool inductive,
                                   const std::string& selection, double train_frac,
                                   bool clip) {
    const int n = (int)X.rows();
    const int p = (int)X.cols();
    if (n < 3 || p < 1) throw std::runtime_error("X must have at least 3 rows and 1 column");
    if (Y.size() != n) throw std::runtime_error("length(Y) must equal nrow(X)");
    if (npc < 1) throw std::runtime_error("npcs must be >= 1");
    const bool holdout = (selection == "holdout");
    if (!holdout && selection != "refit" && selection != "fold_average")
        throw std::runtime_error("selection must be 'refit', 'fold_average' or 'holdout'");
    if (predict.rows() > 0 && predict.cols() != p)
        throw std::runtime_error("predict must have the same number of columns as X");
    const bool want_pred = predict.rows() > 0;

    // ---- partition: same folds as fasthal_cv_call; hold-out by a shuffled index
    std::vector<int> foldid(n);
    std::vector<Part> parts;
    if (holdout) {
        if (!(train_frac > 0.0 && train_frac < 1.0))
            throw std::runtime_error("train_frac must lie in (0, 1)");
        const int n1 = (int)std::lround(train_frac * n);
        if (n1 < 2 || n1 > n - 1)
            throw std::runtime_error("train_frac leaves fewer than 2 training or 1 validation rows");
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 rng(12345);
        std::shuffle(perm.begin(), perm.end(), rng);
        Part pt;
        for (int i = 0; i < n; ++i) foldid[perm[i]] = (i < n1) ? 0 : 1;
        for (int i = 0; i < n; ++i) (foldid[i] == 0 ? pt.tr : pt.te).push_back(i);
        parts.push_back(pt);
    } else {
        if (nfolds < 2 || nfolds > n) throw std::runtime_error("nfolds must lie in [2, n]");
        const int fold_size = n / nfolds;
        for (int i = 0; i < n; ++i) foldid[i] = (i / fold_size) + 1;
        for (int i = fold_size * nfolds; i < n; ++i) foldid[i] = nfolds;
        std::mt19937 rng(12345);
        std::shuffle(foldid.begin(), foldid.end(), rng);
        for (int k = 1; k <= nfolds; ++k) {
            Part pt;
            for (int i = 0; i < n; ++i) (foldid[i] == k ? pt.te : pt.tr).push_back(i);
            if (pt.tr.size() < 2 || pt.te.empty())
                throw std::runtime_error("every fold needs >= 2 training and >= 1 held-out rows");
            parts.push_back(pt);
        }
    }

    // ---- full-sample basis: for the refit, and for the transductive fold fits
    const bool need_full = !inductive || selection == "refit";
    Basis full;
    MatrixXd Zfull;  // U S of the full sample (transductive held-out scores)
    if (need_full) {
        full = kernel_basis(mkernel_call(X, maxdeg, center), npc, center);
        Zfull = full.U * full.D2.cwiseSqrt().asDiagonal();
    }
    MatrixXd Zpred_full;  // predict scores in the full basis (transductive, no refit)
    if (!inductive && want_pred && selection != "refit")
        Zpred_full = new_scores(X, predict, full, maxdeg, center);

    // ---- per-part bases and centred outcomes
    for (Part& pt : parts) {
        VectorXd ytr = rows_of(Y, pt.tr);
        pt.ymean = center ? ytr.mean() : 0.0;
        pt.yc = ytr.array() - pt.ymean;
        pt.lo = ytr.minCoeff();
        pt.hi = ytr.maxCoeff();
        if (inductive) {
            const MatrixXd Xtr = rows_of(X, pt.tr);
            pt.basis = kernel_basis(mkernel_call(Xtr, maxdeg, center), npc, center);
            pt.Zte = new_scores(Xtr, rows_of(X, pt.te), pt.basis, maxdeg, center);
            if (want_pred && selection != "refit")
                pt.Zpred = new_scores(Xtr, predict, pt.basis, maxdeg, center);
        } else {
            pt.basis.U = rows_of(full.U, pt.tr);  // not orthonormal: legacy fold fit
            pt.basis.D2 = full.D2;
            pt.Zte = rows_of(Zfull, pt.te);
            if (want_pred && selection != "refit") pt.Zpred = Zpred_full;
        }
    }

    // ---- lambda grid: user supplied, or scaled by the basis that is fitted
    std::vector<double> lambdas = lambdas_in;
    std::string grid = "user";
    if (lambdas.empty()) {
        if (nlambda < 2) throw std::runtime_error("nlambda must be >= 2");
        grid = "scaled";
        double scale = 0.0;
        auto basis_scale = [&](const Basis& b, const VectorXd& yc) {
            const double nb = (double)b.U.rows();
            if (!l1) return b.D2(0) / nb;  // s_1^2 / n
            VectorXd s_uty = b.D2.cwiseSqrt().cwiseProduct(b.U.transpose() * yc);
            return s_uty.cwiseAbs().maxCoeff() / nb;  // lambda_max of the PC lasso
        };
        if (inductive) {
            for (const Part& pt : parts) scale = std::max(scale, basis_scale(pt.basis, pt.yc));
        } else {
            const double ym = center ? Y.mean() : 0.0;
            scale = basis_scale(full, Y.array() - ym);
        }
        if (!(scale > 0.0) || !std::isfinite(scale))
            throw std::runtime_error("cannot scale the lambda grid (constant outcome?)");
        const double a = l1 ? -5.0 : -9.0, z = l1 ? 0.0 : 1.0;
        lambdas.resize(nlambda);
        for (int j = 0; j < nlambda; ++j)
            lambdas[j] = scale * std::pow(10.0, a + (z - a) * j / (nlambda - 1.0));
    }
    const int L = (int)lambdas.size();
    for (double lam : lambdas)
        if (!(lam >= 0.0) || !std::isfinite(lam))
            throw std::runtime_error("lambdas must be finite and non-negative");

    // ---- fits on every part and held-out scoring
    const int P = (int)parts.size();
    GeneralCVOutput out;
    out.fold_mses = MatrixXd::Constant(P, L, NaN);
    out.cv_predictions = MatrixXd::Constant(n, L, NaN);
    out.fold_alphas.resize(P);
    for (int k = 0; k < P; ++k) {
        Part& pt = parts[k];
        MatrixXd& A = out.fold_alphas[k];
        A.resize(pt.basis.D2.size(), L);
        for (int j = 0; j < L; ++j) {
            A.col(j) = fit_alpha(pt.basis, pt.yc, lambdas[j], l1);
            VectorXd pred = (pt.Zte * A.col(j)).array() + pt.ymean;
            if (clip) clip_to(pred, pt.lo, pt.hi);
            double sse = 0.0;
            for (size_t i = 0; i < pt.te.size(); ++i) {
                const double e = Y(pt.te[i]) - pred(i);
                sse += e * e;
                out.cv_predictions(pt.te[i], j) = pred(i);
            }
            out.fold_mses(k, j) = sse / pt.te.size();
        }
        out.fold_intercepts.push_back(pt.ymean);
        out.fold_ranks.push_back((int)pt.basis.D2.size());
    }

    out.mses.assign(L, NaN);
    for (int j = 0; j < L; ++j) out.mses[j] = out.fold_mses.col(j).mean();
    int best = 0;
    for (int j = 1; j < L; ++j)
        if (out.mses[j] < out.mses[best]) best = j;  // first minimiser
    out.best_index = best;
    out.best_lambda = lambdas[best];
    out.lambdas = lambdas;
    out.grid = grid;
    out.foldid = foldid;
    out.predictions = VectorXd::Zero(0);
    out.intercept = NaN;
    out.rank = 0;

    // ---- returned fit
    if (selection == "refit") {
        const double ym = center ? Y.mean() : 0.0;
        VectorXd yc = Y.array() - ym;
        out.best_alpha = fit_alpha(full, yc, out.best_lambda, l1);
        out.intercept = ym;
        out.rank = (int)full.D2.size();
        if (want_pred) {
            out.predictions =
                (new_scores(X, predict, full, maxdeg, center) * out.best_alpha).array() + ym;
            if (clip) clip_to(out.predictions, Y.minCoeff(), Y.maxCoeff());
        }
    } else if (holdout) {
        const Part& pt = parts[0];
        out.best_alpha = out.fold_alphas[0].col(best);
        out.intercept = pt.ymean;
        out.rank = (int)pt.basis.D2.size();
        if (want_pred) {
            out.predictions = (pt.Zpred * out.best_alpha).array() + pt.ymean;
            if (clip) clip_to(out.predictions, pt.lo, pt.hi);
        }
    } else {  // fold_average
        out.best_alpha = VectorXd::Zero(0);
        if (want_pred) {
            out.predictions = VectorXd::Zero(predict.rows());
            for (int k = 0; k < P; ++k) {
                VectorXd pk = (parts[k].Zpred * out.fold_alphas[k].col(best)).array() +
                              parts[k].ymean;
                if (clip) clip_to(pk, parts[k].lo, parts[k].hi);
                out.predictions += pk / P;
            }
        }
    }
    return out;
}
