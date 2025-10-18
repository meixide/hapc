// pcghal_call.cpp
// Full pc_hal port with Eigen, .Call interface, and robust guards.

#define R_NO_REMAP
#include <Rinternals.h>

// ---- Fix R macro pollution (breaks <locale>, Eigen, etc.) ----
#ifdef length
#undef length
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <R_ext/Print.h>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static inline double sign_double(double x) {
    return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
}

extern "C" SEXP pcghal_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                             SEXP max_iter_, SEXP tol_, SEXP step_factor_, SEXP verbose_, SEXP crit_)
{
    // ---- Inputs & sizes ----
    const int n = Rf_nrows(Xtilde_);
    const int k = Rf_ncols(Xtilde_);
    const int q = Rf_nrows(ENn_);

    if (k <= 0 || n <= 0 || q <= 0)
        Rf_error("Invalid dimensions: n=%d k=%d q=%d", n, k, q);

    const double* Yp      = REAL(Y_);
    const double* Xp      = REAL(Xtilde_);
    const double* Vp      = REAL(ENn_);
    const double* alpha0p = REAL(alpha0_);

    const int    max_iter   = INTEGER(max_iter_)[0];
    const double tol        = REAL(tol_)[0];
    const double step_fac   = REAL(step_factor_)[0];
    const int    verbose    = (Rf_asLogical(verbose_) == TRUE);
    std::string crit = CHAR(STRING_ELT(crit_, 0));

    const double eps = 1e-12;

    // ---- Map to Eigen (column-major like R) ----
    Map<const VectorXd> y(Yp, n);
    Map<const MatrixXd> X(Xp, n, k);   // n x k
    Map<const MatrixXd> V(Vp, q, k);   // q x k
    Map<const VectorXd> a0(alpha0p, k);

    // ---- Work vectors ----
    VectorXd alpha = a0;            // k
    VectorXd mu(n), g(k), beta(q), a(k), g_tan(k);
    VectorXd sgn(q), Vt_s(k), gt_alpha(k), numer(q);

    // ---- Helpers matching your R code ----
    auto risk = [&](const VectorXd& a)->double {
        mu.noalias() = X * a;
        return (y - mu).squaredNorm() / static_cast<double>(n);
    };

    auto grad = [&](const VectorXd& a)->VectorXd {
        // grad_j = alpha_j * mean(2 * X[,j] * (Y - mu))
        mu.noalias() = X * a;
        VectorXd r = y - mu;
        VectorXd gtmp(k);
        for (int j = 0; j < k; ++j)
            gtmp[j] = a[j] * 2.0 * (X.col(j).dot(r)) / static_cast<double>(n);
        return gtmp;
    };

    // ---- History matrix (max_iter+1 rows: initial + each iter) ----
    SEXP alphaiters_ = PROTECT(Rf_allocMatrix(REALSXP, max_iter + 1, k));
    double* Ahist = REAL(alphaiters_);
    auto push_hist_row = [&](int row, const VectorXd& a) {
        for (int j = 0; j < k; ++j) Ahist[row + j * (max_iter + 1)] = a[j];
    };
    push_hist_row(0, alpha);

    double R_old = risk(alpha);
    if (!std::isfinite(R_old)) Rf_error("Non-finite initial risk");

    if (verbose) {
        Rprintf("Using criterion: %s\n", crit.c_str());
        Rprintf("Init | Risk = %.6g  L1(beta) = %.6g\n",
                R_old, (V * alpha).cwiseAbs().sum());
    }

    int iter_done = 0;
    for (int iter = 1; iter <= max_iter; ++iter) {
        // gradient
        g = grad(alpha);
        if (!g.allFinite()) Rf_error("Non-finite gradient at iter %d", iter);

        // beta = V * alpha
        beta.noalias() = V * alpha;

        // a = alpha * (t(V) %*% sign(beta))
        for (int i = 0; i < q; ++i) sgn[i] = sign_double(beta[i]);
        Vt_s.noalias() = V.transpose() * sgn;     // k
        a = alpha.array() * Vt_s.array();         // k

        // Project gradient onto tangent space: g_tan = g - proj_a(g)
        const double denom = a.squaredNorm();
        VectorXd proj(k);
        if (denom > eps) proj = (g.dot(a) / denom) * a;
        else             proj = VectorXd::Zero(k);
        g_tan = g - proj;

        // Step restriction:
        // restr = (V %*% (g_tan * alpha)) / beta   (elementwise)
        gt_alpha = g_tan.array() * alpha.array();
        numer.noalias() = V * gt_alpha;

        // restr = numer / beta (elementwise)
        VectorXd restr(q);
        for (int i = 0; i < q; ++i) {
            const double bi = beta[i];
            if (std::abs(bi) > eps)
                restr[i] = numer[i] / bi;
            else
                restr[i] = 0.0;  // safe filler
        }

        // Find indices where restr < 0
        std::vector<double> valid;
        valid.reserve(q);
        for (int i = 0; i < q; ++i) {
            if (restr[i] < 0.0)
                valid.push_back(-1.0 / restr[i]);  // -1/restr
        }

        // step = step_factor * min(valid)
        double step = 0.0;
        if (!valid.empty()) {
            double min_val = *std::min_element(valid.begin(), valid.end());
            step = step_fac * min_val;
        } else {
            step = 0.0; // matches R: no negatives => Inf => effectively 0
        }                               // no restriction → don't move

        // Avoid exploding steps
        if (!std::isfinite(step) || std::abs(step) > 1e6) step = 0.0;

        // Update: alpha_new = alpha * (1 + step * g_tan)
        VectorXd alpha_new = alpha.array() * (1.0 + step * g_tan.array());

        // Risk & stopping
        const double R_new = risk(alpha_new);
        if (verbose) {
            const double l1_new = (V * alpha_new).cwiseAbs().sum();
            Rprintf("Iter %3d | step=%.6g  Risk=%.6g  dRisk=%.6g  ||g_tan||=%.6g  L1(beta)=%.6g\n",
                    iter, step, R_new, (R_old - R_new), g_tan.norm(), l1_new);
        }

        push_hist_row(iter, alpha_new);
        iter_done = iter;

        if (crit == "grad") {
            if (!std::isfinite(R_new) || g_tan.norm() < tol) {
                alpha = alpha_new;
                R_old = R_new;
                break;
            }
        } else if (crit == "risk") {
            if (!std::isfinite(R_new) || (R_old - R_new) < tol) {
                alpha = alpha_new;
                R_old = R_new;
                break;
            }
        } else {
            Rf_error("Invalid crit argument: must be 'grad' or 'risk'");
        }

        alpha = alpha_new;
        R_old = R_new;
    }

    // ---- Prepare return objects ----
    // alpha (k)
    SEXP alpha_out_ = PROTECT(Rf_allocVector(REALSXP, k));
    for (int j = 0; j < k; ++j) REAL(alpha_out_)[j] = alpha[j];

    // beta (q) at the solution
    VectorXd beta_final = V * alpha;
    SEXP beta_ = PROTECT(Rf_allocVector(REALSXP, q));
    for (int i = 0; i < q; ++i) REAL(beta_)[i] = beta_final[i];

    // risk (1)
    SEXP risk_ = PROTECT(Rf_allocVector(REALSXP, 1));
    REAL(risk_)[0] = R_old;

    // iter (1)
    SEXP iter_ = PROTECT(Rf_allocVector(INTSXP, 1));
    INTEGER(iter_)[0] = iter_done;

    // Trim alphaiters to used rows (iter_done+1 rows)
    SEXP alphaiters_trim_ = PROTECT(Rf_allocMatrix(REALSXP, iter_done + 1, k));
    {
        double* T = REAL(alphaiters_trim_);
        const int src_rows = max_iter + 1;
        for (int j = 0; j < k; ++j) {
            for (int r = 0; r <= iter_done; ++r) {
                T[r + j * (iter_done + 1)] = Ahist[r + j * src_rows];
            }
        }
    }

    // Build named list: list(alpha, alphaiters, beta, risk, iter)
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 5));
    SET_VECTOR_ELT(out, 0, alpha_out_);
    SET_VECTOR_ELT(out, 1, alphaiters_trim_);
    SET_VECTOR_ELT(out, 2, beta_);
    SET_VECTOR_ELT(out, 3, risk_);
    SET_VECTOR_ELT(out, 4, iter_);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_STRING_ELT(names, 0, Rf_mkChar("alpha"));
    SET_STRING_ELT(names, 1, Rf_mkChar("alphaiters"));
    SET_STRING_ELT(names, 2, Rf_mkChar("beta"));
    SET_STRING_ELT(names, 3, Rf_mkChar("risk"));
    SET_STRING_ELT(names, 4, Rf_mkChar("iter"));
    Rf_setAttrib(out, R_NamesSymbol, names);

    UNPROTECT(8); // alphaiters_, alpha_out_, beta_, risk_, iter_, alphaiters_trim_, out, names
    return out;
}