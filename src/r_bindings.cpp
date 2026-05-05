#define R_NO_REMAP
#include <Rinternals.h>

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
#include <cmath>
#include "hapc_core.hpp"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Wrapper: pchal_des (called by design.hapc)
extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_) {
    if (!Rf_isReal(X_)) Rf_error("X must be numeric");
    
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    int maxdeg = Rf_isInteger(maxdeg_) ? INTEGER(maxdeg_)[0] : (int)REAL(maxdeg_)[0];
    int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];
    bool center = Rf_isLogical(center_) ? LOGICAL(center_)[0] : true;
    
    Map<const MatrixXd> X(REAL(X_), n, p);
    DesignOutput des = pchal_des(X, maxdeg, npc, center);
    
    // Build return list
    SEXP H_ = PROTECT(Rf_allocMatrix(REALSXP, n, des.H.cols()));
    SEXP U_ = PROTECT(Rf_allocMatrix(REALSXP, n, des.U.cols()));
    SEXP d_ = PROTECT(Rf_allocVector(REALSXP, des.d.size()));
    SEXP V_ = PROTECT(Rf_allocMatrix(REALSXP, des.V.rows(), des.V.cols()));
    
    std::copy(des.H.data(), des.H.data() + des.H.size(), REAL(H_));
    std::copy(des.U.data(), des.U.data() + des.U.size(), REAL(U_));
    std::copy(des.d.data(), des.d.data() + des.d.size(), REAL(d_));
    std::copy(des.V.data(), des.V.data() + des.V.size(), REAL(V_));
    
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, H_);
    SET_VECTOR_ELT(out, 1, U_);
    SET_VECTOR_ELT(out, 2, d_);
    SET_VECTOR_ELT(out, 3, V_);
    
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(names, 0, Rf_mkChar("H"));
    SET_STRING_ELT(names, 1, Rf_mkChar("U"));
    SET_STRING_ELT(names, 2, Rf_mkChar("d"));
    SET_STRING_ELT(names, 3, Rf_mkChar("V"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    
    UNPROTECT(6);
    return out;
}

// Wrapper: ridge_call
extern "C" SEXP ridge_call(SEXP Y_, SEXP U_, SEXP D2_, SEXP lambda_) {
    if (!Rf_isReal(Y_)) Rf_error("Y must be numeric");
    if (!Rf_isReal(U_)) Rf_error("U must be numeric");
    if (!Rf_isReal(D2_)) Rf_error("D2 must be numeric");
    
    const int n = Rf_length(Y_);
    const int p = Rf_length(D2_);
    double lambda = REAL(lambda_)[0];
    
    Map<const VectorXd> Y(REAL(Y_), n);
    Map<const MatrixXd> U(REAL(U_), n, p);
    Map<const VectorXd> D2(REAL(D2_), p);
    
    VectorXd beta = ridge_call(Y, U, D2, lambda);
    
    SEXP out = PROTECT(Rf_allocVector(REALSXP, p));
    std::copy(beta.data(), beta.data() + p, REAL(out));
    UNPROTECT(1);
    return out;
}

// Wrapper: mkernel_call (called by kernel.hapc)
extern "C" SEXP mkernel_call(SEXP X_, SEXP m_, SEXP center_) {
    if (!Rf_isReal(X_)) Rf_error("X must be numeric");
    
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    int m = Rf_isInteger(m_) ? INTEGER(m_)[0] : (int)REAL(m_)[0];
    bool center = Rf_isLogical(center_) ? LOGICAL(center_)[0] : true;
    
    Map<const MatrixXd> X(REAL(X_), n, p);
    MatrixXd K = mkernel_call(X, m, center);
    
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    std::copy(K.data(), K.data() + n * n, REAL(out));
    UNPROTECT(1);
    return out;
}

// Wrapper: kernel_cross_call (called by cross_kernel.hapc)
extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_) {
    if (!Rf_isReal(Xtr_)) Rf_error("Xtr must be numeric");
    if (!Rf_isReal(Xte_)) Rf_error("Xte must be numeric");
    
    const int n = Rf_nrows(Xtr_);
    const int p = Rf_ncols(Xtr_);
    const int m_test = Rf_nrows(Xte_);
    int m = Rf_isInteger(m_) ? INTEGER(m_)[0] : (int)REAL(m_)[0];
    bool center = Rf_isLogical(center_) ? LOGICAL(center_)[0] : true;
    
    Map<const MatrixXd> Xtr(REAL(Xtr_), n, p);
    Map<const MatrixXd> Xte(REAL(Xte_), m_test, p);
    MatrixXd K = kernel_cross_call(Xtr, Xte, m, center);
    
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, m_test, n));
    std::copy(K.data(), K.data() + m_test * n, REAL(out));
    UNPROTECT(1);
    return out;
}

// Wrapper: pcghal_call
extern "C" SEXP pcghal_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                            SEXP max_iter_, SEXP tol_, SEXP step_factor_, 
                            SEXP verbose_, SEXP crit_) {
    Map<const VectorXd> Y(REAL(Y_), Rf_length(Y_));
    Map<const MatrixXd> Xtilde(REAL(Xtilde_), Rf_nrows(Xtilde_), Rf_ncols(Xtilde_));
    Map<const MatrixXd> ENn(REAL(ENn_), Rf_nrows(ENn_), Rf_ncols(ENn_));
    Map<const VectorXd> alpha0(REAL(alpha0_), Rf_length(alpha0_));
    
    int max_iter = INTEGER(max_iter_)[0];
    double tol = REAL(tol_)[0];
    double step_factor = REAL(step_factor_)[0];
    bool verbose = LOGICAL(verbose_)[0];
    std::string crit = CHAR(STRING_ELT(crit_, 0));
    
    OptimizerOutput res = pcghal_call(Y, Xtilde, ENn, alpha0, max_iter, tol, step_factor, verbose, crit);
    
    SEXP alpha = PROTECT(Rf_allocVector(REALSXP, res.alpha.size()));
    SEXP alphaiters = PROTECT(Rf_allocMatrix(REALSXP, res.alphaiters.rows(), res.alphaiters.cols()));
    SEXP beta = PROTECT(Rf_allocVector(REALSXP, res.beta.size()));
    SEXP risk = PROTECT(Rf_allocVector(REALSXP, 1));
    SEXP iter = PROTECT(Rf_allocVector(INTSXP, 1));
    
    std::copy(res.alpha.data(), res.alpha.data() + res.alpha.size(), REAL(alpha));
    std::copy(res.alphaiters.data(), res.alphaiters.data() + res.alphaiters.size(), REAL(alphaiters));
    std::copy(res.beta.data(), res.beta.data() + res.beta.size(), REAL(beta));
    REAL(risk)[0] = res.risk;
    INTEGER(iter)[0] = res.iter;
    
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 5));
    SET_VECTOR_ELT(out, 0, alpha);
    SET_VECTOR_ELT(out, 1, alphaiters);
    SET_VECTOR_ELT(out, 2, beta);
    SET_VECTOR_ELT(out, 3, risk);
    SET_VECTOR_ELT(out, 4, iter);
    
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_STRING_ELT(names, 0, Rf_mkChar("alpha"));
    SET_STRING_ELT(names, 1, Rf_mkChar("alphaiters"));
    SET_STRING_ELT(names, 2, Rf_mkChar("beta"));
    SET_STRING_ELT(names, 3, Rf_mkChar("risk"));
    SET_STRING_ELT(names, 4, Rf_mkChar("iter"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    
    UNPROTECT(7);
    return out;
}

// Wrapper: pcghal_classi_call (called by pc_hal_classi_cpp)
extern "C" SEXP pcghal_classi_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                                   SEXP max_iter_, SEXP tol_, SEXP step_factor_, 
                                   SEXP verbose_) {
    Map<const VectorXd> Y(REAL(Y_), Rf_length(Y_));
    Map<const MatrixXd> Xtilde(REAL(Xtilde_), Rf_nrows(Xtilde_), Rf_ncols(Xtilde_));
    Map<const MatrixXd> ENn(REAL(ENn_), Rf_nrows(ENn_), Rf_ncols(ENn_));
    Map<const VectorXd> alpha0(REAL(alpha0_), Rf_length(alpha0_));
    
    int max_iter = INTEGER(max_iter_)[0];
    double tol = REAL(tol_)[0];
    double step_factor = REAL(step_factor_)[0];
    bool verbose = LOGICAL(verbose_)[0];
    
    OptimizerOutput res = pcghal_classi_call(Y, Xtilde, ENn, alpha0, max_iter, tol, step_factor, verbose);
    
    SEXP alpha = PROTECT(Rf_allocVector(REALSXP, res.alpha.size()));
    SEXP alphaiters = PROTECT(Rf_allocMatrix(REALSXP, res.alphaiters.rows(), res.alphaiters.cols()));
    SEXP beta = PROTECT(Rf_allocVector(REALSXP, res.beta.size()));
    SEXP risk = PROTECT(Rf_allocVector(REALSXP, 1));
    SEXP iter = PROTECT(Rf_allocVector(INTSXP, 1));
    
    std::copy(res.alpha.data(), res.alpha.data() + res.alpha.size(), REAL(alpha));
    std::copy(res.alphaiters.data(), res.alphaiters.data() + res.alphaiters.size(), REAL(alphaiters));
    std::copy(res.beta.data(), res.beta.data() + res.beta.size(), REAL(beta));
    REAL(risk)[0] = res.risk;
    INTEGER(iter)[0] = res.iter;
    
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 5));
    SET_VECTOR_ELT(out, 0, alpha);
    SET_VECTOR_ELT(out, 1, alphaiters);
    SET_VECTOR_ELT(out, 2, beta);
    SET_VECTOR_ELT(out, 3, risk);
    SET_VECTOR_ELT(out, 4, iter);
    
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_STRING_ELT(names, 0, Rf_mkChar("alpha"));
    SET_STRING_ELT(names, 1, Rf_mkChar("alphaiters"));
    SET_STRING_ELT(names, 2, Rf_mkChar("beta"));
    SET_STRING_ELT(names, 3, Rf_mkChar("risk"));
    SET_STRING_ELT(names, 4, Rf_mkChar("iter"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    
    UNPROTECT(7);
    return out;
}

// Wrapper: fast_pchal_call
extern "C" SEXP fast_pchal_call(SEXP U_, SEXP D2_, SEXP Y_, SEXP lambda_) {
    Map<const MatrixXd> U(REAL(U_), Rf_nrows(U_), Rf_ncols(U_));
    Map<const VectorXd> D2(REAL(D2_), Rf_length(D2_));
    Map<const VectorXd> Y(REAL(Y_), Rf_length(Y_));
    double lambda = REAL(lambda_)[0];
    
    VectorXd beta = fast_pchal_call(U, D2, Y, lambda);
    
    SEXP out = PROTECT(Rf_allocVector(REALSXP, beta.size()));
    std::copy(beta.data(), beta.data() + beta.size(), REAL(out));
    UNPROTECT(1);
    return out;
}

// Wrapper: single_pcghal_call (called by hapc.R)
//
// Initialiser is selectable via the ``ini`` argument (string "1" = LASSO via
// fast_pchal_call, "2" = ridge via ridge_call). Default is "1" (LASSO) so that
// R and Python ``hapc(norm="sv")`` are bit-identical by default.
extern "C" SEXP single_pcghal_call(SEXP X_, SEXP Y_, SEXP maxdeg_, SEXP npc_,
                                   SEXP single_lambda_, SEXP max_iter_, SEXP tol_, 
                                   SEXP step_factor_, SEXP verbose_, SEXP crit_,
                                   SEXP predict_, SEXP center_, SEXP ini_) {
    if (!Rf_isReal(X_)) Rf_error("X must be numeric");
    if (!Rf_isReal(Y_)) Rf_error("Y must be numeric");
    
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    int maxdeg = Rf_isInteger(maxdeg_) ? INTEGER(maxdeg_)[0] : (int)REAL(maxdeg_)[0];
    int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];
    double lambda = REAL(single_lambda_)[0];
    int max_iter = INTEGER(max_iter_)[0];
    double tol = REAL(tol_)[0];
    double step_factor = REAL(step_factor_)[0];
    bool verbose = LOGICAL(verbose_)[0];
    std::string crit = CHAR(STRING_ELT(crit_, 0));
    bool center = Rf_isLogical(center_) ? LOGICAL(center_)[0] : true;
    std::string ini = (Rf_isNull(ini_) || !Rf_isString(ini_))
                          ? std::string("1")
                          : std::string(CHAR(STRING_ELT(ini_, 0)));
    
    Map<const MatrixXd> X(REAL(X_), n, p);
    Map<const VectorXd> Y(REAL(Y_), n);
    
    // Design matrix
    DesignOutput des = pchal_des(X, maxdeg, npc, center);
    MatrixXd Xtilde = des.U * des.d.asDiagonal();
    MatrixXd ENn = des.V;
    
    double ymean = 0.0;
    VectorXd Y_fit = Y;
    if (center) {
        ymean = Y.mean();
        Y_fit = Y.array() - ymean;
    }
    
    // Initialiser: LASSO (default) or ridge
    VectorXd D2 = des.d.array().square();
    VectorXd alpha0;
    if (ini == "2") {
        alpha0 = ridge_call(Y_fit, des.U, D2, lambda);
    } else {
        // "1" or anything else falls back to LASSO (matches Python default)
        alpha0 = fast_pchal_call(des.U, D2, Y_fit, lambda);
    }
    
    // Optimize
    OptimizerOutput res_opt = pcghal_call(Y_fit, Xtilde, ENn, alpha0, 
                                          max_iter, tol, step_factor, verbose, crit);
    
    // Build return list
    SEXP alpha_out = PROTECT(Rf_allocVector(REALSXP, res_opt.alpha.size()));
    SEXP alphaiters = PROTECT(Rf_allocMatrix(REALSXP, res_opt.alphaiters.rows(), res_opt.alphaiters.cols()));
    SEXP beta = PROTECT(Rf_allocVector(REALSXP, res_opt.beta.size()));
    SEXP risk = PROTECT(Rf_allocVector(REALSXP, 1));
    SEXP iter = PROTECT(Rf_allocVector(INTSXP, 1));
    
    std::copy(res_opt.alpha.data(), res_opt.alpha.data() + res_opt.alpha.size(), REAL(alpha_out));
    std::copy(res_opt.alphaiters.data(), res_opt.alphaiters.data() + res_opt.alphaiters.size(), REAL(alphaiters));
    std::copy(res_opt.beta.data(), res_opt.beta.data() + res_opt.beta.size(), REAL(beta));
    REAL(risk)[0] = res_opt.risk;
    INTEGER(iter)[0] = res_opt.iter;
    
    int prot = 5;  // alpha_out, alphaiters, beta, risk, iter
    SEXP predictions = PROTECT(Rf_allocVector(REALSXP, 0)); prot++;
    if (!Rf_isNull(predict_) && Rf_nrows(predict_) > 0) {
        const int m_pred = Rf_nrows(predict_);
        Map<const MatrixXd> Xtest(REAL(predict_), m_pred, p);
        MatrixXd Ktest = kernel_cross_call(X, Xtest, maxdeg, center);

        VectorXd d_inv = des.d.array().cwiseInverse();
        VectorXd v = des.U * (d_inv.asDiagonal() * res_opt.alpha);
        VectorXd preds = Ktest * v;

        if (center) {
            preds.array() += ymean;
        }

        predictions = PROTECT(Rf_allocVector(REALSXP, m_pred)); prot++;
        std::copy(preds.data(), preds.data() + m_pred, REAL(predictions));
    }

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4)); prot++;
    SET_VECTOR_ELT(out, 0, alpha_out);
    SET_VECTOR_ELT(out, 1, alphaiters);
    SET_VECTOR_ELT(out, 2, beta);
    SET_VECTOR_ELT(out, 3, predictions);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4)); prot++;
    SET_STRING_ELT(names, 0, Rf_mkChar("alpha"));
    SET_STRING_ELT(names, 1, Rf_mkChar("alphaiters"));
    SET_STRING_ELT(names, 2, Rf_mkChar("beta"));
    SET_STRING_ELT(names, 3, Rf_mkChar("predictions"));
    Rf_setAttrib(out, R_NamesSymbol, names);

    UNPROTECT(prot);
    return out;
}

// Binomial + norm="2": logistic ridge initialiser only (no PGD). Mirrors Python
// `single_pcghal_classification_ridge_only`.
extern "C" SEXP single_pcghal_classi_ridge_call(SEXP X_, SEXP Y_, SEXP maxdeg_,
                                                SEXP npc_, SEXP lambda_,
                                                SEXP predict_, SEXP center_) {
    if (!Rf_isReal(X_) || !Rf_isReal(Y_)) Rf_error("X and Y must be numeric");
    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");
    Map<const MatrixXd> X(REAL(X_), n, p);
    Map<const VectorXd> Y01(REAL(Y_), n);
    for (int i = 0; i < n; ++i) {
        if (Y01[i] != 0.0 && Y01[i] != 1.0) Rf_error("Y must contain only 0 and 1");
    }
    int maxdeg = Rf_isInteger(maxdeg_) ? INTEGER(maxdeg_)[0] : (int)REAL(maxdeg_)[0];
    int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];
    double lambda = REAL(lambda_)[0];
    bool center = LOGICAL(center_)[0];

    if (center) {
        if (npc >= n) npc = n - 1;
    } else {
        if (npc > n) npc = n;
    }

    DesignOutput des = pchal_des(X, maxdeg, npc, center);
    const int final_npc = (int)des.d.size();
    MatrixXd Xtilde = des.U * des.d.asDiagonal();

    VectorXd Y_pm1(n);
    for (int i = 0; i < n; ++i) Y_pm1[i] = (Y01[i] == 1.0) ? 1.0 : -1.0;

    auto calibrate_b0 = [](const VectorXd& y01, const VectorXd& eta) {
        double b0 = 0.0;
        for (int it = 0; it < 50; ++it) {
            VectorXd z = eta.array() + b0;
            VectorXd p = (1.0 + (-z.array()).exp()).inverse();
            double g = (p - y01).sum();
            double h = (p.array() * (1.0 - p.array())).sum();
            if (std::abs(g) < 1e-10 || h < 1e-12) break;
            b0 -= g / h;
        }
        return b0;
    };

    VectorXd alpha = logistic_ridge_init(Y_pm1, Xtilde, lambda);
    VectorXd eta = Xtilde * alpha;
    const double b0 = calibrate_b0(Y01, eta);
    double risk = 0.0;
    for (int i = 0; i < n; ++i) {
        double ymu = Y_pm1[i] * (eta[i] + b0);
        if (ymu > 0)
            risk += std::log1p(std::exp(-ymu));
        else
            risk += -ymu + std::log1p(std::exp(ymu));
    }
    risk /= n;

    int prot = 0;
    SEXP alpha_sexp = PROTECT(Rf_allocVector(REALSXP, final_npc)); prot++;
    std::copy(alpha.data(), alpha.data() + final_npc, REAL(alpha_sexp));

    SEXP predictions = PROTECT(Rf_allocVector(REALSXP, 0)); prot++;
    if (!Rf_isNull(predict_) && Rf_nrows(predict_) > 0) {
        const int m_pred = Rf_nrows(predict_);
        Map<const MatrixXd> Xtest(REAL(predict_), m_pred, p);
        MatrixXd Ktest = kernel_cross_call(X, Xtest, maxdeg, center);
        VectorXd d_inv = des.d.array().cwiseInverse();
        VectorXd v = des.U * (d_inv.asDiagonal() * alpha);
        VectorXd log_odds = (Ktest * v).array() + b0;
        predictions = PROTECT(Rf_allocVector(REALSXP, m_pred)); prot++;
        std::copy(log_odds.data(), log_odds.data() + m_pred, REAL(predictions));
    }

    SEXP risk_sexp = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
    REAL(risk_sexp)[0] = risk;
    SEXP iter_sexp = PROTECT(Rf_allocVector(INTSXP, 1)); prot++;
    INTEGER(iter_sexp)[0] = 0;
    SEXP lam_sexp = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
    REAL(lam_sexp)[0] = lambda;

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 5)); prot++;
    SET_VECTOR_ELT(out, 0, alpha_sexp);
    SET_VECTOR_ELT(out, 1, predictions);
    SET_VECTOR_ELT(out, 2, risk_sexp);
    SET_VECTOR_ELT(out, 3, iter_sexp);
    SET_VECTOR_ELT(out, 4, lam_sexp);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 5)); prot++;
    SET_STRING_ELT(names, 0, Rf_mkChar("alpha"));
    SET_STRING_ELT(names, 1, Rf_mkChar("predictions"));
    SET_STRING_ELT(names, 2, Rf_mkChar("risk"));
    SET_STRING_ELT(names, 3, Rf_mkChar("iter"));
    SET_STRING_ELT(names, 4, Rf_mkChar("lambda"));
    Rf_setAttrib(out, R_NamesSymbol, names);

    UNPROTECT(prot);
    return out;
}
