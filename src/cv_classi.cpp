// cv_classi.cpp
// SEXP entry point that delegates to the pure-C++ binomial CV implementation
// (see pcghal_cv_classi_cpp.cpp).  Keeping this file thin guarantees that R
// and Python use exactly the same fold partitioning, optimiser and risk
// computation: there is one source of truth.

#include <RcppEigen.h>
#include <vector>
#include "hapc_core.hpp"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// 14th SEXP slot historically held `single_lambda` (unused). It is now
// repurposed as `with_pgd_`: TRUE -> norm="sv" CV (logistic ridge + PGD),
// FALSE -> norm="2" CV (logistic ridge only). NULL or non-logical -> TRUE.
extern "C" SEXP pchal_cv_classi_call(SEXP X_, SEXP Y_, SEXP maxdeg_, SEXP npc_,
                                      SEXP lambdas_, SEXP nfolds_,
                                      SEXP max_iter_, SEXP tol_, SEXP step_factor_,
                                      SEXP verbose_, SEXP /*crit_*/,
                                      SEXP predict_, SEXP center_,
                                      SEXP with_pgd_) {
    if (!Rf_isReal(X_) || !Rf_isReal(Y_))
        Rf_error("X and Y must be numeric.");

    const int n = Rf_nrows(X_);
    const int p = Rf_ncols(X_);
    if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");

    Map<const MatrixXd> X(REAL(X_), n, p);
    Map<const VectorXd> Y(REAL(Y_), n);

    const int maxdeg = Rf_isInteger(maxdeg_) ? INTEGER(maxdeg_)[0] : (int)REAL(maxdeg_)[0];
    const int npc    = Rf_isInteger(npc_)    ? INTEGER(npc_)[0]    : (int)REAL(npc_)[0];
    const int K      = Rf_isInteger(nfolds_) ? INTEGER(nfolds_)[0] : (int)REAL(nfolds_)[0];

    const int L = Rf_length(lambdas_);
    if (L <= 0) Rf_error("lambdas must be non-empty.");
    std::vector<double> lambdas(L);
    for (int i = 0; i < L; ++i) lambdas[i] = REAL(lambdas_)[i];

    const int   max_iter    = Rf_isInteger(max_iter_) ? INTEGER(max_iter_)[0] : (int)REAL(max_iter_)[0];
    const double tol        = REAL(tol_)[0];
    const double step_factor = REAL(step_factor_)[0];
    const bool   verbose    = LOGICAL(verbose_)[0];
    const bool   center     = LOGICAL(center_)[0];

    // Predict matrix (or empty)
    MatrixXd predict_data(0, p);
    if (!Rf_isNull(predict_)) {
        if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
            Rf_error("predict must be a numeric matrix with the same number of columns as X.");
        const int m_pred = Rf_nrows(predict_);
        predict_data = Map<const MatrixXd>(REAL(predict_), m_pred, p);
    }

    bool with_pgd = true;
    if (!Rf_isNull(with_pgd_) && Rf_isLogical(with_pgd_) && Rf_length(with_pgd_) >= 1) {
        with_pgd = LOGICAL(with_pgd_)[0];
    }

    CVClassiOutput out = pcghal_cv_classi_python(X, Y, maxdeg, npc, lambdas, K,
                                                   predict_data,
                                                   max_iter, tol, step_factor,
                                                   verbose, center, with_pgd);

    int prot = 0;
    SEXP deviances_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
    for (int j = 0; j < L; ++j) REAL(deviances_out)[j] = out.deviances[j];

    SEXP lambdas_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
    for (int j = 0; j < L; ++j) REAL(lambdas_out)[j] = out.lambdas[j];

    SEXP best_lambda_out = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
    REAL(best_lambda_out)[0] = out.best_lambda;

    // res_opt mimics the original SEXP layout (a list with $alpha at slot 0)
    SEXP alpha_out = PROTECT(Rf_allocVector(REALSXP, out.best_alpha.size())); prot++;
    for (int i = 0; i < out.best_alpha.size(); ++i) REAL(alpha_out)[i] = out.best_alpha[i];

    SEXP res_opt = PROTECT(Rf_allocVector(VECSXP, 1)); prot++;
    SET_VECTOR_ELT(res_opt, 0, alpha_out);
    SEXP res_opt_names = PROTECT(Rf_allocVector(STRSXP, 1)); prot++;
    SET_STRING_ELT(res_opt_names, 0, Rf_mkChar("alpha"));
    Rf_setAttrib(res_opt, R_NamesSymbol, res_opt_names);

    const bool has_pred = (out.predictions.size() > 0);
    SEXP predictions_out = R_NilValue;
    if (has_pred) {
        predictions_out = PROTECT(Rf_allocVector(REALSXP, out.predictions.size())); prot++;
        for (int i = 0; i < out.predictions.size(); ++i) REAL(predictions_out)[i] = out.predictions[i];
    }

    const int n_out = has_pred ? 5 : 4;
    SEXP final_out = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
    SET_VECTOR_ELT(final_out, 0, deviances_out);
    SET_VECTOR_ELT(final_out, 1, lambdas_out);
    SET_VECTOR_ELT(final_out, 2, best_lambda_out);
    SET_VECTOR_ELT(final_out, 3, res_opt);
    if (has_pred) SET_VECTOR_ELT(final_out, 4, predictions_out);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
    SET_STRING_ELT(names, 0, Rf_mkChar("deviances"));
    SET_STRING_ELT(names, 1, Rf_mkChar("lambdas"));
    SET_STRING_ELT(names, 2, Rf_mkChar("best_lambda"));
    SET_STRING_ELT(names, 3, Rf_mkChar("res_opt"));
    if (has_pred) SET_STRING_ELT(names, 4, Rf_mkChar("predictions"));
    Rf_setAttrib(final_out, R_NamesSymbol, names);

    UNPROTECT(prot);
    return final_out;
}
