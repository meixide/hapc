// R entry point for the Gaussian norm "1"/"2" CV of src/cv_inductive.cpp.
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

#include <cstring>
#include <string>
#include "hapc_core.hpp"

using Eigen::Map;

static int as_int(SEXP x) { return Rf_isInteger(x) ? INTEGER(x)[0] : (int)REAL(x)[0]; }

static SEXP vec_out(const VectorXd& v) {
    SEXP out = PROTECT(Rf_allocVector(REALSXP, v.size()));
    std::copy(v.data(), v.data() + v.size(), REAL(out));
    UNPROTECT(1);
    return out;
}

static SEXP mat_out(const MatrixXd& m) {
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, m.rows(), m.cols()));
    std::copy(m.data(), m.data() + m.size(), REAL(out));
    UNPROTECT(1);
    return out;
}

extern "C" SEXP fasthal_cv_general_call(SEXP X_, SEXP Y_, SEXP npc_, SEXP lambdas_,
                                        SEXP nlambda_, SEXP nfolds_, SEXP predict_,
                                        SEXP maxdeg_, SEXP center_, SEXP l1_,
                                        SEXP inductive_, SEXP selection_,
                                        SEXP train_frac_, SEXP clip_) {
    if (!Rf_isReal(X_) || !Rf_isReal(Y_)) Rf_error("X and Y must be numeric.");
    const int n = Rf_nrows(X_), p = Rf_ncols(X_);
    if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");
    Map<const MatrixXd> X(REAL(X_), n, p);
    Map<const VectorXd> Y(REAL(Y_), n);
    std::vector<double> lambdas;
    if (!Rf_isNull(lambdas_))
        lambdas.assign(REAL(lambdas_), REAL(lambdas_) + Rf_length(lambdas_));
    MatrixXd predict(0, p);
    if (!Rf_isNull(predict_)) {
        if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
            Rf_error("predict must be a numeric matrix with the same number of columns as X.");
        predict = Map<const MatrixXd>(REAL(predict_), Rf_nrows(predict_), p);
    }
    const std::string selection = CHAR(STRING_ELT(selection_, 0));

    GeneralCVOutput res;
    char err[512] = {0};
    try {
        res = fasthal_cv_general(X, Y, as_int(npc_), lambdas, as_int(nlambda_), as_int(nfolds_),
                                 predict, as_int(maxdeg_), LOGICAL(center_)[0],
                                 LOGICAL(l1_)[0], LOGICAL(inductive_)[0], selection,
                                 REAL(train_frac_)[0], LOGICAL(clip_)[0]);
    } catch (const std::exception& e) {
        std::strncpy(err, e.what(), sizeof(err) - 1);
    }
    if (err[0]) Rf_error("cv.hapc: %s", err);

    const int L = (int)res.lambdas.size();
    const int P = (int)res.fold_alphas.size();
    int prot = 0;
    SEXP mses = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
    SEXP lams = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
    for (int j = 0; j < L; ++j) {
        REAL(mses)[j] = res.mses[j];
        REAL(lams)[j] = res.lambdas[j];
    }
    SEXP foldid = PROTECT(Rf_allocVector(INTSXP, n)); prot++;
    for (int i = 0; i < n; ++i) INTEGER(foldid)[i] = res.foldid[i];
    SEXP fold_alphas = PROTECT(Rf_allocVector(VECSXP, P)); prot++;
    for (int k = 0; k < P; ++k) SET_VECTOR_ELT(fold_alphas, k, mat_out(res.fold_alphas[k]));
    SEXP fold_int = PROTECT(Rf_allocVector(REALSXP, P)); prot++;
    SEXP fold_rank = PROTECT(Rf_allocVector(INTSXP, P)); prot++;
    for (int k = 0; k < P; ++k) {
        REAL(fold_int)[k] = res.fold_intercepts[k];
        INTEGER(fold_rank)[k] = res.fold_ranks[k];
    }
    const bool fold_average = res.best_alpha.size() == 0;

    const char* nm[] = {"mses", "lambdas", "best_lambda", "res_opt", "best_index",
                        "fold_mses", "cv_predictions", "foldid", "fold_alphas",
                        "fold_intercepts", "fold_ranks", "intercept", "rank", "lambda_grid",
                        "predictions"};
    const int n_out = Rf_isNull(predict_) ? 14 : 15;
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
    SET_VECTOR_ELT(out, 0, mses);
    SET_VECTOR_ELT(out, 1, lams);
    SET_VECTOR_ELT(out, 2, Rf_ScalarReal(res.best_lambda));
    SET_VECTOR_ELT(out, 3, fold_average ? R_NilValue : vec_out(res.best_alpha));
    SET_VECTOR_ELT(out, 4, Rf_ScalarInteger(res.best_index + 1));
    SET_VECTOR_ELT(out, 5, mat_out(res.fold_mses));
    SET_VECTOR_ELT(out, 6, mat_out(res.cv_predictions));
    SET_VECTOR_ELT(out, 7, foldid);
    SET_VECTOR_ELT(out, 8, fold_alphas);
    SET_VECTOR_ELT(out, 9, fold_int);
    SET_VECTOR_ELT(out, 10, fold_rank);
    SET_VECTOR_ELT(out, 11, Rf_ScalarReal(res.intercept));
    SET_VECTOR_ELT(out, 12, Rf_ScalarInteger(res.rank));
    SET_VECTOR_ELT(out, 13, Rf_mkString(res.grid.c_str()));
    if (n_out == 15) SET_VECTOR_ELT(out, 14, mat_out(res.predictions));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
    for (int i = 0; i < n_out; ++i) SET_STRING_ELT(names, i, Rf_mkChar(nm[i]));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(prot);
    return out;
}
