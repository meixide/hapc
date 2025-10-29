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
#include <RcppEigen.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// externs you already have
extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_);
extern "C" SEXP ridge_call(SEXP Y_, SEXP X_, SEXP lambda_);
extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_);
extern "C" SEXP pcghal_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                             SEXP max_iter_, SEXP tol_, SEXP step_factor_, SEXP verbose_, SEXP crit_);

extern "C" SEXP pchal_cv_call(SEXP X_, SEXP Y_, SEXP maxdeg_, SEXP npc_,
                              SEXP lambdas_, SEXP nfolds_,
                              SEXP max_iter_, SEXP tol_, SEXP step_factor_,
                              SEXP verbose_, SEXP crit_,
                              SEXP predict_, SEXP center_) {
  if (!Rf_isReal(X_) || !Rf_isReal(Y_))
    Rf_error("X and Y must be numeric.");

  const int n  = Rf_nrows(X_);
  const int p  = Rf_ncols(X_);
  if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");

  int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];
  const int K   = Rf_isInteger(nfolds_) ? INTEGER(nfolds_)[0] : (int)REAL(nfolds_)[0];

  const int L = Rf_length(lambdas_);
  if (L <= 0) Rf_error("lambdas must be non-empty.");
  std::vector<double> lambdas(L);
  for (int i = 0; i < L; ++i) lambdas[i] = REAL(lambdas_)[i];

  int prot = 0;

  // if center is TRUE, then npc cannot exceed n - 1
  bool center = true;
  if (Rf_isLogical(center_)) center = LOGICAL(center_)[0];
  else Rf_error("center must be logical");

  if (center) {
      if (npc >= n) {
          npc = n - 1;
          Rf_warning("npc reduced to n - 1 due to centering.");
      }
  } else {
      if (npc > n) {
          npc = n;
          Rf_warning("npc reduced to n due to no centering.");
      }
  } 

  // Build design: des = list(H, U, d, V)
  SEXP des_out = PROTECT(pchal_des(X_, maxdeg_, npc_, center_)); prot++;
  SEXP U_ = VECTOR_ELT(des_out, 1);
  SEXP d_ = VECTOR_ELT(des_out, 2);
  SEXP V_ = VECTOR_ELT(des_out, 3);

  Map<const MatrixXd> U(REAL(U_), Rf_nrows(U_), Rf_ncols(U_));
  Map<const VectorXd> d(REAL(d_), Rf_length(d_));
  Map<const MatrixXd> V(REAL(V_), Rf_nrows(V_), Rf_ncols(V_));

  MatrixXd Xtilde = U.leftCols(npc) * d.head(npc).asDiagonal();
  MatrixXd E_Nn   = V.leftCols(npc);
  Map<const VectorXd> Y(REAL(Y_), n);

  // Create folds: folds = sample(cut(seq(1, n), breaks = K, labels = FALSE))
  std::vector<int> folds(n);
  const int fold_size = n / K;
  for (int i = 0; i < n; ++i) {
    folds[i] = (i / fold_size) + 1;  // 1-indexed fold assignment
  }
  // Cap the last fold if n is not divisible by K
  for (int i = fold_size * K; i < n; ++i) {
    folds[i] = K;
  }
  // Shuffle fold assignments
  std::mt19937 rng(12345);
  std::shuffle(folds.begin(), folds.end(), rng);

  MatrixXd fold_error = MatrixXd::Constant(K, L, std::numeric_limits<double>::quiet_NaN());

  for (int j = 0; j < L; ++j) {
    const double lambda = lambdas[j];

    for (int i = 1; i <= K; ++i) {  // folds are 1-indexed
      // Collect test and train indices
      std::vector<int> test_idx, train_idx;
      for (int ii = 0; ii < n; ++ii) {
        if (folds[ii] == i) {
          test_idx.push_back(ii);
        } else {
          train_idx.push_back(ii);
        }
      }

      const int ntrain = (int)train_idx.size();
      const int ntest  = (int)test_idx.size();
      if (ntrain == 0 || ntest == 0) { 
        fold_error(i - 1, j) = NA_REAL;
        continue; 
      }

      // Extract train data
      MatrixXd Xtrain(ntrain, npc), Xtest(ntest, npc);
      VectorXd Ytrain(ntrain), Ytest(ntest);
      for (int ii = 0; ii < ntrain; ++ii) {
        Xtrain.row(ii) = Xtilde.row(train_idx[ii]);
        Ytrain[ii]     = Y[train_idx[ii]];
      }
      for (int ii = 0; ii < ntest; ++ii) {
        Xtest.row(ii) = Xtilde.row(test_idx[ii]);
        Ytest[ii]     = Y[test_idx[ii]];
      }

      // Inner protect block
      int nprot = 0;
      SEXP Y_in   = PROTECT(Rf_allocVector(REALSXP, ntrain)); nprot++;
      // first copy Ytrain into Y_in, then compute mean from Ytrain directly
      std::copy(Ytrain.data(), Ytrain.data() + ntrain, REAL(Y_in));
      double ymean = Ytrain.mean();
      // now subtract the mean from Y_in
      for (int ii = 0; ii < ntrain; ++ii) {
        REAL(Y_in)[ii] -= ymean;
      }

      SEXP X_in   = PROTECT(Rf_allocMatrix(REALSXP, ntrain, npc)); nprot++;
      SEXP lam_in = PROTECT(Rf_allocVector(REALSXP, 1)); nprot++;
      REAL(lam_in)[0] = lambda;
      // (copy already done above)
      std::copy(Xtrain.data(), Xtrain.data() + ntrain * npc, REAL(X_in));

      // alpha0 from ridge
      SEXP alpha0_ = PROTECT(ridge_call(Y_in, X_in, lam_in)); nprot++;

      // pcghal on train
      SEXP ENn_in  = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_), npc)); nprot++;
      std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_) * npc, REAL(ENn_in));

      SEXP out = PROTECT(pcghal_call(Y_in, X_in, ENn_in, alpha0_,
                                     max_iter_, tol_, step_factor_, verbose_, crit_)); nprot++;

      // alpha from pcghal result
      SEXP alpha_out = VECTOR_ELT(out, 0);
      Map<VectorXd> alpha_hat(REAL(alpha_out), Rf_length(alpha_out));

      VectorXd y_pred = Xtest * alpha_hat;
      // if centering was done, need to add back mean of Ytrain
      if (center) {
          double ymean = Ytrain.mean();
          y_pred.array() += ymean;
      }
      double mse = (Ytest - y_pred).squaredNorm() / (double)ntest;
      fold_error(i - 1, j) = mse;

      UNPROTECT(nprot);
    }
  }

  // mses = column means (apply(fold.error.m, 2, mean))
  VectorXd mses(L);
  for (int j = 0; j < L; ++j) {
    double sum = 0.0; int cnt = 0;
    for (int i = 0; i < K; ++i) {
      double v = fold_error(i, j);
      if (!std::isnan(v)) { sum += v; cnt++; }
    }
    mses[j] = (cnt > 0) ? (sum / cnt) : NA_REAL;
  }

  // Find best lambda
  int best_idx = 0;
  double best_val = mses[0];
  for (int j = 1; j < L; ++j) {
    if (std::isnan(mses[j])) continue;
    if (std::isnan(best_val) || mses[j] < best_val) { best_val = mses[j]; best_idx = j; }
  }
  const double best_lambda = lambdas[best_idx];

  // Refit on full data at best λ
  SEXP Y_full  = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  // save the mean of Y_full
  Map<const VectorXd> Yfull_map(REAL(Y_), n);
  double ymean = Yfull_map.mean();
  // now subtract the mean from Y_full if center=1
  if (center) {
      for (int i = 0; i < n; ++i) {
          REAL(Y_full)[i] = REAL(Y_)[i] - ymean;
      }
  } else {
      std::copy(REAL(Y_), REAL(Y_) + n, REAL(Y_full));
  }
  SEXP X_full  = PROTECT(Rf_allocMatrix(REALSXP, n, npc)); prot++;
  SEXP lam_full= PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(lam_full)[0] = best_lambda;
  std::copy(REAL(Y_), REAL(Y_) + n, REAL(Y_full));
  std::copy(Xtilde.data(), Xtilde.data() + n * npc, REAL(X_full));

  SEXP alpha_full = PROTECT(ridge_call(Y_full, X_full, lam_full)); prot++;

  SEXP ENn_full = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_), npc)); prot++;
  std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_) * npc, REAL(ENn_full));

  SEXP res_opt = PROTECT(pcghal_call(Y_full, X_full, ENn_full, alpha_full,
                                     max_iter_, tol_, step_factor_, verbose_, crit_)); prot++;

                                       // === Optional predictions on new data ===
  SEXP predictions_out = R_NilValue;
  if (!Rf_isNull(predict_)) {
    if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
      Rf_error("predict must be a numeric matrix with the same number of columns as X.");
    const int m_pred = Rf_nrows(predict_);

    // ktest = har.kernel.cross(X, predict)
    int nprot_pred = 0;
    SEXP ktest_sexp = PROTECT(kernel_cross_call(X_, predict_, maxdeg_, center_)); nprot_pred++;
    Map<const MatrixXd> Ktest(REAL(ktest_sexp), m_pred, n);

    // Extract alpha from res_opt (first element of pcghal result)
    SEXP alpha_out = VECTOR_ELT(res_opt, 0);
    if (!Rf_isReal(alpha_out))
      Rf_error("pcghal_call result[0] (alpha) must be numeric.");
    const int alpha_len = Rf_length(alpha_out);
    if (alpha_len != npc)
      Rf_error("alpha length (%d) != npc (%d).", alpha_len, npc);
    Map<const VectorXd> alpha_hat(REAL(alpha_out), npc);

    // Build U %*% diag(1/d) %*% alpha   (no sqrt)
    MatrixXd U_npc = U.leftCols(npc);                 // n x npc
    VectorXd invd  = d.head(npc).cwiseInverse();      // npc
    VectorXd tmp   = invd.asDiagonal() * alpha_hat;   // npc
    VectorXd v     = U_npc * tmp;                     // n

    // predictions = ktest %*% v    -> length m_pred
    VectorXd preds = Ktest * v;
    // if centering was done, need to add back mean of Y_full
    if (center) {       
        preds.array() += ymean;
    }

    // Return as a numeric vector (m_pred)
    predictions_out = PROTECT(Rf_allocVector(REALSXP, m_pred)); nprot_pred++;
    std::copy(preds.data(), preds.data() + m_pred, REAL(predictions_out));

    UNPROTECT(nprot_pred);
  }
  // Build return list
  SEXP mses_out     = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
  for (int j = 0; j < L; ++j) REAL(mses_out)[j] = mses[j];

  SEXP lambdas_out  = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
  for (int j = 0; j < L; ++j) REAL(lambdas_out)[j] = lambdas[j];

  SEXP best_lambda_ = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(best_lambda_)[0] = best_lambda;
    const int n_out = Rf_isNull(predict_) ? 4 : 5;
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
  SET_VECTOR_ELT(out_final, 0, mses_out);
  SET_VECTOR_ELT(out_final, 1, lambdas_out);
  SET_VECTOR_ELT(out_final, 2, best_lambda_);
  SET_VECTOR_ELT(out_final, 3, res_opt);
  if (n_out == 5) {
    SET_VECTOR_ELT(out_final, 4, predictions_out);
  }

  SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
  SET_STRING_ELT(names, 0, Rf_mkChar("mses"));
  SET_STRING_ELT(names, 1, Rf_mkChar("lambdas"));
  SET_STRING_ELT(names, 2, Rf_mkChar("best_lambda"));
  SET_STRING_ELT(names, 3, Rf_mkChar("res_opt"));
  if (n_out == 5) {
    SET_STRING_ELT(names, 4, Rf_mkChar("predictions"));
  }
  Rf_setAttrib(out_final, R_NamesSymbol, names);

 

  UNPROTECT(prot);
  return out_final;
}