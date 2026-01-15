

#include <RcppEigen.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// --------------------------------------------------------------------------
// External Function Declarations
// --------------------------------------------------------------------------

extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_);

// UPDATED: Matches usage (Y, U, D2, lambda)
extern "C" SEXP ridge_call(SEXP Y_, SEXP U_, SEXP D2_, SEXP lambda_);

extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_);
extern "C" SEXP pcghal_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                             SEXP max_iter_, SEXP tol_, SEXP step_factor_, SEXP verbose_, SEXP crit_);

// --------------------------------------------------------------------------
// CV Function
// --------------------------------------------------------------------------
extern "C" SEXP pchal_cv_call(SEXP X_, SEXP Y_, SEXP maxdeg_, SEXP npc_,
                              SEXP lambdas_, SEXP nfolds_,
                              SEXP max_iter_, SEXP tol_, SEXP step_factor_,
                              SEXP verbose_, SEXP crit_,
                              SEXP predict_, SEXP center_) {
  
  // 1. Input Validation
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

  // 2. Logic for Centering & NPC Cap
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

  // 3. Generate Design
  // FIX: Pass the corrected 'npc' as a scalar integer SEXP, not the raw 'npc_'
  SEXP npc_sexp = PROTECT(Rf_ScalarInteger(npc)); prot++;
  SEXP des_out  = PROTECT(pchal_des(X_, maxdeg_, npc_sexp, center_)); prot++;
  
  SEXP U_sexp = VECTOR_ELT(des_out, 1);
  SEXP d_sexp = VECTOR_ELT(des_out, 2);
  SEXP V_sexp = VECTOR_ELT(des_out, 3);

  Map<const MatrixXd> U_full(REAL(U_sexp), Rf_nrows(U_sexp), Rf_ncols(U_sexp));
  Map<const VectorXd> d_full(REAL(d_sexp), Rf_length(d_sexp));
  Map<const MatrixXd> V_full(REAL(V_sexp), Rf_nrows(V_sexp), Rf_ncols(V_sexp));

  // FIX: Determine actual NPC (in case pchal_des returned fewer components)
  int final_npc = (npc < d_full.size()) ? npc : d_full.size();

  MatrixXd Xtilde = U_full.leftCols(final_npc) * d_full.head(final_npc).asDiagonal();
  MatrixXd E_Nn   = V_full.leftCols(final_npc);
  Map<const VectorXd> Y_raw_map(REAL(Y_), n);

  // 4. Set up Folds
  std::vector<int> folds(n);
  const int fold_size = n / K;
  for (int i = 0; i < n; ++i) folds[i] = (i / fold_size) + 1;
  for (int i = fold_size * K; i < n; ++i) folds[i] = K;
  
  std::mt19937 rng(12345);
  std::shuffle(folds.begin(), folds.end(), rng);

  MatrixXd fold_error = MatrixXd::Constant(K, L, std::numeric_limits<double>::quiet_NaN());

  // 5. CV Loop
  for (int j = 0; j < L; ++j) {
    const double lambda = lambdas[j];

    for (int i = 1; i <= K; ++i) {
      std::vector<int> test_idx, train_idx;
      for (int ii = 0; ii < n; ++ii) {
        if (folds[ii] == i) test_idx.push_back(ii);
        else train_idx.push_back(ii);
      }

      const int ntrain = (int)train_idx.size();
      const int ntest  = (int)test_idx.size();
      if (ntrain == 0 || ntest == 0) { 
        fold_error(i - 1, j) = NA_REAL;
        continue; 
      }

      // Subset Data
      MatrixXd Xtrain(ntrain, final_npc), Xtest(ntest, final_npc);
      MatrixXd Utrain(ntrain, final_npc);
      VectorXd Ytrain(ntrain), Ytest(ntest);

      for (int ii = 0; ii < ntrain; ++ii) {
        Xtrain.row(ii) = Xtilde.row(train_idx[ii]);
        Utrain.row(ii) = U_full.leftCols(final_npc).row(train_idx[ii]);
        Ytrain[ii]     = Y_raw_map[train_idx[ii]];
      }
      for (int ii = 0; ii < ntest; ++ii) {
        Xtest.row(ii) = Xtilde.row(test_idx[ii]);
        Ytest[ii]     = Y_raw_map[test_idx[ii]];
      }

      // Inner protect block
      int nprot = 0;

      // Prepare Y_in (Centered)
      SEXP Y_in = PROTECT(Rf_allocVector(REALSXP, ntrain)); nprot++;
      std::copy(Ytrain.data(), Ytrain.data() + ntrain, REAL(Y_in));
      double ymean_train = 0.0;
      if (center) {
          ymean_train = Ytrain.mean();
          for (int ii = 0; ii < ntrain; ++ii) REAL(Y_in)[ii] -= ymean_train;
      }

      // Prepare X_in (for pcghal_call)
      SEXP X_in = PROTECT(Rf_allocMatrix(REALSXP, ntrain, final_npc)); nprot++;
      std::copy(Xtrain.data(), Xtrain.data() + ntrain * final_npc, REAL(X_in));
      
      // Prepare U_in and D2_in (for ridge_call)
      // Note: We use the subset of U, and the global D^2 (as per original code logic)
      SEXP U_in = PROTECT(Rf_allocMatrix(REALSXP, ntrain, final_npc)); nprot++;
      std::copy(Utrain.data(), Utrain.data() + ntrain * final_npc, REAL(U_in));

      SEXP D2_in = PROTECT(Rf_allocVector(REALSXP, final_npc)); nprot++;
      VectorXd d_squared = d_full.head(final_npc).array().square();
      std::copy(d_squared.data(), d_squared.data() + final_npc, REAL(D2_in));

      SEXP lam_in = PROTECT(Rf_allocVector(REALSXP, 1)); nprot++;
      REAL(lam_in)[0] = lambda;

      // Initialize alpha using Ridge (Approximate due to non-orthogonal subset U)
      SEXP alpha0_ = PROTECT(ridge_call(Y_in, U_in, D2_in, lam_in)); nprot++;

      // PC-GHAL on train
      SEXP ENn_in  = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_sexp), final_npc)); nprot++;
      std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_sexp) * final_npc, REAL(ENn_in));

      SEXP out = PROTECT(pcghal_call(Y_in, X_in, ENn_in, alpha0_,
                                     max_iter_, tol_, step_factor_, verbose_, crit_)); nprot++;

      // Prediction
      SEXP alpha_out = VECTOR_ELT(out, 0);
      Map<VectorXd> alpha_hat(REAL(alpha_out), Rf_length(alpha_out));

      VectorXd y_pred = Xtest * alpha_hat;
      if (center) {
          y_pred.array() += ymean_train;
      }
      double mse = (Ytest - y_pred).squaredNorm() / (double)ntest;
      fold_error(i - 1, j) = mse;

      UNPROTECT(nprot);
    }
  }

  // 6. Aggregate Results
  VectorXd mses(L);
  for (int j = 0; j < L; ++j) {
    double sum = 0.0; int cnt = 0;
    for (int i = 0; i < K; ++i) {
      double v = fold_error(i, j);
      if (!std::isnan(v)) { sum += v; cnt++; }
    }
    mses[j] = (cnt > 0) ? (sum / cnt) : NA_REAL;
  }

  int best_idx = 0;
  double best_val = mses[0];
  for (int j = 1; j < L; ++j) {
    if (std::isnan(mses[j])) continue;
    if (std::isnan(best_val) || mses[j] < best_val) { best_val = mses[j]; best_idx = j; }
  }
  const double best_lambda = lambdas[best_idx];

  // --------------------------------------------------------------------------
  // 7. Refit on Full Data (Use Global Components)
  // --------------------------------------------------------------------------
  
  // Prepare Y_full
  SEXP Y_full = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  double ymean_full = 0.0;
  if (center) {
      ymean_full = Y_raw_map.mean();
      for (int i = 0; i < n; ++i) REAL(Y_full)[i] = Y_raw_map[i] - ymean_full;
  } else {
      std::copy(Y_raw_map.data(), Y_raw_map.data() + n, REAL(Y_full));
  }

  // Prepare X_full
  SEXP X_full = PROTECT(Rf_allocMatrix(REALSXP, n, final_npc)); prot++;
  std::copy(Xtilde.data(), Xtilde.data() + n * final_npc, REAL(X_full));

  // Prepare Inputs for Ridge (Global U, D2)
  SEXP U_fit = PROTECT(Rf_allocMatrix(REALSXP, n, final_npc)); prot++;
  SEXP D2_fit = PROTECT(Rf_allocVector(REALSXP, final_npc)); prot++;
  
  MatrixXd U_sub_mat = U_full.leftCols(final_npc);
  VectorXd D2_sub_vec = d_full.head(final_npc).array().square();

  std::copy(U_sub_mat.data(), U_sub_mat.data() + n * final_npc, REAL(U_fit));
  std::copy(D2_sub_vec.data(), D2_sub_vec.data() + final_npc, REAL(D2_fit));

  SEXP lam_full = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(lam_full)[0] = best_lambda;

  SEXP alpha_full = PROTECT(ridge_call(Y_full, U_fit, D2_fit, lam_full)); prot++;

  SEXP ENn_full = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_sexp), final_npc)); prot++;
  std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_sexp) * final_npc, REAL(ENn_full));

  SEXP res_opt = PROTECT(pcghal_call(Y_full, X_full, ENn_full, alpha_full,
                                     max_iter_, tol_, step_factor_, verbose_, crit_)); prot++;

  // --------------------------------------------------------------------------
  // 8. Predictions
  // --------------------------------------------------------------------------
  SEXP predictions_out = R_NilValue;
  if (!Rf_isNull(predict_)) {
    if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
      Rf_error("predict must be a numeric matrix with the same number of columns as X.");
    const int m_pred = Rf_nrows(predict_);

    int nprot_pred = 0;
    SEXP ktest_sexp = PROTECT(kernel_cross_call(X_, predict_, maxdeg_, center_)); nprot_pred++;
    Map<const MatrixXd> Ktest(REAL(ktest_sexp), m_pred, n);

    SEXP alpha_out = VECTOR_ELT(res_opt, 0);
    if (!Rf_isReal(alpha_out)) Rf_error("Invalid alpha output.");
    Map<const VectorXd> alpha_hat(REAL(alpha_out), final_npc);

    VectorXd d_inv = d_full.head(final_npc).cwiseInverse();
    VectorXd v = U_sub_mat * (d_inv.asDiagonal() * alpha_hat);

    VectorXd preds = Ktest * v;
    if (center) preds.array() += ymean_full;

    predictions_out = PROTECT(Rf_allocVector(REALSXP, m_pred)); nprot_pred++;
    std::copy(preds.data(), preds.data() + m_pred, REAL(predictions_out));

    UNPROTECT(nprot_pred);
  }

  // 9. Return List
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
  if (n_out == 5) SET_VECTOR_ELT(out_final, 4, predictions_out);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
  SET_STRING_ELT(names, 0, Rf_mkChar("mses"));
  SET_STRING_ELT(names, 1, Rf_mkChar("lambdas"));
  SET_STRING_ELT(names, 2, Rf_mkChar("best_lambda"));
  SET_STRING_ELT(names, 3, Rf_mkChar("res_opt"));
  if (n_out == 5) SET_STRING_ELT(names, 4, Rf_mkChar("predictions"));
  Rf_setAttrib(out_final, R_NamesSymbol, names);

  UNPROTECT(prot);
  return out_final;
}