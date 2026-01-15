

#include <RcppEigen.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>


// --------------------------------------------------------------------------
// External Function Declarations
// --------------------------------------------------------------------------

extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_);

// Logistic regression function
extern "C" SEXP logistic_call(SEXP Y_, SEXP X_, SEXP lambda_);

extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_);

extern "C" SEXP pcghal_classi_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                                    SEXP max_iter_, SEXP tol_, SEXP step_factor_, 
                                    SEXP verbose_);

// --------------------------------------------------------------------------
// CV Function for Classification
// --------------------------------------------------------------------------
extern "C" SEXP pchal_cv_classi_call(SEXP X_, SEXP Y_, SEXP maxdeg_, SEXP npc_,
                                      SEXP lambdas_, SEXP nfolds_,
                                      SEXP max_iter_, SEXP tol_, SEXP step_factor_,
                                      SEXP verbose_, SEXP crit_,
                                      SEXP predict_, SEXP center_, SEXP single_lambda_) {
  
  // 1. Input Validation
  if (!Rf_isReal(X_) || !Rf_isReal(Y_))
    Rf_error("X and Y must be numeric.");

  const int n  = Rf_nrows(X_);
  const int p  = Rf_ncols(X_);
  if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");

  // Validate Y is binary (0/1)
  Map<const VectorXd> Y_check(REAL(Y_), n);
  for (int i = 0; i < n; ++i) {
    double yi = Y_check[i];
    if (yi != 0.0 && yi != 1.0) {
      Rf_error("Y must contain only 0 and 1 values for classification.");
    }
  }

  int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];
  
  // Check if single_lambda mode
  bool do_single = !Rf_isNull(single_lambda_) && Rf_length(single_lambda_) == 1;
  double single_lambda = do_single ? REAL(single_lambda_)[0] : NA_REAL;
  
  const int K = do_single ? 0 : (Rf_isInteger(nfolds_) ? INTEGER(nfolds_)[0] : (int)REAL(nfolds_)[0]);

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
  SEXP npc_sexp = PROTECT(Rf_ScalarInteger(npc)); prot++;
  SEXP des_out  = PROTECT(pchal_des(X_, maxdeg_, npc_sexp, center_)); prot++;
  
  SEXP U_sexp = VECTOR_ELT(des_out, 1);
  SEXP d_sexp = VECTOR_ELT(des_out, 2);
  SEXP V_sexp = VECTOR_ELT(des_out, 3);

  Map<const MatrixXd> U_full(REAL(U_sexp), Rf_nrows(U_sexp), Rf_ncols(U_sexp));
  Map<const VectorXd> d_full(REAL(d_sexp), Rf_length(d_sexp));
  Map<const MatrixXd> V_full(REAL(V_sexp), Rf_nrows(V_sexp), Rf_ncols(V_sexp));

  int final_npc = (npc < d_full.size()) ? npc : d_full.size();

  MatrixXd Xtilde = U_full.leftCols(final_npc) * d_full.head(final_npc).asDiagonal();
  MatrixXd E_Nn   = V_full.leftCols(final_npc);
  Map<const VectorXd> Y_raw_map(REAL(Y_), n);

  SEXP deviances_out = R_NilValue;
  SEXP lambdas_out = R_NilValue;
  SEXP best_lambda_ = R_NilValue;
  double best_lambda = single_lambda;
  SEXP res_opt = R_NilValue;

  if (do_single) {
    Rprintf("Single lambda mode: lambda = %f\n", single_lambda);
  } else {
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
        VectorXd Ytrain(ntrain), Ytest(ntest);

        for (int ii = 0; ii < ntrain; ++ii) {
          Xtrain.row(ii) = Xtilde.row(train_idx[ii]);
          Ytrain[ii]     = Y_raw_map[train_idx[ii]];
        }
        for (int ii = 0; ii < ntest; ++ii) {
          Xtest.row(ii) = Xtilde.row(test_idx[ii]);
          Ytest[ii]     = Y_raw_map[test_idx[ii]];
        }

        // Inner protect block
        int nprot = 0;

        // Prepare Y_in (Convert from {0,1} to {-1,1} for pcghal_classi_call)
        SEXP Y_in = PROTECT(Rf_allocVector(REALSXP, ntrain)); nprot++;
        for (int ii = 0; ii < ntrain; ++ii) {
          REAL(Y_in)[ii] = (Ytrain[ii] == 1.0) ? 1.0 : -1.0;
        }

        // Prepare X_in (for pcghal_classi_call)
        SEXP X_in = PROTECT(Rf_allocMatrix(REALSXP, ntrain, final_npc)); nprot++;
        std::copy(Xtrain.data(), Xtrain.data() + ntrain * final_npc, REAL(X_in));

        SEXP lam_in = PROTECT(Rf_allocVector(REALSXP, 1)); nprot++;
        REAL(lam_in)[0] = lambda;

        // Initialize alpha using Logistic Regression
        SEXP alpha0_ = PROTECT(logistic_call(Y_in, X_in, lam_in)); nprot++;

        // PC-GHAL Classification on train
        SEXP ENn_in  = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_sexp), final_npc)); nprot++;
        std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_sexp) * final_npc, REAL(ENn_in));

        SEXP out = PROTECT(pcghal_classi_call(Y_in, X_in, ENn_in, alpha0_,
                                               max_iter_, tol_, step_factor_, verbose_)); nprot++;

        // Prediction
        SEXP alpha_out = VECTOR_ELT(out, 0);
        Map<VectorXd> alpha_hat(REAL(alpha_out), Rf_length(alpha_out));

        VectorXd eta = Xtest * alpha_hat;
        VectorXd probs = (1.0 + (-eta.array()).exp()).inverse();

        // Compute binomial deviance (negative log-likelihood)
        double deviance = 0.0;
        for (int ii = 0; ii < ntest; ++ii) {
          double p = std::max(1e-15, std::min(1.0 - 1e-15, probs[ii]));
          if (Ytest[ii] == 1.0) {
            deviance -= std::log(p);
          } else {
            deviance -= std::log(1.0 - p);
          }
        }
        deviance /= ntest; // Average deviance
        
        fold_error(i - 1, j) = deviance;

        UNPROTECT(nprot);
      }
    }

    // 6. Aggregate Results
    VectorXd deviances(L);
    for (int j = 0; j < L; ++j) {
      double sum = 0.0; int cnt = 0;
      for (int i = 0; i < K; ++i) {
        double v = fold_error(i, j);
        if (!std::isnan(v)) { sum += v; cnt++; }
      }
      deviances[j] = (cnt > 0) ? (sum / cnt) : NA_REAL;
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
    best_lambda = lambdas[best_idx];

    deviances_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
    for (int j = 0; j < L; ++j) REAL(deviances_out)[j] = deviances[j];

    lambdas_out  = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
    for (int j = 0; j < L; ++j) REAL(lambdas_out)[j] = lambdas[j];

    best_lambda_ = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
    REAL(best_lambda_)[0] = best_lambda;
  }

  // 7. Refit on Full Data
  SEXP Y_full = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  for (int i = 0; i < n; ++i) {
    REAL(Y_full)[i] = (Y_raw_map[i] == 1.0) ? 1.0 : -1.0;
  }

  SEXP X_full = PROTECT(Rf_allocMatrix(REALSXP, n, final_npc)); prot++;
  std::copy(Xtilde.data(), Xtilde.data() + n * final_npc, REAL(X_full));

  SEXP lam_full = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(lam_full)[0] = best_lambda;

  SEXP alpha_full = PROTECT(logistic_call(Y_full, X_full, lam_full)); prot++;

  SEXP ENn_full = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_sexp), final_npc)); prot++;
  std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_sexp) * final_npc, REAL(ENn_full));

  res_opt = PROTECT(pcghal_classi_call(Y_full, X_full, ENn_full, alpha_full,
                                             max_iter_, tol_, step_factor_, verbose_)); prot++;

  // 8. Predictions
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

    MatrixXd U_sub_mat = U_full.leftCols(final_npc);
    VectorXd d_inv = d_full.head(final_npc).cwiseInverse();
    VectorXd v = U_sub_mat * (d_inv.asDiagonal() * alpha_hat);

    VectorXd eta_pred = Ktest * v;
    VectorXd probs_pred = (1.0 + (-eta_pred.array()).exp()).inverse();

    predictions_out = PROTECT(Rf_allocVector(REALSXP, m_pred)); nprot_pred++;
    std::copy(probs_pred.data(), probs_pred.data() + m_pred, REAL(predictions_out));

    UNPROTECT(nprot_pred);
  }

  // 9. Return List
  const int n_out = do_single ? (Rf_isNull(predict_) ? 1 : 2) : (Rf_isNull(predict_) ? 4 : 5);
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
  
  if (do_single) {
    SET_VECTOR_ELT(out_final, 0, res_opt);
    if (n_out == 2) SET_VECTOR_ELT(out_final, 1, predictions_out);
  } else {
    SET_VECTOR_ELT(out_final, 0, deviances_out);
    SET_VECTOR_ELT(out_final, 1, lambdas_out);
    SET_VECTOR_ELT(out_final, 2, best_lambda_);
    SET_VECTOR_ELT(out_final, 3, res_opt);
    if (n_out == 5) SET_VECTOR_ELT(out_final, 4, predictions_out);
  }

  SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
  if (do_single) {
    SET_STRING_ELT(names, 0, Rf_mkChar("res_opt"));
    if (n_out == 2) SET_STRING_ELT(names, 1, Rf_mkChar("predictions"));
  } else {
    SET_STRING_ELT(names, 0, Rf_mkChar("deviances"));
    SET_STRING_ELT(names, 1, Rf_mkChar("lambdas"));
    SET_STRING_ELT(names, 2, Rf_mkChar("best_lambda"));
    SET_STRING_ELT(names, 3, Rf_mkChar("res_opt"));
    if (n_out == 5) SET_STRING_ELT(names, 4, Rf_mkChar("predictions"));
  }
  Rf_setAttrib(out_final, R_NamesSymbol, names);

  UNPROTECT(prot);
  return out_final;
}