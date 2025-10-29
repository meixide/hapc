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
#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;

extern "C" SEXP fast_pchal_call(SEXP U_, SEXP D2_, SEXP Y_, SEXP lambda_);
extern "C" SEXP mkernel_call(SEXP X_, SEXP m_, SEXP center_);
extern "C" SEXP kernel_cross_call(SEXP X_, SEXP X2_, SEXP m_, SEXP center_);

extern "C" SEXP fasthal_cv_call(SEXP X_, SEXP Y_, SEXP npc_,
                              SEXP lambdas_, SEXP nfolds_, SEXP predict_, SEXP m_, SEXP center_) {
  if (!Rf_isReal(X_) || !Rf_isReal(Y_))
    Rf_error("X and Y must be numeric.");
  const int n  = Rf_nrows(X_);
  const int p  = Rf_ncols(X_);
  if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");
  int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];
  const int nfolds = Rf_isInteger(nfolds_) ? INTEGER(nfolds_)[0] : (int)REAL(nfolds_)[0];
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

  // Compute kernel matrix K
  SEXP K_sexp = PROTECT(mkernel_call(X_, m_, center_)); prot++;
  Map<const MatrixXd> K(REAL(K_sexp), n, n);
  Rprintf("Number of arguments received: 6? predict_ is %s\n",
        Rf_isNull(predict_) ? "NULL" : "non-NULL");

  // Diagonalize K: K = U * D * U^T where D is diagonal
  SelfAdjointEigenSolver<MatrixXd> solver(K);
  if (solver.info() != Eigen::Success)
    Rf_error("Kernel matrix eigendecomposition failed.");
  
  VectorXd eigvals = solver.eigenvalues();
  MatrixXd eigvecs = solver.eigenvectors();
  
  // Reverse order to get descending eigenvalues
  VectorXd D2 = eigvals.reverse();
  MatrixXd U = eigvecs.rowwise().reverse();

  // Print first five elements of D2 and the top-left 5x5 block of U
  Rprintf("First five eigenvalues (D2): ");
  for (int i = 0; i < std::min(5, (int)D2.size()); ++i) {
      Rprintf("%f ", D2[i]);
  }
  Rprintf("\nFirst 5x5 block of U:\n");
  for (int i = 0; i < std::min(5, (int)U.rows()); ++i) {
      for (int j = 0; j < std::min(5, (int)U.cols()); ++j) {
          Rprintf("%f ", U(i, j));
      }
      Rprintf("\n");
  }

  

  // Create Xtilde = U[:, 1:npc] * D2[1:npc]^(1/2)
  MatrixXd Xtilde = U.leftCols(npc) * D2.head(npc).cwiseSqrt().asDiagonal();
  
  Map<const VectorXd> Y(REAL(Y_), n);
  
  // Create folds
  std::vector<int> folds(n);
  const int fold_size = n / nfolds;
  for (int i = 0; i < n; ++i) {
    folds[i] = (i / fold_size) + 1;
  }
  for (int i = fold_size * nfolds; i < n; ++i) {
    folds[i] = nfolds;
  }
  std::mt19937 rng(12345);
  std::shuffle(folds.begin(), folds.end(), rng);
  
  MatrixXd fold_error = MatrixXd::Constant(nfolds, L, std::numeric_limits<double>::quiet_NaN());
  
  for (int j = 0; j < L; ++j) {
    const double lambda = lambdas[j];
    for (int i = 1; i <= nfolds; ++i) {
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
      
      // Extract train and test data
      MatrixXd Xtest(ntest, npc);            // only need Xtest for prediction
      MatrixXd Utrain(ntrain, npc);          // need eigenvector rows for training
      VectorXd Ytrain(ntrain), Ytest(ntest);
      
      for (int ii = 0; ii < ntrain; ++ii) {
        Utrain.row(ii) = U.row(train_idx[ii]).leftCols(npc);
        Ytrain[ii]     = Y[train_idx[ii]];
      }
      for (int ii = 0; ii < ntest; ++ii) {
        Xtest.row(ii) = Xtilde.row(test_idx[ii]);
        Ytest[ii]     = Y[test_idx[ii]];
      }
      
      // Prepare inputs for fast_pchal_call
      int nprot = 0;
      SEXP Y_train = PROTECT(Rf_allocVector(REALSXP, ntrain)); nprot++;
      SEXP U_train = PROTECT(Rf_allocMatrix(REALSXP, ntrain, npc)); nprot++;
      SEXP D2_train = PROTECT(Rf_allocVector(REALSXP, npc)); nprot++;
      SEXP lam_in = PROTECT(Rf_allocVector(REALSXP, 1)); nprot++;
      
      std::copy(Ytrain.data(), Ytrain.data() + ntrain, REAL(Y_train));
      std::copy(Utrain.data(), Utrain.data() + ntrain * npc, REAL(U_train));
      std::copy(D2.data(), D2.data() + npc, REAL(D2_train));
      REAL(lam_in)[0] = lambda;
      
      SEXP beta_out = PROTECT(fast_pchal_call(U_train, D2_train, Y_train, lam_in)); nprot++;
      
      // Extract predictions - beta has length npc
      if (!Rf_isReal(beta_out))
        Rf_error("fast_pchal_call must return a numeric vector");
      
      Map<VectorXd> alpha_hat(REAL(beta_out), npc);
      VectorXd y_pred = Xtest * alpha_hat; // this is correct, Xtest should be called Xtildetest
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
  
  // Compute mean CV error per lambda
  VectorXd mses(L);
  for (int j = 0; j < L; ++j) {
    double sum = 0.0; 
    int cnt = 0;
    for (int i = 0; i < nfolds; ++i) {
      double v = fold_error(i, j);
      if (!std::isnan(v)) { 
        sum += v; 
        cnt++; 
      }
    }
    mses[j] = (cnt > 0) ? (sum / cnt) : NA_REAL;
  }
  
  // Find best lambda
  int best_idx = 0;
  double best_val = mses[0];
  for (int j = 1; j < L; ++j) {
    if (std::isnan(mses[j])) continue;
    if (std::isnan(best_val) || mses[j] < best_val) { 
      best_val = mses[j]; 
      best_idx = j; 
    }
  }
  const double best_lambda = lambdas[best_idx];
  
  // Refit on full data with best lambda
  SEXP Y_full = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  SEXP U_full = PROTECT(Rf_allocMatrix(REALSXP, n, npc)); prot++;
  SEXP D2_full = PROTECT(Rf_allocVector(REALSXP, npc)); prot++;
  SEXP lam_full = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;

  
  std::copy(Y.data(), Y.data() + n, REAL(Y_full));
  std::copy(U.data(), U.data() + n * npc, REAL(U_full));
  std::copy(D2.data(), D2.data() + npc, REAL(D2_full));
  REAL(lam_full)[0] = best_lambda;

  
  SEXP res_opt = PROTECT(fast_pchal_call(U_full, D2_full, Y_full, lam_full)); prot++;


   SEXP predictions_out = R_NilValue;
  if (!Rf_isNull(predict_)) {
    if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
      Rf_error("predict must be a numeric matrix with the same number of columns as X.");
    const int m_pred = Rf_nrows(predict_);
    SEXP ktest_sexp = PROTECT(kernel_cross_call(X_, predict_, m_, center_)); prot++;
    Map<const MatrixXd> Ktest(REAL(ktest_sexp), m_pred, n);

    MatrixXd D2inv_sqrt = D2.head(npc).cwiseSqrt().cwiseInverse().asDiagonal();
    Map<VectorXd> alpha_hat(REAL(res_opt), npc);
    MatrixXd predictions = Ktest * U.leftCols(npc) * D2inv_sqrt * alpha_hat;
    if (center) {       
        double ymean = Y.mean();
        predictions.array() += ymean;
    }

    predictions_out = PROTECT(Rf_allocMatrix(REALSXP, m_pred, 1)); prot++;
    std::copy(predictions.data(), predictions.data() + m_pred, REAL(predictions_out));

  }
  
  // Build return list
  SEXP mses_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
  for (int j = 0; j < L; ++j) REAL(mses_out)[j] = mses[j];
  
  SEXP lambdas_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
  for (int j = 0; j < L; ++j) REAL(lambdas_out)[j] = lambdas[j];
  
  SEXP best_lambda_out = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(best_lambda_out)[0] = best_lambda;
  
  const int n_out = (predictions_out == R_NilValue) ? 4 : 5;
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;  SET_VECTOR_ELT(out_final, 0, mses_out);
  SET_VECTOR_ELT(out_final, 1, lambdas_out);
  SET_VECTOR_ELT(out_final, 2, best_lambda_out);
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