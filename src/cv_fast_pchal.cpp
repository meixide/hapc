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
#include "hapc_core.hpp"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <RcppEigen.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

// External function declarations
extern "C" SEXP fast_pchal_call(SEXP U_, SEXP D2_, SEXP Y_, SEXP lambda_);
extern "C" SEXP ridge_call(SEXP Y_, SEXP U_, SEXP D2_, SEXP lambda_);
extern "C" SEXP mkernel_call(SEXP X_, SEXP m_, SEXP center_);
extern "C" SEXP kernel_cross_call(SEXP X_, SEXP X2_, SEXP m_, SEXP center_);

// Simple power iteration for top k eigenvectors (used by both R and Python)
static void power_iteration_top_k(const MatrixXd& A, int k, MatrixXd& V, VectorXd& D) {
  const int n = A.rows();
  V.resize(n, k);
  D.resize(k);
  
  MatrixXd A_deflated = A;
  
  for (int i = 0; i < k; ++i) {
    // Random initialization
    VectorXd v = VectorXd::Random(n);
    v.normalize();
    
    // Power iteration
    double lambda_old = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
      VectorXd v_new = A_deflated * v;
      double lambda = v_new.norm();
      v_new.normalize();
      
      if (std::abs(lambda - lambda_old) < 1e-9) break;
      
      v = v_new;
      lambda_old = lambda;
    }
    
    // Store eigenvector and eigenvalue
    V.col(i) = v;
    D(i) = v.dot(A_deflated * v);
    
    // Deflate matrix: A := A - lambda * v * v^T
    A_deflated -= D(i) * v * v.transpose();
  }
}

// R-specific CV function
extern "C" SEXP fasthal_cv_call(SEXP X_, SEXP Y_, SEXP npc_,
                              SEXP lambdas_, SEXP nfolds_, SEXP predict_, SEXP m_, SEXP center_, SEXP approx_, SEXP l1_) {
  // 1. Input Validation
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

  // 2. Boolean Flag Logic
  bool center = true;
  if (Rf_isLogical(center_)) center = LOGICAL(center_)[0];
  else Rf_error("center must be logical");
  
  bool approx = false;
  if (Rf_isLogical(approx_)) approx = LOGICAL(approx_)[0];
  else Rf_error("approx must be logical");
  
  bool l1 = false;
  if (Rf_isLogical(l1_)) l1 = LOGICAL(l1_)[0];
  else Rf_error("l1 must be logical");
  
  // Adjust npc based on centering
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

  // 3. Compute Kernel and Eigen Decomposition
  SEXP K_sexp = PROTECT(mkernel_call(X_, m_, center_)); prot++;
  Map<const MatrixXd> K(REAL(K_sexp), n, n);

  MatrixXd U(n, npc);
  VectorXd D2(npc);
  
  if (approx) {
    Rprintf("Using approximate eigendecomposition (power iteration)\n");
    power_iteration_top_k(K, npc, U, D2);
  } else {
    Rprintf("Using exact eigendecomposition\n");
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(K);
    if (eigensolver.info() != Eigen::Success) {
      Rf_error("Eigendecomposition failed");
    }
    VectorXd eigenvalues = eigensolver.eigenvalues();
    MatrixXd eigenvectors = eigensolver.eigenvectors();
    
    // Sort descending
    for (int i = 0; i < npc; ++i) {
      int idx = n - 1 - i;
      D2(i) = eigenvalues(idx);
      U.col(i) = eigenvectors.col(idx);
    }
  }

  // Create Xtilde = U * D2^(1/2)
  MatrixXd Xtilde = U * D2.cwiseSqrt().asDiagonal();
  
  Map<const VectorXd> Y(REAL(Y_), n);
  
  // 4. Create Folds
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
  
  // 5. Cross Validation Loop
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
      
      MatrixXd Xtest(ntest, npc);
      MatrixXd Utrain(ntrain, npc);
      VectorXd Ytrain(ntrain), Ytest(ntest);
      
      for (int ii = 0; ii < ntrain; ++ii) {
        Utrain.row(ii) = U.row(train_idx[ii]);
        Ytrain[ii]     = Y[train_idx[ii]];
      }
      for (int ii = 0; ii < ntest; ++ii) {
        Xtest.row(ii) = Xtilde.row(test_idx[ii]);
        Ytest[ii]     = Y[test_idx[ii]];
      }
      
      // Compute mean and center Ytrain (Conditionally)
      double ymean = 0.0;
      if (center) {
          ymean = Ytrain.mean();
          Ytrain.array() -= ymean;
      }
      
      // Prepare inputs for solver
      int nprot = 0;
      SEXP Y_train = PROTECT(Rf_allocVector(REALSXP, ntrain)); nprot++;
      SEXP U_train = PROTECT(Rf_allocMatrix(REALSXP, ntrain, npc)); nprot++;
      SEXP D2_train = PROTECT(Rf_allocVector(REALSXP, npc)); nprot++;
      SEXP lam_in = PROTECT(Rf_allocVector(REALSXP, 1)); nprot++;
      
      std::copy(Ytrain.data(), Ytrain.data() + ntrain, REAL(Y_train));
      std::copy(Utrain.data(), Utrain.data() + ntrain * npc, REAL(U_train));
      std::copy(D2.data(), D2.data() + npc, REAL(D2_train));
      REAL(lam_in)[0] = lambda;
      
      SEXP beta_out;
      if (l1) {
        beta_out = PROTECT(fast_pchal_call(U_train, D2_train, Y_train, lam_in)); nprot++;
      } else {
        beta_out = PROTECT(ridge_call(Y_train, U_train, D2_train, lam_in)); nprot++;
      }
      
      if (!Rf_isReal(beta_out))
        Rf_error("Solver must return a numeric vector");
      
      Map<VectorXd> alpha_hat(REAL(beta_out), npc);
      VectorXd y_pred = Xtest * alpha_hat;
      
      // Add mean back (Conditionally)
      if (center) {
          y_pred.array() += ymean;
      }
      
      double mse = (Ytest - y_pred).squaredNorm() / (double)ntest;
      fold_error(i - 1, j) = mse;
      
      UNPROTECT(nprot);
    }
  }
  
  // 6. Select Best Lambda
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
  
  // 7. Refit on Full Data
  SEXP Y_full = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  SEXP U_full = PROTECT(Rf_allocMatrix(REALSXP, n, npc)); prot++;
  SEXP D2_full = PROTECT(Rf_allocVector(REALSXP, npc)); prot++;
  SEXP lam_full = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  
  // Handle Centering for Full Data
  double ymean_full = 0.0;
  if (center) {
      ymean_full = Y.mean();
      VectorXd Y_centered = Y.array() - ymean_full;
      std::copy(Y_centered.data(), Y_centered.data() + n, REAL(Y_full));
  } else {
      std::copy(Y.data(), Y.data() + n, REAL(Y_full));
  }
  
  std::copy(U.data(), U.data() + n * npc, REAL(U_full));
  std::copy(D2.data(), D2.data() + npc, REAL(D2_full));
  REAL(lam_full)[0] = best_lambda;
  
  SEXP res_opt;
  if (l1) {
    Rprintf("Using L1 penalty (LASSO)\n");
    Rprintf("Using new cv\n");
    res_opt = PROTECT(fast_pchal_call(U_full, D2_full, Y_full, lam_full)); prot++;
  } else {
    Rprintf("Using L2 penalty (Ridge)\n");
    Rprintf("Using new cv\n");
    res_opt = PROTECT(ridge_call(Y_full, U_full, D2_full, lam_full)); prot++;
  }

  // 8. Final Prediction (Optional)
  SEXP predictions_out = R_NilValue;
  if (!Rf_isNull(predict_)) {
    if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
      Rf_error("predict must be a numeric matrix with the same number of columns as X.");
    const int m_pred = Rf_nrows(predict_);
    SEXP ktest_sexp = PROTECT(kernel_cross_call(X_, predict_, m_, center_)); prot++;
    Map<const MatrixXd> Ktest(REAL(ktest_sexp), m_pred, n);

    MatrixXd D2inv_sqrt = D2.cwiseSqrt().cwiseInverse().asDiagonal();
    Map<VectorXd> alpha_hat(REAL(res_opt), npc);
    MatrixXd predictions = Ktest * U * D2inv_sqrt * alpha_hat;
    
    // Add mean back (Conditionally)
    if (center) {       
        predictions.array() += ymean_full;
    }

    predictions_out = PROTECT(Rf_allocMatrix(REALSXP, m_pred, 1)); prot++;
    std::copy(predictions.data(), predictions.data() + m_pred, REAL(predictions_out));
  }
  
  // 9. Build Return Object
  SEXP mses_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
  for (int j = 0; j < L; ++j) REAL(mses_out)[j] = mses[j];
  
  SEXP lambdas_out = PROTECT(Rf_allocVector(REALSXP, L)); prot++;
  for (int j = 0; j < L; ++j) REAL(lambdas_out)[j] = lambdas[j];
  
  SEXP best_lambda_out = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(best_lambda_out)[0] = best_lambda;
  
  const int n_out = (predictions_out == R_NilValue) ? 4 : 5;
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
  SET_VECTOR_ELT(out_final, 0, mses_out);
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