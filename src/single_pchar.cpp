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

extern "C" SEXP mkernel_call(SEXP X_, SEXP m_, SEXP center_);
extern "C" SEXP kernel_cross_call(SEXP X_, SEXP X2_, SEXP m_, SEXP center_);

// UPDATED: ridge_call now takes 4 arguments (Y, U, D2, lambda)
extern "C" SEXP ridge_call(SEXP Y_, SEXP U_, SEXP D2_, SEXP lambda_);

// Simple power iteration for top k eigenvectors
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

// --------------------------------------------------------
// NEW FUNCTION: single_lambda_pchar
// --------------------------------------------------------
extern "C" SEXP single_lambda_pchar(SEXP X_, SEXP Y_, SEXP npc_, 
                                    SEXP lambda_, SEXP predict_, SEXP m_, SEXP center_, SEXP approx_) {
  if (!Rf_isReal(X_) || !Rf_isReal(Y_))
    Rf_error("X and Y must be numeric.");
    
  const int n = Rf_nrows(X_);
  const int p = Rf_ncols(X_);
  if (Rf_length(Y_) != n) Rf_error("length(Y) must equal nrow(X).");
  
  int npc = Rf_isInteger(npc_) ? INTEGER(npc_)[0] : (int)REAL(npc_)[0];

  // Centering logic
  bool center = true;
  if (Rf_isLogical(center_)) center = LOGICAL(center_)[0];
  else Rf_error("center must be logical");
  
  // Approximation flag
  bool approx = false;
  if (Rf_isLogical(approx_)) approx = LOGICAL(approx_)[0];
  else Rf_error("approx must be logical");
  
  if (center) {
      if (npc >= n) { npc = n - 1; Rf_warning("npc reduced to n - 1 due to centering."); }
  } else {
      if (npc > n) { npc = n; Rf_warning("npc reduced to n due to no centering."); }
  }

  int prot = 0;

  // Compute Kernel Matrix
  SEXP K_sexp = PROTECT(mkernel_call(X_, m_, center_)); prot++;
  Map<const MatrixXd> K(REAL(K_sexp), n, n);

  // Eigen decomposition
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
    // Extract top npc eigenvalues and eigenvectors (in descending order)
    VectorXd eigenvalues = eigensolver.eigenvalues();
    MatrixXd eigenvectors = eigensolver.eigenvectors();
    
    // Eigenvalues are in ascending order, so reverse to get descending
    for (int i = 0; i < npc; ++i) {
      int idx = n - 1 - i;
      D2(i) = eigenvalues(idx);
      U.col(i) = eigenvectors.col(idx);
    }
  }

  // Prepare data for ridge_call (full dataset)
  SEXP U_sexp = PROTECT(Rf_allocMatrix(REALSXP, n, npc)); prot++;
  SEXP D2_sexp = PROTECT(Rf_allocVector(REALSXP, npc)); prot++;
  
  std::copy(U.data(), U.data() + n * npc, REAL(U_sexp));
  std::copy(D2.data(), D2.data() + npc, REAL(D2_sexp));
  
  // Call ridge_call directly
  SEXP res_opt = PROTECT(ridge_call(Y_, U_sexp, D2_sexp, lambda_)); prot++;

  // Predictions (if needed)
  SEXP predictions_out = R_NilValue;
  if (!Rf_isNull(predict_)) {
      if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
        Rf_error("predict must be numeric matrix with same cols as X");
      
      const int m_pred = Rf_nrows(predict_);
      SEXP ktest_sexp = PROTECT(kernel_cross_call(X_, predict_, m_, center_)); prot++;
      Map<const MatrixXd> Ktest(REAL(ktest_sexp), m_pred, n);
      
      MatrixXd D2inv_sqrt = D2.cwiseSqrt().cwiseInverse().asDiagonal();
      Map<VectorXd> alpha_hat(REAL(res_opt), npc);
      
      MatrixXd predictions = Ktest * U * D2inv_sqrt * alpha_hat;
      
      if (center) {
          Map<const VectorXd> Y_vec(REAL(Y_), n);
          double ymean = Y_vec.mean();
          predictions.array() += ymean;
      }
      
      predictions_out = PROTECT(Rf_allocMatrix(REALSXP, m_pred, 1)); prot++;
      std::copy(predictions.data(), predictions.data() + m_pred, REAL(predictions_out));
  }

  // Return list
  const int n_out = (predictions_out == R_NilValue) ? 1 : 2;
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
  
  SET_VECTOR_ELT(out_final, 0, res_opt);
  if (n_out == 2) {
      SET_VECTOR_ELT(out_final, 1, predictions_out);
  }
  
  SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
  SET_STRING_ELT(names, 0, Rf_mkChar("alpha"));
  if (n_out == 2) {
      SET_STRING_ELT(names, 1, Rf_mkChar("predictions"));
  }
  
  Rf_setAttrib(out_final, R_NamesSymbol, names);
  
  UNPROTECT(prot);
  return out_final;
}