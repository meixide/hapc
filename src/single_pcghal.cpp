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

// Extern declarations
extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_);
extern "C" SEXP ridge_call(SEXP Y_, SEXP X_, SEXP lambda_);
extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_);
extern "C" SEXP pcghal_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                             SEXP max_iter_, SEXP tol_, SEXP step_factor_, SEXP verbose_, SEXP crit_);

// Main function: Single lambda fit without Cross Validation
extern "C" SEXP single_pcghal_call(SEXP X_, SEXP Y_, SEXP maxdeg_, SEXP npc_,
                           SEXP single_lambda_, 
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
  
  // Validate single lambda
  if (Rf_length(single_lambda_) != 1) Rf_error("single_lambda must be a scalar.");
  double lambda = REAL(single_lambda_)[0];

  int prot = 0;

  // 2. Centering Logic and NPC Adjustment
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

  // 3. Construct Design Matrix
  // pchal_des returns list(H, U, d, V)
  SEXP des_out = PROTECT(pchal_des(X_, maxdeg_, npc_, center_)); prot++;
  SEXP U_ = VECTOR_ELT(des_out, 1);
  SEXP d_ = VECTOR_ELT(des_out, 2);
  SEXP V_ = VECTOR_ELT(des_out, 3);

  Map<const MatrixXd> U(REAL(U_), Rf_nrows(U_), Rf_ncols(U_));
  Map<const VectorXd> d(REAL(d_), Rf_length(d_));
  Map<const MatrixXd> V(REAL(V_), Rf_nrows(V_), Rf_ncols(V_));

  // Create Xtilde (PC scores scaled) and E_Nn (Eigenvectors)
  MatrixXd Xtilde = U.leftCols(npc) * d.head(npc).asDiagonal();
  MatrixXd E_Nn   = V.leftCols(npc);
  Map<const VectorXd> Y_raw(REAL(Y_), n);

  // 4. Prepare Y (Centering)
  SEXP Y_fit  = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  double ymean = Y_raw.mean();
  
  if (center) {
      for (int i = 0; i < n; ++i) {
          REAL(Y_fit)[i] = Y_raw[i] - ymean;
      }
  } else {
      std::copy(Y_raw.data(), Y_raw.data() + n, REAL(Y_fit));
      ymean = 0.0;
  }

  // Copy Xtilde for fitting
  SEXP X_fit = PROTECT(Rf_allocMatrix(REALSXP, n, npc)); prot++;
  std::copy(Xtilde.data(), Xtilde.data() + n * npc, REAL(X_fit));

  // Prepare lambda SEXP
  SEXP lam_sexp = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(lam_sexp)[0] = lambda;

  // 5. Fit Model
  // A. Initialize alpha using Ridge Regression
  SEXP alpha0 = PROTECT(ridge_call(Y_fit, X_fit, lam_sexp)); prot++;

  // B. Run PC-GHAL Optimization
  SEXP ENn_fit = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_), npc)); prot++;
  std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_) * npc, REAL(ENn_fit));

  SEXP res_opt = PROTECT(pcghal_call(Y_fit, X_fit, ENn_fit, alpha0,
                                     max_iter_, tol_, step_factor_, verbose_, crit_)); prot++;

  // 6. Generate Predictions (Optional)
  SEXP predictions_out = R_NilValue;
  if (!Rf_isNull(predict_)) {
    if (!Rf_isReal(predict_) || Rf_ncols(predict_) != p)
      Rf_error("predict must be a numeric matrix with the same number of columns as X.");
    const int m_pred = Rf_nrows(predict_);

    // Compute Kernel Cross Product
    int nprot_pred = 0;
    SEXP ktest_sexp = PROTECT(kernel_cross_call(X_, predict_, maxdeg_, center_)); nprot_pred++;
    Map<const MatrixXd> Ktest(REAL(ktest_sexp), m_pred, n);

    // Extract alpha coefficients
    SEXP alpha_out = VECTOR_ELT(res_opt, 0);
    if (!Rf_isReal(alpha_out))
      Rf_error("pcghal_call result[0] (alpha) must be numeric.");
    
    const int alpha_len = Rf_length(alpha_out);
    if (alpha_len != npc)
      Rf_error("alpha length (%d) != npc (%d).", alpha_len, npc);
    Map<const VectorXd> alpha_hat(REAL(alpha_out), npc);

    // Transform coefficients back to prediction space
    // v = U_npc * diag(1/d) * alpha
    MatrixXd U_npc = U.leftCols(npc);                 // n x npc
    VectorXd invd  = d.head(npc).cwiseInverse();      // npc
    VectorXd tmp   = invd.asDiagonal() * alpha_hat;   // npc
    VectorXd v     = U_npc * tmp;                     // n

    // Compute predictions: preds = Ktest * v
    VectorXd preds = Ktest * v;
    
    // Add back mean if centering was performed
    if (center) {       
        preds.array() += ymean;
    }

    // Allocate output vector
    predictions_out = PROTECT(Rf_allocVector(REALSXP, m_pred)); nprot_pred++;
    std::copy(preds.data(), preds.data() + m_pred, REAL(predictions_out));

    UNPROTECT(nprot_pred);
  }

  // 7. Construct Output List
  const int n_out = Rf_isNull(predict_) ? 2 : 3;
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
  
  SET_VECTOR_ELT(out_final, 0, res_opt);
  SET_VECTOR_ELT(out_final, 1, lam_sexp); // Return the used lambda
  
  if (n_out == 3) {
    SET_VECTOR_ELT(out_final, 2, predictions_out);
  }

  SEXP names = PROTECT(Rf_allocVector(STRSXP, n_out)); prot++;
  SET_STRING_ELT(names, 0, Rf_mkChar("res_opt"));
  SET_STRING_ELT(names, 1, Rf_mkChar("lambda"));
  if (n_out == 3) {
    SET_STRING_ELT(names, 2, Rf_mkChar("predictions"));
  }
  Rf_setAttrib(out_final, R_NamesSymbol, names);

  UNPROTECT(prot);
  return out_final;
}