

#include <RcppEigen.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>


// --------------------------------------------------------------------------
// External Function Declarations
// --------------------------------------------------------------------------

// Generates design components: list(H, U, d, V)
extern "C" SEXP pchal_des(SEXP X_, SEXP maxdeg_, SEXP npc_, SEXP center_);

// UPDATED: Ridge now takes (Y, U, D2, lambda) to match your previous code
extern "C" SEXP ridge_call(SEXP Y_, SEXP U_, SEXP D2_, SEXP lambda_);

// Kernel cross-product for prediction
extern "C" SEXP kernel_cross_call(SEXP Xtr_, SEXP Xte_, SEXP m_, SEXP center_);

// Main optimizer for PC-GHAL
extern "C" SEXP pcghal_call(SEXP Y_, SEXP Xtilde_, SEXP ENn_, SEXP alpha0_,
                             SEXP max_iter_, SEXP tol_, SEXP step_factor_, SEXP verbose_, SEXP crit_);

// --------------------------------------------------------------------------
// Main function: Single lambda fit (PC-GHAL)
// --------------------------------------------------------------------------
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
  
  if (Rf_length(single_lambda_) != 1) Rf_error("single_lambda must be a scalar.");
  double lambda = REAL(single_lambda_)[0];

  int prot = 0;

  // 2. Centering Logic
  bool center = true;
  if (Rf_isLogical(center_)) center = LOGICAL(center_)[0];
  else Rf_error("center must be logical");

  // Adjust NPC based on centering (rank reduction)
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

  // 3. Construct Design Matrix (pchal_des)
  // Returns list(H, U, d, V)
  SEXP des_out = PROTECT(pchal_des(X_, maxdeg_, Rf_ScalarInteger(npc), center_)); prot++;
  
  SEXP U_full_sexp = VECTOR_ELT(des_out, 1);
  SEXP d_full_sexp = VECTOR_ELT(des_out, 2);
  SEXP V_full_sexp = VECTOR_ELT(des_out, 3);

  Map<const MatrixXd> U_full(REAL(U_full_sexp), Rf_nrows(U_full_sexp), Rf_ncols(U_full_sexp));
  Map<const VectorXd> d_full(REAL(d_full_sexp), Rf_length(d_full_sexp));
  Map<const MatrixXd> V_full(REAL(V_full_sexp), Rf_nrows(V_full_sexp), Rf_ncols(V_full_sexp));

  // Determine actual NPC (in case pchal_des returned fewer components than requested)
  int final_npc = (npc < d_full.size()) ? npc : d_full.size();

  // 4. Prepare Subsets for Solvers
  
  // A. Create Xtilde = U * d (for pcghal_call)
  MatrixXd Xtilde = U_full.leftCols(final_npc) * d_full.head(final_npc).asDiagonal();
  SEXP X_fit = PROTECT(Rf_allocMatrix(REALSXP, n, final_npc)); prot++;
  std::copy(Xtilde.data(), Xtilde.data() + n * final_npc, REAL(X_fit));

  // B. Create U and D2 subsets (for ridge_call)
  // We need to pass these as SEXP to match the signature: ridge_call(Y, U, D2, lambda)
  SEXP U_subset = PROTECT(Rf_allocMatrix(REALSXP, n, final_npc)); prot++;
  SEXP D2_subset = PROTECT(Rf_allocVector(REALSXP, final_npc)); prot++;
  
  MatrixXd U_sub_mat = U_full.leftCols(final_npc);
  VectorXd d_sub_vec = d_full.head(final_npc);
  VectorXd D2_sub_vec = d_sub_vec.cwiseProduct(d_sub_vec);  // Square the eigenvalues
  
  std::copy(U_sub_mat.data(), U_sub_mat.data() + n * final_npc, REAL(U_subset));
  std::copy(D2_sub_vec.data(), D2_sub_vec.data() + final_npc, REAL(D2_subset));

  // C. Create E_Nn (Eigenvectors for penalty)
  MatrixXd E_Nn = V_full.leftCols(final_npc);
  SEXP ENn_fit = PROTECT(Rf_allocMatrix(REALSXP, Rf_nrows(V_full_sexp), final_npc)); prot++;
  std::copy(E_Nn.data(), E_Nn.data() + Rf_nrows(V_full_sexp) * final_npc, REAL(ENn_fit));

  // 5. Prepare Y (Handle Centering)
  Map<const VectorXd> Y_raw(REAL(Y_), n);
  SEXP Y_fit = PROTECT(Rf_allocVector(REALSXP, n)); prot++;
  double ymean = 0.0;
  
  if (center) {
      ymean = Y_raw.mean();
      for (int i = 0; i < n; ++i) {
          REAL(Y_fit)[i] = Y_raw[i] - ymean;
      }
  } else {
      // If no centering, use raw Y
      std::copy(Y_raw.data(), Y_raw.data() + n, REAL(Y_fit));
      ymean = 0.0;
  }

  // Prepare lambda SEXP
  SEXP lam_sexp = PROTECT(Rf_allocVector(REALSXP, 1)); prot++;
  REAL(lam_sexp)[0] = lambda;

  // 6. Fit Model
  
  // A. Initialize alpha using Ridge Regression
  SEXP alpha0 = PROTECT(ridge_call(Y_fit, U_subset, D2_subset, lam_sexp)); prot++;
  Rprintf("Initial alpha from ridge_call:\n");
  for (int i = 0; i < final_npc; ++i) {
      Rprintf("%f ", REAL(alpha0)[i]);
  }
  Rprintf("\n");


  // B. Run PC-GHAL Optimization
  // Note: pcghal_call still uses X_fit (constructed from U*d) based on your original code
  SEXP res_opt = PROTECT(pcghal_call(Y_fit, X_fit, ENn_fit, alpha0,
                                     max_iter_, tol_, step_factor_, verbose_, crit_)); prot++;

  // 7. Generate Predictions (Optional)
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
    if (!Rf_isReal(alpha_out) || Rf_length(alpha_out) != final_npc)
      Rf_error("Invalid alpha output from optimizer.");
    
    Map<const VectorXd> alpha_hat(REAL(alpha_out), final_npc);

    // Transform coefficients: v = U * D^-1 * alpha
    // (Assuming pchal_des decomposition aligns with this reconstruction)
    VectorXd d_inv = d_sub_vec.cwiseInverse();
    VectorXd v = U_sub_mat * (d_inv.asDiagonal() * alpha_hat);

    // Compute predictions
    VectorXd preds = Ktest * v;
    
    // Add back mean (Conditionally)
    if (center) {
        preds.array() += ymean;
    }

    // Allocate output
    predictions_out = PROTECT(Rf_allocVector(REALSXP, m_pred)); nprot_pred++;
    std::copy(preds.data(), preds.data() + m_pred, REAL(predictions_out));

    UNPROTECT(nprot_pred);
  }

  // 8. Construct Output List
  const int n_out = Rf_isNull(predict_) ? 2 : 3;
  SEXP out_final = PROTECT(Rf_allocVector(VECSXP, n_out)); prot++;
  
  SET_VECTOR_ELT(out_final, 0, res_opt);
  SET_VECTOR_ELT(out_final, 1, lam_sexp);
  
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