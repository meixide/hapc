// Single lambda fit with PC-GHAL optimizer (C++ wrapper for Python)
#include "hapc_core.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

SinglePcghalOutput single_pcghal_fit(const MatrixXd& X, const VectorXd& Y, 
                                     int maxdeg, int npc, double single_lambda,
                                     const MatrixXd& predict_data,
                                     int max_iter, double tol, double step_factor,
                                     bool verbose, const std::string& crit,
                                     bool center, bool approx) {
  
  const int n = X.rows();
  const int p = X.cols();
  
  if (Y.size() != n) throw std::runtime_error("Dimension mismatch: Y and X");
  
  // Adjust NPC based on centering
  if (center) {
      if (npc >= n) {
          npc = n - 1;
      }
  } else {
      if (npc > n) {
          npc = n;
      }
  }
  
  // Step 1: Generate design matrix
  DesignOutput des = pchal_des(X, maxdeg, npc, center);
  int final_npc = des.d.size();
  
  if (verbose) {
      std::cout << "Design matrix generated:" << std::endl;
      std::cout << "  H shape: " << des.H.rows() << " x " << des.H.cols() << std::endl;
      std::cout << "  U shape: " << des.U.rows() << " x " << des.U.cols() << std::endl;
      std::cout << "  d size: " << des.d.size() << std::endl;
      std::cout << "  Final npc: " << final_npc << std::endl;
  }
  
  // Step 2: Compute kernel matrix
  MatrixXd K = mkernel_call(X, maxdeg, center);
  
  // Step 3: Eigendecomposition
  Eigen::SelfAdjointEigenSolver<MatrixXd> solver(K);
  VectorXd evals = solver.eigenvalues();
  MatrixXd evecs = solver.eigenvectors();
  
  // Sort descending
  std::vector<int> idx(evals.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&evals](int i, int j) {
      return evals(i) > evals(j);
  });
  
  MatrixXd U = MatrixXd::Zero(n, final_npc);
  VectorXd D = VectorXd::Zero(final_npc);
  VectorXd D2 = VectorXd::Zero(final_npc);
  
  for (int i = 0; i < final_npc; ++i) {
      U.col(i) = evecs.col(idx[i]);
      D(i) = std::sqrt(std::max(0.0, evals(idx[i])));
      D2(i) = evals(idx[i]);
  }
  
  if (verbose) {
      std::cout << "Kernel eigendecomposition:" << std::endl;
      std::cout << "  Top 5 eigenvalues: ";
      for (int i = 0; i < std::min(5, (int)evals.size()); ++i) {
          std::cout << evals(idx[i]) << " ";
      }
      std::cout << std::endl;
  }
  
  // Step 4: Create Xtilde = U * D
  MatrixXd Xtilde = U * D.asDiagonal();
  
  // Step 5: Prepare Y (handle centering)
  VectorXd Y_centered = Y;
  double ymean = 0.0;
  
  if (center) {
      ymean = Y.mean();
      Y_centered.array() -= ymean;
  }
  
  if (verbose) {
      std::cout << "Y preprocessing:" << std::endl;
      std::cout << "  Y mean: " << ymean << std::endl;
      std::cout << "  Lambda: " << single_lambda << std::endl;
      std::cout << "  Ridge regularization: " << (single_lambda * n) << std::endl;
  }
  
  // Step 6: Initialize alpha using ridge regression
  VectorXd alpha0 = ridge_call(Y_centered, U, D2, single_lambda);
  
  if (verbose) {
      std::cout << "Initial alpha (from ridge):" << std::endl;
      std::cout << "  Mean: " << alpha0.mean() << std::endl;
      std::cout << "  Max abs: " << alpha0.cwiseAbs().maxCoeff() << std::endl;
  }
  
  // Step 7: Create ENn (penalty matrix)
  MatrixXd ENn = des.V.leftCols(final_npc);
  
  // Step 8: Run PC-GHAL optimizer
  if (verbose) {
      std::cout << "Calling PC-GHAL optimizer..." << std::endl;
      std::cout << "  max_iter: " << max_iter << std::endl;
      std::cout << "  tol: " << tol << std::endl;
      std::cout << std::endl;
  }
  
  OptimizerOutput opt_out = pcghal_call(Y_centered, Xtilde, ENn, alpha0,
                                        max_iter, tol, step_factor, verbose, crit);
  
  if (verbose) {
      std::cout << std::endl;
      std::cout << "PC-GHAL Optimization Complete:" << std::endl;
      std::cout << "  Final risk: " << opt_out.risk << std::endl;
      std::cout << "  Iterations: " << opt_out.iter << std::endl;
      std::cout << "  Final alpha mean: " << opt_out.alpha.mean() << std::endl;
      std::cout << "  Final alpha max abs: " << opt_out.alpha.cwiseAbs().maxCoeff() << std::endl;
  }
  
  // Step 9: Generate predictions if needed
  VectorXd predictions = VectorXd::Zero(0);
  
  if (predict_data.rows() > 0 && predict_data.cols() == p) {
      int m_pred = predict_data.rows();
      
      MatrixXd Ktest = kernel_cross_call(X, predict_data, maxdeg, center);
      
      // Transform coefficients: v = U * D^-1 * alpha
      VectorXd d_inv = D.cwiseInverse();
      VectorXd v = U * (d_inv.asDiagonal() * opt_out.alpha);
      
      // Compute predictions
      predictions = Ktest * v;
      
      // Add back mean if centered
      if (center) {
          predictions.array() += ymean;
      }
      
      if (verbose) {
          std::cout << "Predictions generated:" << std::endl;
          std::cout << "  Shape: " << predictions.size() << std::endl;
          std::cout << "  Mean: " << predictions.mean() << std::endl;
          std::cout << "  Range: [" << predictions.minCoeff() << ", " 
                    << predictions.maxCoeff() << "]" << std::endl;
      }
  }
  
  // Return result
  SinglePcghalOutput result;
  result.alpha = opt_out.alpha;
  result.predictions = predictions;
  result.lambda_val = single_lambda;
  result.risk = opt_out.risk;
  result.iter = opt_out.iter;
  
  return result;
}
