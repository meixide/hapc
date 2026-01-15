// Cross-validation with PC-GHAL optimizer (C++ wrapper for Python)
#include "hapc_core.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

CVOutput pcghal_cv_fit(const MatrixXd& X, const VectorXd& Y,
                       int maxdeg, int npc, const std::vector<double>& lambdas,
                       int nfolds, const MatrixXd& predict_data,
                       int max_iter, double tol, double step_factor,
                       bool verbose, const std::string& crit,
                       bool center, bool approx) {
  
  const int n = X.rows();
  const int p = X.cols();
  const int L = lambdas.size();
  
  if (Y.size() != n) throw std::runtime_error("Dimension mismatch: Y and X");
  if (L == 0) throw std::runtime_error("lambdas vector is empty");
  
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
  
  if (verbose) {
      std::cout << "=" << std::string(58, '=') << std::endl;
      std::cout << "PC-GHAL Cross-Validation" << std::endl;
      std::cout << "=" << std::string(58, '=') << std::endl;
      std::cout << "Data: n=" << n << ", p=" << p << ", nfolds=" << nfolds << std::endl;
      std::cout << "Lambda range: [" << lambdas.front() << ", " << lambdas.back() << "]" << std::endl;
      std::cout << "Number of lambdas: " << L << std::endl;
  }
  
  // Step 1: Generate design matrix once
  DesignOutput des = pchal_des(X, maxdeg, npc, center);
  int final_npc = des.d.size();
  
  if (verbose) {
      std::cout << "Design matrix: " << des.H.rows() << " x " << des.H.cols() << std::endl;
  }
  
  // Step 2: Compute kernel matrix once
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
  
  // Xtilde = U * D
  MatrixXd Xtilde = U * D.asDiagonal();
  
  // ENn: penalty matrix
  MatrixXd ENn = des.V.leftCols(final_npc);
  
  // Step 4: K-fold CV
  std::vector<double> cv_mses(L, 0.0);
  
  // Simple fold indices
  std::vector<int> fold_assignment(n);
  for (int i = 0; i < n; ++i) {
      fold_assignment[i] = i % nfolds;
  }
  
  if (verbose) {
      std::cout << "Running " << nfolds << "-fold cross-validation..." << std::endl;
  }
  
  for (int fold = 0; fold < nfolds; ++fold) {
      if (verbose) {
          std::cout << "  Fold " << (fold + 1) << "/" << nfolds << std::endl;
      }
      
      // Train/test split
      std::vector<int> train_idx, test_idx;
      for (int i = 0; i < n; ++i) {
          if (fold_assignment[i] == fold) {
              test_idx.push_back(i);
          } else {
              train_idx.push_back(i);
          }
      }
      
      int n_train = train_idx.size();
      int n_test = test_idx.size();
      
      // Extract train/test data
      MatrixXd Xtilde_train(n_train, final_npc);
      MatrixXd Xtilde_test(n_test, final_npc);
      MatrixXd U_train(n_train, final_npc);
      VectorXd Y_train(n_train);
      VectorXd Y_test(n_test);
      
      for (int i = 0; i < n_train; ++i) {
          Xtilde_train.row(i) = Xtilde.row(train_idx[i]);
          U_train.row(i) = U.row(train_idx[i]);
          Y_train(i) = Y(train_idx[i]);
      }
      for (int i = 0; i < n_test; ++i) {
          Xtilde_test.row(i) = Xtilde.row(test_idx[i]);
          Y_test(i) = Y(test_idx[i]);
      }
      
      // Center Y on training set
      double ymean_train = Y_train.mean();
      VectorXd Y_train_centered = Y_train.array() - ymean_train;
      
      // Test each lambda
      for (int j = 0; j < L; ++j) {
          double lambda = lambdas[j];
          
          // Initialize with ridge
          VectorXd alpha = ridge_call(Y_train_centered, U_train, D2, lambda);
          
          // Predictions on test set
          VectorXd y_pred = Xtilde_test * alpha;
          if (center) {
              y_pred.array() += ymean_train;
          }
          
          // MSE
          VectorXd residuals = Y_test - y_pred;
          cv_mses[j] += residuals.squaredNorm() / n_test;
      }
  }
  
  // Average MSE across folds
  for (int j = 0; j < L; ++j) {
      cv_mses[j] /= nfolds;
  }
  
  // Find best lambda
  int best_idx = 0;
  double best_mse = cv_mses[0];
  for (int j = 1; j < L; ++j) {
      if (cv_mses[j] < best_mse) {
          best_mse = cv_mses[j];
          best_idx = j;
      }
  }
  double best_lambda = lambdas[best_idx];
  
  if (verbose) {
      std::cout << "Best lambda: " << best_lambda << " (MSE: " << best_mse << ")" << std::endl;
  }
  
  // Step 5: Refit on full data with best lambda
  double ymean = Y.mean();
  VectorXd Y_centered = Y.array() - ymean;
  VectorXd best_alpha = ridge_call(Y_centered, U, D2, best_lambda);
  
  // Step 6: Generate predictions if needed
  VectorXd predictions = VectorXd::Zero(0);
  
  if (predict_data.rows() > 0 && predict_data.cols() == p) {
      int m_pred = predict_data.rows();
      
      MatrixXd Ktest = kernel_cross_call(X, predict_data, maxdeg, center);
      
      // Transform coefficients: v = U * D^-1 * alpha
      VectorXd d_inv = D.cwiseInverse();
      VectorXd v = U * (d_inv.asDiagonal() * best_alpha);
      
      // Compute predictions
      predictions = Ktest * v;
      
      // Add back mean if centered
      if (center) {
          predictions.array() += ymean;
      }
      
      if (verbose) {
          std::cout << "Predictions generated: " << predictions.size() << " points" << std::endl;
      }
  }
  
  // Return result
  CVOutput result;
  result.mses = cv_mses;
  result.lambdas = lambdas;
  result.best_lambda = best_lambda;
  result.best_alpha = best_alpha;
  result.predictions = predictions;
  
  return result;
}
