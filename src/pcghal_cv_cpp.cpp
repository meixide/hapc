// Cross-validation with PC-GHAL optimizer (C++ wrapper for Python).
//
// This function mirrors R's pchal_cv_call: per fold, initialise α with LASSO
// (or ridge if ini="2"), run projected gradient descent via pcghal_call, and
// score on the held-out fold.  After CV, refit on full data with the same
// (init + PGD) pipeline at the best λ.
//
// Fold assignment uses std::mt19937(12345) + shuffle, identical to R, so the
// partitions match between languages.

#include "hapc_core.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

CVOutput pcghal_cv_fit(const MatrixXd& X, const VectorXd& Y,
                       int maxdeg, int npc, const std::vector<double>& lambdas,
                       int nfolds, const MatrixXd& predict_data,
                       int max_iter, double tol, double step_factor,
                       bool verbose, const std::string& crit,
                       bool center, bool approx, const std::string& ini) {
  
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
  
  // Fold indices: block partition with last fold absorbing the remainder, then
  // shuffled with std::mt19937(12345). Identical scheme to R's pchal_cv_call so
  // R and Python produce the same fold assignments for the same n & nfolds.
  std::vector<int> fold_assignment(n);
  const int fold_size = n / nfolds;
  for (int i = 0; i < n; ++i) fold_assignment[i] = i / fold_size;
  for (int i = fold_size * nfolds; i < n; ++i) fold_assignment[i] = nfolds - 1;
  std::mt19937 rng(12345);
  std::shuffle(fold_assignment.begin(), fold_assignment.end(), rng);
  
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
      
      // Test each lambda — full PGD pipeline: init with LASSO/ridge, then run
      // pcghal_call (projected gradient descent) and use the optimised α.
      MatrixXd ENn_train = ENn;  // V is shared across folds (computed on full data)
      for (int j = 0; j < L; ++j) {
          double lambda = lambdas[j];

          VectorXd alpha0;
          if (ini == "2") {
              alpha0 = ridge_call(Y_train_centered, U_train, D2, lambda);
          } else if (ini == "1") {
              alpha0 = fast_pchal_call(U_train, D2, Y_train_centered, lambda);
          } else {
              throw std::runtime_error("ini must be '1' (LASSO) or '2' (ridge), got: " + ini);
          }

          OptimizerOutput opt = pcghal_call(Y_train_centered, Xtilde_train,
                                            ENn_train, alpha0,
                                            max_iter, tol, step_factor,
                                            /*verbose=*/false, crit);

          VectorXd y_pred = Xtilde_test * opt.alpha;
          if (center) {
              y_pred.array() += ymean_train;
          }

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
  
  // Step 5: Refit on full data with best lambda (init + PGD, same pipeline)
  double ymean = center ? Y.mean() : 0.0;
  VectorXd Y_centered = center ? (Y.array() - ymean).matrix() : Y;

  VectorXd alpha0_full;
  if (ini == "2") {
      alpha0_full = ridge_call(Y_centered, U, D2, best_lambda);
  } else if (ini == "1") {
      alpha0_full = fast_pchal_call(U, D2, Y_centered, best_lambda);
  } else {
      throw std::runtime_error("ini must be '1' (LASSO) or '2' (ridge), got: " + ini);
  }

  OptimizerOutput opt_full = pcghal_call(Y_centered, Xtilde, ENn, alpha0_full,
                                         max_iter, tol, step_factor, verbose, crit);
  VectorXd best_alpha = opt_full.alpha;
  
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
