// Python-friendly CV implementation (no R dependencies)
#include "hapc_core.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <Eigen/Eigenvalues>

using Eigen::MatrixXd;
using Eigen::VectorXd;

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

// Python-friendly CV function (pure C++, no R dependencies)
// Parameter order matches R fasthal_cv_call: X, Y, npc, lambdas, nfolds, predict, maxdeg, center, approx, l1
FastCVOutput fasthal_cv_python(const MatrixXd& X, const VectorXd& Y, int npc,
                                const std::vector<double>& lambdas, int nfolds,
                                const MatrixXd& predict, int maxdeg,
                                bool center, bool approx, bool l1) {
    const int n = X.rows();
    const int p = X.cols();
    const int L = lambdas.size();
    
    // Adjust npc based on centering
    if (center) {
        if (npc >= n) npc = n - 1;
    } else {
        if (npc > n) npc = n;
    }
    
    // Compute kernel matrix
    MatrixXd K = mkernel_call(X, maxdeg, center);
    
    // Eigendecomposition
    MatrixXd U(n, npc);
    VectorXd D2(npc);
    
    if (approx) {
        power_iteration_top_k(K, npc, U, D2);
    } else {
        Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(K);
        if (eigensolver.info() != Eigen::Success) {
            throw std::runtime_error("Eigendecomposition failed");
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
    
    // Cross-validation loop
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
            const int ntest = (int)test_idx.size();
            
            if (ntrain == 0 || ntest == 0) {
                fold_error(i - 1, j) = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            
            MatrixXd Xtest(ntest, npc);
            MatrixXd Utrain(ntrain, npc);
            VectorXd Ytrain(ntrain), Ytest(ntest);
            
            for (int ii = 0; ii < ntrain; ++ii) {
                Utrain.row(ii) = U.row(train_idx[ii]);
                Ytrain[ii] = Y[train_idx[ii]];
            }
            for (int ii = 0; ii < ntest; ++ii) {
                Xtest.row(ii) = Xtilde.row(test_idx[ii]);
                Ytest[ii] = Y[test_idx[ii]];
            }
            
            // Center Ytrain if needed
            double ymean = 0.0;
            if (center) {
                ymean = Ytrain.mean();
                Ytrain.array() -= ymean;
            }
            
            // Solve for alpha
            VectorXd alpha_hat;
            if (l1) {
                alpha_hat = fast_pchal_call(Utrain, D2, Ytrain, lambda);
            } else {
                alpha_hat = ridge_call(Ytrain, Utrain, D2, lambda);
            }
            
            VectorXd y_pred = Xtest * alpha_hat;
            if (center) {
                y_pred.array() += ymean;
            }
            
            double mse = (Ytest - y_pred).squaredNorm() / (double)ntest;
            fold_error(i - 1, j) = mse;
        }
    }
    
    // Select best lambda
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
        mses[j] = (cnt > 0) ? (sum / cnt) : std::numeric_limits<double>::quiet_NaN();
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
    
    // Refit on full data
    double ymean_full = 0.0;
    VectorXd Y_centered = Y;
    if (center) {
        ymean_full = Y.mean();
        Y_centered = Y.array() - ymean_full;
    }
    
    VectorXd best_alpha;
    if (l1) {
        best_alpha = fast_pchal_call(U, D2, Y_centered, best_lambda);
    } else {
        best_alpha = ridge_call(Y_centered, U, D2, best_lambda);
    }
    
    // Predictions
    VectorXd predictions_out = VectorXd::Zero(predict.rows());
    if (predict.rows() > 0) {
        MatrixXd Ktest = kernel_cross_call(X, predict, maxdeg, center);
        MatrixXd D2inv_sqrt = D2.cwiseSqrt().cwiseInverse().asDiagonal();
        predictions_out = Ktest * U * D2inv_sqrt * best_alpha;
        if (center) {
            predictions_out.array() += ymean_full;
        }
    }
    
    // Build output
    std::vector<double> mses_vec(L);
    std::vector<double> lambdas_vec(L);
    for (int j = 0; j < L; ++j) {
        mses_vec[j] = mses[j];
        lambdas_vec[j] = lambdas[j];
    }
    
    return FastCVOutput{mses_vec, lambdas_vec, best_lambda, best_alpha, predictions_out};
}
