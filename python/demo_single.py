"""Demo of single_pcghal high-level API."""

import numpy as np
from hapc.single import single_pcghal
from hapc.cv import pcghal_cv

np.random.seed(42)
n, p = 150, 5
X = np.random.randn(n, p)
beta_true = np.array([1.0, 0.5, 0.2, 0.0, 0.0])
Y = X @ beta_true + np.random.randn(n) * 0.1

print("=" * 60)
print("Demo: Single Lambda Fit")
print("=" * 60)

# Single fit
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01,
                       max_iter=100, verbose=False)
print(f"Risk at convergence: {result.optimizer_output.risk:.6f}")
print(f"Iterations: {result.optimizer_output.iter}")

# With predictions
X_test = np.random.randn(20, p)
result_pred = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01,
                            predict=X_test)
print(f"Predictions shape: {result_pred.predictions.shape}")

print("\n" + "=" * 60)
print("Demo: Cross-Validation")
print("=" * 60)

# CV fit
lambdas = np.logspace(-4, 0, 10)
cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, nfolds=5,
                      verbose=False)
print(f"Best lambda: {cv_result.best_lambda:.6f}")
print(f"Best MSE: {cv_result.best_lambda:.6f}")
print(f"All MSEs: {cv_result.mses}")
