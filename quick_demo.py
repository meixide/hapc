"""Quick demo of HAPC package."""

import numpy as np
from hapc.single import single_pcghal
from hapc.cv import pcghal_cv

print("=" * 70)
print("HAPC Quick Demo")
print("=" * 70)

# Generate synthetic data
np.random.seed(42)
n, p = 150, 5
X = np.random.randn(n, p)
beta_true = np.array([1.0, 0.5, 0.2, 0.0, 0.0])
Y = X @ beta_true + np.random.randn(n) * 0.1

print(f"\nData: n={n}, p={p}")
print(f"True beta: {beta_true}")

# Test 1: Single fit
print("\n" + "=" * 70)
print("Test 1: Single Lambda Fit")
print("=" * 70)

result = single_pcghal(
    X, Y, 
    maxdeg=2, 
    npc=5, 
    single_lambda=0.01,
    max_iter=100, 
    verbose=False
)

print(f"Final risk: {result.optimizer_output.risk:.6f}")
print(f"Iterations: {result.optimizer_output.iter}")
print(f"Lambda: {result.lambda_}")

# Test 2: Single fit with predictions
print("\n" + "=" * 70)
print("Test 2: Single Fit with Predictions")
print("=" * 70)

X_test = np.random.randn(20, p)
result_pred = single_pcghal(
    X, Y,
    maxdeg=2,
    npc=5,
    single_lambda=0.01,
    predict=X_test
)

print(f"Predictions shape: {result_pred.predictions.shape}")
print(f"Predictions (first 5): {result_pred.predictions[:5]}")
print(f"Mean prediction: {result_pred.predictions.mean():.6f}")

# Test 3: Cross-validation
print("\n" + "=" * 70)
print("Test 3: Cross-Validation")
print("=" * 70)

lambdas = np.logspace(-4, 0, 10)
cv_result = pcghal_cv(
    X, Y,
    maxdeg=2,
    npc=5,
    lambdas=lambdas,
    nfolds=5,
    verbose=False
)

print(f"Best lambda: {cv_result.best_lambda:.6f}")
print(f"Best MSE: {cv_result.mses.min():.6f}")
print(f"All MSEs:")
for lam, mse in zip(lambdas, cv_result.mses):
    print(f"  lambda={lam:.6f}, MSE={mse:.6f}")

# Test 4: CV with predictions
print("\n" + "=" * 70)
print("Test 4: CV with Predictions")
print("=" * 70)

cv_result_pred = pcghal_cv(
    X, Y,
    maxdeg=2,
    npc=5,
    lambdas=lambdas,
    nfolds=5,
    predict=X_test,
    verbose=False
)

print(f"Best lambda: {cv_result_pred.best_lambda:.6f}")
print(f"Predictions shape: {cv_result_pred.predictions.shape}")
print(f"Predictions (first 5): {cv_result_pred.predictions[:5]}")

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
