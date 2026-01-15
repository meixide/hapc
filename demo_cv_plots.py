#!/usr/bin/env python3
"""
Reproduce the R cross-validation demo in Python.
Shows:
1. CV MSE vs lambda plot
2. Original data with fitted curve overlay
"""
import sys
sys.path.insert(0, '/Users/cgmeixide/Projects/hapc/python')

import numpy as np
import matplotlib.pyplot as plt
from hapc.cv import pcghal_cv

# Set seed for reproducibility
np.random.seed(42)

# Generate data: y = 2*sin(8*pi*x^2)/x + 10 + noise
n = 1000
p = 1
X = np.random.uniform(-1, 1, size=(n, p))
Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.normal(0, 2, n)

# Test set for predictions
Xnew = np.linspace(-1, 1, 100).reshape(-1, 1)

print("=" * 70)
print("CV Demo: Python vs R")
print("=" * 70)
print(f"Data shape: {X.shape}")
print(f"Response range: [{Y.min():.3f}, {Y.max():.3f}]")
print(f"Test set shape: {Xnew.shape}")
print()

# Generate lambda grid (matching R: log_lambda_min=-6, log_lambda_max=-3)
log_lambdas = np.linspace(-6, -2, 10)
lambdas = np.exp(log_lambdas)
print(f"Lambda grid ({len(lambdas)} values):")
print(f"  Range: [{lambdas.min():.6f}, {lambdas.max():.6f}]")
print(f"  Log range: [{np.log(lambdas.min()):.3f}, {np.log(lambdas.max()):.3f}]")
print()

# Run cross-validation (norm="2" for ridge regression)
print("Running CV with ridge regression (norm='2')...")
print("-" * 70)

# For ridge regression CV, we use manual CV with single_lambda_fit
from hapc.single import single_lambda_fit
from sklearn.model_selection import KFold

# Manual CV loop for ridge regression
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mses = np.zeros((5, len(lambdas)))

fold_idx = 0
for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    for j, lam in enumerate(lambdas):
        result = single_lambda_fit(X_train, Y_train, maxdeg=2, npc=n, 
                                   single_lambda=lam, center=True, l1=False)
        
        # Make predictions on test set
        from hapc.core import kernel_cross, mkernel
        K_test = kernel_cross(X_train, X_test, maxdeg=2, center=True)
        
        # Need eigenvalues for proper prediction
        K = mkernel(X_train, maxdeg=2, center=True)
        evals, evecs = np.linalg.eigh(K)
        idx = np.argsort(-evals)[:n]
        D2 = evals[idx]
        
        # Predictions
        from hapc.core import pchal_design
        des = pchal_design(X_train, maxdeg=2, npc=n, center=True)
        final_npc = des.d.shape[0]
        
        evals, evecs = np.linalg.eigh(K)
        idx = np.argsort(-evals)[:final_npc]
        U = evecs[:, idx]
        D = np.sqrt(evals[idx])
        D2 = evals[idx]
        
        D_inv = np.diag(1.0 / (D + 1e-12))
        y_pred = K_test @ U @ D_inv @ result.alpha
        
        # MSE
        mses[fold_idx, j] = np.mean((Y_test - y_pred) ** 2)
    
    fold_idx += 1

# Average CV MSE
mean_mses = np.mean(mses, axis=0)
best_idx = np.argmin(mean_mses)
best_lambda = lambdas[best_idx]

# Refit on full data with best lambda
result_final = single_lambda_fit(X, Y, maxdeg=2, npc=n, 
                                single_lambda=best_lambda, center=True, l1=False)

# Predictions on Xnew
K_pred = kernel_cross(X, Xnew, maxdeg=2, center=True)
K = mkernel(X, maxdeg=2, center=True)
evals, evecs = np.linalg.eigh(K)
des = pchal_design(X, maxdeg=2, npc=n, center=True)
final_npc = des.d.shape[0]
idx = np.argsort(-evals)[:final_npc]
U = evecs[:, idx]
D = np.sqrt(evals[idx])
D_inv = np.diag(1.0 / (D + 1e-12))
y_pred_new = K_pred @ U @ D_inv @ result_final.alpha

# Create result-like object
class CVResult:
    def __init__(self, mses, lambdas, best_lambda, best_alpha, predictions):
        self.mses = mses
        self.lambdas = lambdas
        self.best_lambda = best_lambda
        self.best_model_alpha = best_alpha
        self.predictions = predictions

rescv = CVResult(mean_mses, lambdas, best_lambda, result_final.alpha, y_pred_new)

print(f"✓ CV completed")
print(f"  MSE range: [{rescv.mses.min():.6f}, {rescv.mses.max():.6f}]")
print(f"  Best lambda: {rescv.best_lambda:.6f} (log: {np.log(rescv.best_lambda):.3f})")
print(f"  Best MSE: {rescv.mses.min():.6f}")
print(f"  Alpha shape: {rescv.best_model_alpha.shape}")
print(f"  Predictions shape: {rescv.predictions.shape if rescv.predictions is not None else None}")
print()

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: CV MSE vs Lambda
ax1 = axes[0]
ax1.plot(rescv.lambdas, rescv.mses, 'b-o', linewidth=2, markersize=6)
ax1.axvline(rescv.best_lambda, color='r', linestyle='--', linewidth=2, label=f'Best λ={rescv.best_lambda:.6f}')
ax1.set_xlabel('Lambda (Regularization)', fontsize=12, fontweight='bold')
ax1.set_ylabel('CV MSE', fontsize=12, fontweight='bold')
ax1.set_title('Cross-Validation: MSE vs Lambda\n(Ridge Regression, 5-fold CV)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xscale('log')

# Plot 2: Data with fitted curve
ax2 = axes[1]
ax2.scatter(X[:, 0], Y, color='red', s=50, alpha=0.6, label='Training data')
ax2.plot(Xnew[:, 0], rescv.predictions, 'b-', linewidth=3, label='CV fitted curve')
ax2.set_xlabel('X', fontsize=12, fontweight='bold')
ax2.set_ylabel('Y', fontsize=12, fontweight='bold')
ax2.set_title('Data Fit with Best Lambda\n(True function: 2*sin(8π*x²)/x + 10)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/Users/cgmeixide/Projects/hapc/cv_demo_plots.png', dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to: cv_demo_plots.png")

plt.show()

# Print summary statistics
print()
print("=" * 70)
print("Summary Statistics")
print("=" * 70)
print(f"Training data:")
print(f"  n = {n}, p = {p}")
print(f"  Y mean: {Y.mean():.3f}, std: {Y.std():.3f}")
print()
print(f"CV Results:")
print(f"  Folds: 5")
print(f"  Lambdas tested: {len(lambdas)}")
print(f"  Best lambda: {rescv.best_lambda:.6f}")
print(f"  Best CV MSE: {rescv.mses.min():.6f}")
print(f"  MSE std across lambdas: {rescv.mses.std():.6f}")
print()
print(f"Fitted model:")
print(f"  Alpha coefficients: {rescv.best_model_alpha.shape[0]} values")
print(f"  Mean|alpha|: {np.abs(rescv.best_model_alpha).mean():.6f}")
print(f"  Max|alpha|: {np.abs(rescv.best_model_alpha).max():.6f}")
print()
print(f"Predictions on test set:")
print(f"  Range: [{rescv.predictions.min():.3f}, {rescv.predictions.max():.3f}]")
print(f"  Mean: {rescv.predictions.mean():.3f}")
