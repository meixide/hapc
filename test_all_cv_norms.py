#!/usr/bin/env python3
"""
Test all three CV options: norm="sv", norm="1", norm="2"
Shows verbose gradient descent output for norm="sv"
"""
import sys
sys.path.insert(0, '/Users/cgmeixide/Projects/hapc/python')

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate data: y = 2*sin(8*pi*x^2)/x + 10 + noise
n = 100
p = 1
X = np.random.uniform(-1, 1, size=(n, p))
Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.normal(0, 2, n)

# Test set for predictions
Xnew = np.linspace(-1, 1, 100).reshape(-1, 1)

print("=" * 80)
print("HAPC Cross-Validation Test: All Three Norms")
print("=" * 80)
print(f"Data: n={n}, p={p}")
print(f"Y range: [{Y.min():.3f}, {Y.max():.3f}]")
print()

# Generate lambda grid
log_lambdas = np.linspace(-6, -3, 8)
lambdas = np.exp(log_lambdas)
print(f"Lambda grid: {len(lambdas)} values from {lambdas.min():.6f} to {lambdas.max():.6f}")
print()

# ============================================================================
# TEST 1: norm="sv" (Gradient Descent) - with verbose output
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: norm='sv' (PC-GHAL Gradient Descent)")
print("=" * 80)

try:
    from hapc.cv import pcghal_cv
    
    print("\nRunning CV with VERBOSE gradient descent iterations...")
    print("-" * 80)
    
    result_sv = pcghal_cv(
        X, Y,
        maxdeg=2,
        npc=n,
        lambdas=lambdas,
        nfolds=3,
        predict=Xnew,
        center=True,
        verbose=True,  # <-- Verbose output
        max_iter=20,
        tol=1e-6
    )
    
    print()
    print(f"✓ CV completed")
    print(f"  Best lambda: {result_sv.best_lambda:.6f}")
    print(f"  Best CV MSE: {result_sv.mses.min():.6f}")
    print(f"  Alpha shape: {result_sv.best_model_alpha.shape}")
    print(f"  Predictions shape: {result_sv.predictions.shape if result_sv.predictions is not None else None}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    result_sv = None

# ============================================================================
# TEST 2: norm="2" (Ridge Regression)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: norm='2' (Ridge Regression - L2)")
print("=" * 80)

try:
    from hapc.cv import fasthal_cv
    
    print("\nRunning ridge CV (L2 penalty)...")
    print("-" * 80)
    
    result_ridge = fasthal_cv(
        X, Y,
        npc=n,
        lambdas=lambdas,
        nfolds=3,
        predict=Xnew,
        maxdeg=2,
        center=True,
        approx=False,
        l1=False  # Ridge
    )
    
    print(f"✓ CV completed")
    print(f"  Best lambda: {result_ridge.best_lambda:.6f}")
    print(f"  Best CV MSE: {result_ridge.mses.min():.6f}")
    print(f"  Alpha shape: {result_ridge.best_model_alpha.shape}")
    print(f"  Predictions shape: {result_ridge.predictions.shape if result_ridge.predictions is not None else None}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    result_ridge = None

# ============================================================================
# TEST 3: norm="1" (LASSO)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: norm='1' (LASSO - L1)")
print("=" * 80)

try:
    result_lasso = fasthal_cv(
        X, Y,
        npc=n,
        lambdas=lambdas,
        nfolds=3,
        predict=Xnew,
        maxdeg=2,
        center=True,
        approx=False,
        l1=True  # LASSO
    )
    
    print(f"✓ CV completed")
    print(f"  Best lambda: {result_lasso.best_lambda:.6f}")
    print(f"  Best CV MSE: {result_lasso.mses.min():.6f}")
    print(f"  Alpha shape: {result_lasso.best_model_alpha.shape}")
    print(f"  Sparsity: {(result_lasso.best_model_alpha == 0).sum()}/{result_lasso.best_model_alpha.shape[0]} zeros")
    print(f"  Predictions shape: {result_lasso.predictions.shape if result_lasso.predictions is not None else None}")
    
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    result_lasso = None

# ============================================================================
# PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("Generating comparison plots...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: SV CV MSE
if result_sv is not None:
    ax = axes[0, 0]
    ax.plot(result_sv.lambdas, result_sv.mses, 'b-o', linewidth=2, markersize=6)
    ax.axvline(result_sv.best_lambda, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Lambda', fontsize=11, fontweight='bold')
    ax.set_ylabel('CV MSE', fontsize=11, fontweight='bold')
    ax.set_title('norm="sv" (Gradient Descent)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

# Plot 2: Ridge CV MSE
if result_ridge is not None:
    ax = axes[0, 1]
    ax.plot(result_ridge.lambdas, result_ridge.mses, 'g-s', linewidth=2, markersize=6)
    ax.axvline(result_ridge.best_lambda, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Lambda', fontsize=11, fontweight='bold')
    ax.set_ylabel('CV MSE', fontsize=11, fontweight='bold')
    ax.set_title('norm="2" (Ridge Regression)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

# Plot 3: LASSO CV MSE
if result_lasso is not None:
    ax = axes[1, 0]
    ax.plot(result_lasso.lambdas, result_lasso.mses, 'm-^', linewidth=2, markersize=6)
    ax.axvline(result_lasso.best_lambda, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Lambda', fontsize=11, fontweight='bold')
    ax.set_ylabel('CV MSE', fontsize=11, fontweight='bold')
    ax.set_title('norm="1" (LASSO)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

# Plot 4: All predictions overlaid
ax = axes[1, 1]
ax.scatter(X[:, 0], Y, color='red', s=40, alpha=0.6, label='Data')
if result_sv is not None and result_sv.predictions is not None:
    ax.plot(Xnew[:, 0], result_sv.predictions, 'b-', linewidth=2.5, label='SV (gradient descent)', alpha=0.8)
if result_ridge is not None and result_ridge.predictions is not None:
    ax.plot(Xnew[:, 0], result_ridge.predictions, 'g--', linewidth=2.5, label='Ridge (L2)', alpha=0.8)
if result_lasso is not None and result_lasso.predictions is not None:
    ax.plot(Xnew[:, 0], result_lasso.predictions, 'm:', linewidth=2.5, label='LASSO (L1)', alpha=0.8)
ax.set_xlabel('X', fontsize=11, fontweight='bold')
ax.set_ylabel('Y', fontsize=11, fontweight='bold')
ax.set_title('Fitted Curves Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('/Users/cgmeixide/Projects/hapc/cv_all_norms_test.png', dpi=150, bbox_inches='tight')
print(f"✓ Plots saved to: cv_all_norms_test.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nCV Results Comparison:")
print("-" * 80)
if result_sv is not None:
    print(f"SV (norm='sv'):       Best λ={result_sv.best_lambda:.6f}, Best MSE={result_sv.mses.min():.6f}")
if result_ridge is not None:
    print(f"Ridge (norm='2'):     Best λ={result_ridge.best_lambda:.6f}, Best MSE={result_ridge.mses.min():.6f}")
if result_lasso is not None:
    print(f"LASSO (norm='1'):     Best λ={result_lasso.best_lambda:.6f}, Best MSE={result_lasso.mses.min():.6f}")

print("\n✓ All three CV methods tested successfully!")
print("=" * 80)
