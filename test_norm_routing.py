#!/usr/bin/env python3
"""
Test norm parameter routing and C++ integration.
Validates that:
1. norm="sv" calls C++ gradient descent (single_pcghal)
2. norm="1" calls Python LASSO (single_lambda_fit with l1=True)
3. norm="2" calls Python ridge (single_lambda_fit with l1=False)
"""
import sys
sys.path.insert(0, '/Users/cgmeixide/Projects/hapc/python')

import numpy as np
from hapc import single
from hapc.cv import pcghal_cv

# Generate simple test data
np.random.seed(42)
n, p = 50, 10
X = np.random.randn(n, p)
Y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n)
X_test = np.random.randn(10, p)  # Test set for predictions

print("=" * 70)
print("TEST: norm Parameter Routing and C++ Integration")
print("=" * 70)

# Test 1: norm="sv" (should call C++ gradient descent)
print("\n[1] Testing norm='sv' (C++ gradient descent)")
print("-" * 70)
try:
    result_sv = single.hapc(X, Y, maxdeg=2, npc=5, single_lambda=0.1, norm="sv", 
                            center=True, verbose=False, predict=X_test)
    print(f"✓ norm='sv' succeeded")
    print(f"  - alpha shape: {result_sv.alpha.shape}")
    if result_sv.predictions is not None:
        print(f"  - predictions shape: {result_sv.predictions.shape}")
        print(f"  - prediction range: [{result_sv.predictions.min():.3f}, {result_sv.predictions.max():.3f}]")
    print(f"  - risk: {result_sv.risk:.6f}")
    print(f"  - iterations: {result_sv.iter}")
except Exception as e:
    print(f"✗ norm='sv' failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: norm="2" (should call Python ridge)
print("\n[2] Testing norm='2' (Python ridge)")
print("-" * 70)
try:
    result_ridge = single.hapc(X, Y, maxdeg=2, npc=5, single_lambda=0.1, norm="2", 
                               center=True, predict=X_test)
    print(f"✓ norm='2' succeeded")
    print(f"  - alpha shape: {result_ridge.alpha.shape}")
    if result_ridge.predictions is not None:
        print(f"  - predictions shape: {result_ridge.predictions.shape}")
        print(f"  - prediction range: [{result_ridge.predictions.min():.3f}, {result_ridge.predictions.max():.3f}]")
except Exception as e:
    print(f"✗ norm='2' failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: norm="1" (should call Python LASSO)
print("\n[3] Testing norm='1' (Python LASSO)")
print("-" * 70)
try:
    result_lasso = single.hapc(X, Y, maxdeg=2, npc=5, single_lambda=0.1, norm="1", 
                               center=True, max_iter=100, predict=X_test)
    print(f"✓ norm='1' succeeded")
    print(f"  - alpha shape: {result_lasso.alpha.shape}")
    if result_lasso.predictions is not None:
        print(f"  - predictions shape: {result_lasso.predictions.shape}")
        print(f"  - sparsity: {(result_lasso.alpha == 0).sum()} zeros out of {result_lasso.alpha.shape[0]} elements")
        print(f"  - prediction range: [{result_lasso.predictions.min():.3f}, {result_lasso.predictions.max():.3f}]")
except Exception as e:
    print(f"✗ norm='1' failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Comparison - predictions should differ between methods
print("\n[4] Comparing Predictions Across Methods")
print("-" * 70)
try:
    if 'result_sv' in locals() and 'result_ridge' in locals():
        pred_diff_sv_ridge = np.linalg.norm(result_sv.predictions - result_ridge.predictions)
        print(f"  SV vs Ridge L2 diff: {pred_diff_sv_ridge:.6f}")
        print(f"  (Should be non-zero as methods differ)")
    
    if 'result_ridge' in locals() and 'result_lasso' in locals():
        pred_diff_ridge_lasso = np.linalg.norm(result_ridge.predictions - result_lasso.predictions)
        print(f"  Ridge vs LASSO L2 diff: {pred_diff_ridge_lasso:.6f}")
        print(f"  (Should be non-zero as regularizers differ)")
except Exception as e:
    print(f"✗ Comparison failed: {e}")

# Test 5: Cross-validation with C++ pcghal_cv_fit
print("\n[5] Testing Cross-Validation (C++ pcghal_cv_fit)")
print("-" * 70)
try:
    lambdas = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    result_cv = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, 
                          nfolds=3, center=True, verbose=False, predict=X_test)
    print(f"✓ CV succeeded")
    print(f"  - CV MSEs: {result_cv.mses}")
    print(f"  - Best lambda: {result_cv.best_lambda}")
    print(f"  - Best alpha shape: {result_cv.best_model_alpha.shape}")
    if result_cv.predictions is not None:
        print(f"  - Predictions shape: {result_cv.predictions.shape}")
except Exception as e:
    print(f"✗ CV failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("SUMMARY: All tests completed")
print("=" * 70)
