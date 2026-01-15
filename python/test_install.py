"""Quick test of hapc package installation."""

import numpy as np
from hapc.core import (
    pchal_design, ridge_regression, mkernel, kernel_cross, pcghal
)

# Generate sample data
np.random.seed(42)
n, p = 100, 5
X = np.random.randn(n, p)
Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1

print("Testing HAPC Python package...")
print(f"X shape: {X.shape}, Y shape: {Y.shape}\n")

# Test 1: Design matrix
print("1. Testing pchal_design...")
des = pchal_design(X, maxdeg=2, npc=5, center=True)
print(f"   H shape: {des.H.shape}")
print(f"   U shape: {des.U.shape}")
print(f"   d shape: {des.d.shape}")
print(f"   V shape: {des.V.shape}\n")

# Test 2: Kernel
print("2. Testing mkernel...")
K = mkernel(X, m=2, center=True)
print(f"   Kernel shape: {K.shape}")
print(f"   Kernel[:3, :3]:\n{K[:3, :3]}\n")

# Test 3: Ridge regression
print("3. Testing ridge_regression...")
U = des.U[:, :3]
D2 = des.d[:3] ** 2
beta = ridge_regression(Y, U, D2, lambda_=0.01)
print(f"   Beta shape: {beta.shape}")
print(f"   Beta: {beta}\n")

# Test 4: PC-GHAL optimizer
print("4. Testing pcghal optimizer...")
final_npc = des.d.shape[0]
Xtilde = des.U[:, :final_npc] @ np.diag(des.d[:final_npc])
ENn = des.V[:, :final_npc]
ymean = Y.mean()
Y_centered = Y - ymean

alpha0 = ridge_regression(Y_centered, des.U[:, :final_npc], D2, 0.01)
result = pcghal(Y_centered, Xtilde, ENn, alpha0, max_iter=50, 
                tol=1e-6, verbose=True, crit="grad")
print(f"\n   Alpha shape: {result.alpha.shape}")
print(f"   Final risk: {result.risk:.6f}")
print(f"   Iterations: {result.iter}\n")

# Test 5: Predictions
print("5. Testing kernel_cross for predictions...")
X_test = np.random.randn(10, p)
K_test = kernel_cross(X, X_test, m=2, center=True)
print(f"   Cross-kernel shape: {K_test.shape}")
d_inv = 1.0 / (des.d[:final_npc] + 1e-12)
v = des.U[:, :final_npc] @ (d_inv * result.alpha)
predictions = K_test @ v + ymean
print(f"   Predictions shape: {predictions.shape}")
print(f"   Predictions: {predictions}\n")

print("✓ All tests passed!")
