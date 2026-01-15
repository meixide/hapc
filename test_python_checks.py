"""
Python equivalent of R checks for HAPC package.
Replicates the R test script and generates plots in debug_plots/ folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add python directory to path if needed
sys.path.insert(0, str(Path(__file__).parent / "python"))

from hapc import pchal_design, mkernel
from hapc.single import single_lambda_fit

def is_col_permutation(A, B, tol=0):
    """Check if A and B are column permutations of each other."""
    if A.shape != B.shape:
        return False
    used = np.zeros(B.shape[1], dtype=bool)
    for j in range(A.shape[1]):
        # Find all candidate matches
        matches = []
        for k in range(B.shape[1]):
            if not used[k]:
                if np.allclose(A[:, j], B[:, k], atol=tol):
                    matches.append(k)
        if len(matches) == 0:
            return False
        used[matches[0]] = True
    return True

# Create output directory
output_dir = Path("debug_plots")
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Python HAPC Package Checks")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Setup test data
n = 100
d = 5
X = np.random.uniform(0, 1, size=(n, d))

print(f"\nTest data: n={n}, d={d}")
print(f"X shape: {X.shape}")
print(f"X range: [{X.min():.4f}, {X.max():.4f}]")

# Note: We can't use hal9001 in Python, so we'll skip that part
# and focus on comparing hapc design with SVD of H*H^T

print("\n" + "-" * 60)
print("1. Design matrix generation")
print("-" * 60)

des = pchal_design(X, maxdeg=2, npc=n, center=False)
print(f"Design output shapes:")
print(f"  H: {des.H.shape}")
print(f"  U: {des.U.shape}")
print(f"  d: {des.d.shape}")
print(f"  V: {des.V.shape}")

# SVD of H
H = des.H
U_svd, D_svd, Vt_svd = np.linalg.svd(H, full_matrices=False)
V_svd = Vt_svd.T
D_svd_full = np.zeros(n)
D_svd_full[:len(D_svd)] = D_svd

print(f"\nSVD of H:")
print(f"  U_svd: {U_svd.shape}")
print(f"  D_svd: {D_svd.shape}")
print(f"  V_svd: {V_svd.shape}")

# Plot: des$U[,3] vs U[,3]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(des.U[:, 2], U_svd[:, 2], alpha=0.6)
ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
ax.set_xlabel('des.U[:, 2] (hapc)')
ax.set_ylabel('U_svd[:, 2] (SVD of H)')
ax.set_title('Comparison: hapc U vs SVD U (column 3)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "plot1_U_comparison.png", dpi=150)
print(f"\nSaved plot: {output_dir / 'plot1_U_comparison.png'}")
plt.close()

# Test function
def f1(X, n):
    """Test function matching R code."""
    return (np.sin(np.pi * (X[:, 0] * X[:, 2])) / X[:, 0] + 
            np.sqrt(X[:, 1]) * np.log(X[:, 2]) + 
            np.random.normal(0, 0.05, n))

f = f1
Y = f(X, n)
print(f"\nY shape: {Y.shape}")
print(f"Y range: [{Y.min():.4f}, {Y.max():.4f}]")

# Generate test data
nnew = 100
Xnew = np.random.uniform(0.1, 1, size=(nnew, d))

print("\n" + "-" * 60)
print("2. Kernel matrix computation")
print("-" * 60)

k = mkernel(X, m=2, center=False)
print(f"Kernel matrix shape: {k.shape}")

# Check: max(abs(des$H%*%t(des$H)-k))
HHT = des.H @ des.H.T
diff1 = np.abs(HHT - k)
max_diff1 = np.max(diff1)
print(f"\nMax |H @ H.T - k|: {max_diff1:.2e}")

# Eigendecomposition of kernel
evals_k, evecs_k = np.linalg.eigh(k)
# Sort descending
idx_k = np.argsort(-evals_k)
D2k = evals_k[idx_k]
Uk = evecs_k[:, idx_k]

print(f"\nKernel eigendecomposition:")
print(f"  Uk shape: {Uk.shape}")
print(f"  D2k shape: {D2k.shape}")
print(f"  Top 5 eigenvalues: {D2k[:5]}")

# Plot: Uk[,1] vs des$U[,1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Uk[:, 0], des.U[:, 0], alpha=0.6)
ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
ax.set_xlabel('Uk[:, 0] (kernel eigenvectors)')
ax.set_ylabel('des.U[:, 0] (hapc design)')
ax.set_title('Comparison: Kernel eigenvectors vs hapc U (column 1)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "plot2_kernel_U_comparison.png", dpi=150)
print(f"Saved plot: {output_dir / 'plot2_kernel_U_comparison.png'}")
plt.close()

# Plot: Uk[,1] vs U[,1] (from SVD of H)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Uk[:, 0], U_svd[:, 0], alpha=0.6)
ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
ax.set_xlabel('Uk[:, 0] (kernel eigenvectors)')
ax.set_ylabel('U_svd[:, 0] (SVD of H)')
ax.set_title('Comparison: Kernel eigenvectors vs SVD U (column 1)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "plot3_kernel_svd_U_comparison.png", dpi=150)
print(f"Saved plot: {output_dir / 'plot3_kernel_svd_U_comparison.png'}")
plt.close()

# Check sqrt(D2k) vs D
print(f"\nComparison: sqrt(D2k) vs des.d")
print(f"  sqrt(D2k)[:10]: {np.sqrt(D2k[:10])}")
print(f"  des.d[:10]: {des.d[:10]}")
print(f"  Max difference (first 10): {np.max(np.abs(np.sqrt(D2k[:10]) - des.d[:10])):.2e}")

print("\n" + "-" * 60)
print("3. Ridge regression comparison")
print("-" * 60)

lambda_val = 0.01
n_val = n

# Manual computation
# solve(lambda*n*diag(1,n) + diag(D^2))%*%diag(D)%*%t(U)%*%Y
D_diag = np.diag(des.d)
D2_diag = np.diag(des.d ** 2)
I_n = np.eye(n)

# Method 1: solve(lambda*n*I + D^2) @ D @ U.T @ Y
alphahand1 = np.linalg.solve(lambda_val * n_val * I_n + D2_diag, 
                             D_diag @ des.U.T @ Y)

# Method 2: U @ diag(D), then solve
desm = des.U @ D_diag
alphahand2 = np.linalg.solve(lambda_val * n_val * I_n + desm.T @ desm,
                              desm.T @ Y)

print(f"Lambda: {lambda_val}")
print(f"alphahand1 shape: {alphahand1.shape}")
print(f"alphahand2 shape: {alphahand2.shape}")

# Call hapc function (equivalent to R hapc with norm="2")
print("\nCalling hapc function (norm='2', center=False)...")
res = single_lambda_fit(X, Y, maxdeg=2, npc=n, single_lambda=lambda_val,
                        predict=Xnew, center=False, approx=False, l1=False)

print(f"res.alpha shape: {res.alpha.shape}")
print(f"res.alpha[:10]: {res.alpha[:10]}")
if res.predictions is not None:
    print(f"res.predictions shape: {res.predictions.shape}")
    print(f"res.predictions[:5]: {res.predictions[:5]}")

# Plot: alphahand vs res$alpha
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: alphahand1 vs res.alpha
axes[0].scatter(alphahand1, res.alpha, alpha=0.6)
axes[0].plot([alphahand1.min(), alphahand1.max()], 
             [alphahand1.min(), alphahand1.max()], 'r--', label='y=x')
axes[0].set_xlabel('alphahand1 (manual computation)')
axes[0].set_ylabel('res.alpha (hapc function)')
axes[0].set_title('Comparison: Manual alpha vs hapc alpha (method 1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: alphahand2 vs res.alpha
axes[1].scatter(alphahand2, res.alpha, alpha=0.6)
axes[1].plot([alphahand2.min(), alphahand2.max()], 
             [alphahand2.min(), alphahand2.max()], 'r--', label='y=x')
axes[1].set_xlabel('alphahand2 (manual computation)')
axes[1].set_ylabel('res.alpha (hapc function)')
axes[1].set_title('Comparison: Manual alpha vs hapc alpha (method 2)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "plot4_alpha_comparison.png", dpi=150)
print(f"Saved plot: {output_dir / 'plot4_alpha_comparison.png'}")
plt.close()

# Print differences
diff_alpha1 = np.abs(alphahand1 - res.alpha)
diff_alpha2 = np.abs(alphahand2 - res.alpha)
print(f"\nMax |alphahand1 - res.alpha|: {np.max(diff_alpha1):.2e}")
print(f"Max |alphahand2 - res.alpha|: {np.max(diff_alpha2):.2e}")
print(f"Mean |alphahand1 - res.alpha|: {np.mean(diff_alpha1):.2e}")
print(f"Mean |alphahand2 - res.alpha|: {np.mean(diff_alpha2):.2e}")

print("\n" + "=" * 60)
print("All checks completed!")
print(f"Plots saved in: {output_dir.absolute()}")
print("=" * 60)
