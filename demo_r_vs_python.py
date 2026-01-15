"""Demo comparing R and Python CV results."""

import numpy as np
import matplotlib.pyplot as plt
from hapc.cv import pcghal_cv

# Replicate R example
np.random.seed(42)
n = 100
p = 1
X = np.random.uniform(-1, 1, (n, p))
Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.normal(0, 2, n)

Xnew = np.linspace(-1, 1, 100).reshape(-1, 1)

print("=" * 70)
print("Python CV (matches R cv.hapc)")
print("=" * 70)

# Lambda grid matching R
lambdas = np.exp(np.linspace(-6, 0, 20))

rescv = pcghal_cv(
    X, Y,
    maxdeg=2,
    npc=n,
    lambdas=lambdas,
    nfolds=5,
    verbose=False,
    predict=Xnew,
    center=True
)

print(f"Best lambda: {rescv.best_lambda:.6f}")
print(f"Best MSE: {rescv.mses.min():.6f}")

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Lambda vs MSE
axes[0].plot(rescv.lambdas, rescv.mses, 'o-', linewidth=2, markersize=6)
axes[0].axvline(rescv.best_lambda, color='red', linestyle='--', 
                label=f'Best: {rescv.best_lambda:.6f}')
axes[0].set_xlabel('Lambda', fontsize=12)
axes[0].set_ylabel('MSE', fontsize=12)
axes[0].set_title('Cross-Validation: Lambda vs MSE', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_xscale('log')

# Plot 2: Data and predictions
axes[1].scatter(X, Y, color='red', alpha=0.5, s=20, label='Training data')
axes[1].plot(Xnew, rescv.predictions, 'b-', linewidth=3, label='Predictions')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Y', fontsize=12)
axes[1].set_title('Data and Fitted Values', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/cgmeixide/Projects/hapc/cv_demo_output.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to cv_demo_output.png")
plt.show()

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Training data: n={n}, p={p}")
print(f"Test points: {len(Xnew)}")
print(f"Lambda grid size: {len(lambdas)}")
print(f"CV folds: 5")
print(f"Best lambda: {rescv.best_lambda:.6f}")
print(f"Best CV MSE: {rescv.mses.min():.6f}")
print("=" * 70)
