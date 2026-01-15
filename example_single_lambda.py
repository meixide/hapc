"""Example: Single lambda fit with L1 and L2 penalties."""

import numpy as np
import matplotlib.pyplot as plt
from hapc.single import single_lambda_fit

print("=" * 70)
print("Single Lambda Fit Example")
print("=" * 70)

# Generate synthetic data
np.random.seed(42)
n = 100
p = 1
X = np.random.uniform(-1, 1, (n, p))
Y = 2 * np.sin(8 * np.pi * (X[:, 0]**2)) / X[:, 0] + 10 + np.random.normal(0, 2, n)

Xnew = np.linspace(-1, 1, 100).reshape(-1, 1)

print(f"\nData: n={n}, p={p}")

# Test 1: L2 (Ridge) fit
print("\n" + "=" * 70)
print("Test 1: L2 Penalty (Ridge)")
print("=" * 70)

result_l2 = single_lambda_fit(
    X, Y,
    maxdeg=2,
    npc=n,
    single_lambda=0.5,
    predict=Xnew,
    center=True,
    l1=False
)

print(f"Lambda: {result_l2.lambda_}")
print(f"Alpha shape: {result_l2.alpha.shape}")
print(f"Predictions shape: {result_l2.predictions.shape}")

# Test 2: L1 (LASSO) fit
print("\n" + "=" * 70)
print("Test 2: L1 Penalty (LASSO)")
print("=" * 70)

result_l1 = single_lambda_fit(
    X, Y,
    maxdeg=2,
    npc=n,
    single_lambda=0.5,
    predict=Xnew,
    center=True,
    l1=True
)

print(f"Lambda: {result_l1.lambda_}")
print(f"Alpha shape: {result_l1.alpha.shape}")
print(f"Predictions shape: {result_l1.predictions.shape}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# L2 fit
axes[0].scatter(X, Y, color='red', alpha=0.5, s=20, label='Training data')
axes[0].plot(Xnew, result_l2.predictions, 'b-', linewidth=3, label='L2 (Ridge)')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('Y', fontsize=12)
axes[0].set_title('L2 Penalty (Ridge) Fit', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# L1 fit
axes[1].scatter(X, Y, color='red', alpha=0.5, s=20, label='Training data')
axes[1].plot(Xnew, result_l1.predictions, 'g-', linewidth=3, label='L1 (LASSO)')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Y', fontsize=12)
axes[1].set_title('L1 Penalty (LASSO) Fit', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/cgmeixide/Projects/hapc/single_lambda_example.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plots saved to single_lambda_example.png")
plt.show()

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Training data: n={n}, p={p}")
print(f"Test points: {len(Xnew)}")
print(f"Lambda: 0.5")
print(f"Max degree: 2")
print(f"NPCs: {n}")
print("=" * 70)
