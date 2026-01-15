"""Manual testing script for HAPC package."""

import numpy as np
from hapc.single import single_pcghal
from hapc.cv import pcghal_cv
from hapc.core import pchal_design, mkernel, kernel_cross

def test_basic():
    """Test basic functionality."""
    print("=" * 70)
    print("TEST 1: Basic PC-HAL Design")
    print("=" * 70)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    
    des = pchal_design(X, maxdeg=2, npc=5, center=True)
    print(f"✓ Design matrix shapes:")
    print(f"  H: {des.H.shape}")
    print(f"  U: {des.U.shape}")
    print(f"  d: {des.d.shape}")
    print(f"  V: {des.V.shape}\n")

def test_kernel():
    """Test kernel computation."""
    print("=" * 70)
    print("TEST 2: Kernel Computation")
    print("=" * 70)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    
    K = mkernel(X, m=2, center=True)
    print(f"✓ Kernel matrix:")
    print(f"  Shape: {K.shape}")
    print(f"  Symmetric: {np.allclose(K, K.T)}")
    print(f"  Mean: {K.mean():.6f}")
    print(f"  Std: {K.std():.6f}\n")

def test_single_fit():
    """Test single lambda fit."""
    print("=" * 70)
    print("TEST 3: Single Lambda Fit")
    print("=" * 70)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    
    result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01,
                          max_iter=100, verbose=False)
    print(f"✓ Single fit results:")
    print(f"  Final risk: {result.optimizer_output.risk:.6f}")
    print(f"  Iterations: {result.optimizer_output.iter}")
    print(f"  Alpha shape: {result.optimizer_output.alpha.shape}")
    print(f"  Beta shape: {result.optimizer_output.beta.shape}\n")

def test_single_with_predictions():
    """Test single fit with predictions."""
    print("=" * 70)
    print("TEST 4: Single Fit with Predictions")
    print("=" * 70)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    X_test = np.random.randn(20, p)
    
    result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01,
                          predict=X_test)
    print(f"✓ Predictions:")
    print(f"  Predictions shape: {result.predictions.shape}")
    print(f"  Mean prediction: {result.predictions.mean():.6f}")
    print(f"  Std prediction: {result.predictions.std():.6f}\n")

def test_cv():
    """Test cross-validation."""
    print("=" * 70)
    print("TEST 5: Cross-Validation")
    print("=" * 70)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    
    lambdas = np.logspace(-4, 0, 10)
    cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, 
                         nfolds=5, verbose=False)
    
    print(f"✓ CV results:")
    print(f"  Best lambda: {cv_result.best_lambda:.6f}")
    print(f"  Best MSE: {cv_result.mses.min():.6f}")
    print(f"  All MSEs: {cv_result.mses}")
    print(f"  Best model risk: {cv_result.best_model.risk:.6f}\n")

def test_cv_with_predictions():
    """Test CV with predictions."""
    print("=" * 70)
    print("TEST 6: CV with Predictions")
    print("=" * 70)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    X_test = np.random.randn(20, p)
    
    lambdas = np.logspace(-4, 0, 5)
    cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, 
                         nfolds=3, predict=X_test)
    
    print(f"✓ CV predictions:")
    print(f"  Predictions shape: {cv_result.predictions.shape}")
    print(f"  Mean prediction: {cv_result.predictions.mean():.6f}")
    print(f"  Best lambda: {cv_result.best_lambda:.6f}\n")

if __name__ == "__main__":
    try:
        test_basic()
        test_kernel()
        test_single_fit()
        test_single_with_predictions()
        test_cv()
        test_cv_with_predictions()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
