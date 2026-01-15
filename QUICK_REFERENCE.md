# HAPC Quick Reference - Python/C++ Integration

## Installation & Build

```bash
# Development install with rebuild
cd /Users/cgmeixide/Projects/hapc
pip install -e .

# Manual build (if needed)
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cp hapc_core.cpython-312-darwin.so ../python/hapc/
```

## API Overview

### High-Level Interface

```python
from hapc import hapc

# Single lambda with norm selection
result = hapc(X, Y, maxdeg=2, npc=5, single_lambda=0.1, 
              norm="sv",    # "sv"=gradient descent, "1"=LASSO, "2"=ridge
              predict=X_test, center=True)

print(result.alpha)           # Coefficients
print(result.predictions)     # Test predictions
print(result.risk)            # Final risk (for norm="sv" only)
print(result.iter)            # Iterations (for norm="sv" only)
```

### Cross-Validation

```python
from hapc.cv import pcghal_cv

lambdas = np.logspace(-3, 0, 10)
result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas,
                   nfolds=5, predict=X_test, center=True)

print(result.mses)              # CV MSE for each lambda
print(result.best_lambda)       # Optimal lambda
print(result.best_model_alpha)  # Coefficients from best lambda
print(result.predictions)       # Predictions on test data
```

### Direct C++ Functions

```python
from hapc import single_pcghal
from hapc.cv import pcghal_cv

# Single lambda (equivalent to hapc(..., norm="sv"))
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.1,
                       predict=X_test, verbose=True, 
                       max_iter=100, tol=1e-6)

# Cross-validation (C++ implementation)
result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas,
                   nfolds=5, predict=X_test)
```

## Key Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| X | ndarray (n, p) | - | Input features |
| Y | ndarray (n,) | - | Response variable |
| maxdeg | int | - | Max degree of interactions |
| npc | int | - | Number of principal components |
| single_lambda | float | - | Regularization parameter |
| lambdas | ndarray | - | Array of lambdas for CV |
| norm | str | "sv" | "sv"=gradient descent, "1"=LASSO, "2"=ridge |
| predict | ndarray (m, p) | None | Test data for predictions |
| center | bool | True | Center design matrix |
| nfolds | int | 5 | Number of CV folds |
| max_iter | int | 100 | Max iterations (norm="sv" only) |
| tol | float | 1e-6 | Convergence tolerance |
| verbose | bool | False | Print progress |

## Result Objects

### SinglePcghalResult (norm="sv" & single_pcghal)
```python
result.alpha            # ndarray - Coefficients
result.predictions      # ndarray or None - Test predictions
result.lambda_          # float - Lambda value used
result.risk             # float - Final optimization risk
result.iter             # int - Iterations taken
```

### SingleLambdaResult (norm="1" or "2")
```python
result.alpha            # ndarray - Coefficients
result.predictions      # ndarray or None - Test predictions
result.lambda_          # float - Lambda value used
```

### CVResult (pcghal_cv)
```python
result.mses             # ndarray - CV MSE for each lambda
result.lambdas          # ndarray - Lambda values tested
result.best_lambda      # float - Optimal lambda
result.best_model_alpha # ndarray - Coefficients from best lambda
result.predictions      # ndarray or None - Test predictions
```

## Norm Parameter Behavior

| norm | Solver | Convergence | Sparse | Cost |
|------|--------|-------------|--------|------|
| "sv" | C++ gradient descent | Iterative | No | Slow (iterative) |
| "1" | Python LASSO soft-thresholding | 1 pass | Yes | Fast (closed-form) |
| "2" | Python ridge regression | Closed-form | No | Fast (closed-form) |

## Troubleshooting

### Import Error: "No module named 'hapc_core'"
```python
# Check module location
import sys
print(sys.path)

# Manually add path if needed
sys.path.insert(0, '/Users/cgmeixide/Projects/hapc/python')
from hapc import hapc
```

### Dimension Mismatch Error
```python
# Ensure arrays are 2D for X and 1D for Y
X = np.atleast_2d(X)  # Shape (n, p)
Y = np.atleast_1d(Y)  # Shape (n,)

# Ensure C-contiguous (handled automatically by _ensure_c_contiguous)
X = np.asarray(X, order='C')
```

### Slow Performance on norm="sv"
- Reduce `maxdeg` (currently quadratic in interactions)
- Reduce `npc` (dimensionality)
- Increase `tol` (convergence tolerance, default 1e-6)

### CV Not Finding Good Lambda
- Expand lambda grid: `np.logspace(-4, 1, 20)`
- Increase nfolds for more stable estimates: `nfolds=10`
- Check Y values: large scale might need preprocessing

## Common Workflows

### Quick Model Fit & Predict
```python
from hapc import hapc
import numpy as np

X_train = np.random.randn(100, 10)
Y_train = np.random.randn(100)
X_test = np.random.randn(20, 10)

# Fit with default settings
result = hapc(X_train, Y_train, maxdeg=2, npc=5, single_lambda=0.1, 
              predict=X_test)

print(f"Test predictions: {result.predictions}")
print(f"Prediction MSE: {np.mean((Y_test - result.predictions)**2)}")
```

### Hyperparameter Tuning
```python
from hapc.cv import pcghal_cv

lambdas = np.logspace(-4, 0, 15)
result = pcghal_cv(X_train, Y_train, maxdeg=2, npc=5, lambdas=lambdas,
                   nfolds=5, predict=X_test)

print(f"Best lambda: {result.best_lambda}")
print(f"CV MSEs: {result.mses}")
```

### Comparing Regularizers
```python
from hapc import hapc

# Compare norm types
sv_result = hapc(X, Y, norm="sv", single_lambda=0.1, predict=X_test)
ridge_result = hapc(X, Y, norm="2", single_lambda=0.1, predict=X_test)
lasso_result = hapc(X, Y, norm="1", single_lambda=0.1, predict=X_test)

# Compare predictions
print(f"SV MSE: {np.mean((Y_test - sv_result.predictions)**2)}")
print(f"Ridge MSE: {np.mean((Y_test - ridge_result.predictions)**2)}")
print(f"LASSO MSE: {np.mean((Y_test - lasso_result.predictions)**2)}")
```

### Verbose Output for Debugging
```python
from hapc import single_pcghal

result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.1,
                       verbose=True, max_iter=20)

# Output shows:
# - Design matrix dimensions
# - Kernel eigenvalues
# - Ridge initialization
# - Iteration details (step size, risk, gradient norm)
# - Final convergence status
```

## Architecture Notes

### Python/C++ Split
- **Python Layer**: API, array handling, result packaging
- **C++ Layer**: Algorithms, linear algebra (Eigen), optimization

### Data Flow
```
Python Input Arrays
    ↓
_ensure_c_contiguous() → Verify C-contiguous layout
    ↓
pybind11 conversion → Eigen MatrixXd/VectorXd
    ↓
C++ Algorithm (pchal_des, mkernel, pcghal_call, etc.)
    ↓
Return C++ Struct (SinglePcghalOutput, CVOutput)
    ↓
pybind11 conversion → Python NumPy arrays
    ↓
Python Result NamedTuple
```

### Key C++ Functions
- `pchal_des()` - Design matrix with PCA compression
- `mkernel_call()` - Kernel matrix computation
- `ridge_call()` - Ridge regression solver
- `pcghal_call()` - Gradient descent optimizer
- `fast_pchal()` - LASSO soft-thresholding
- `kernel_cross()` - Cross-kernel for predictions

## Performance Tips

1. **Reduce Features**: Use domain knowledge to select relevant features
2. **Scale Appropriately**: Center/scale if features have different ranges
3. **Choose Good maxdeg**: Start with 2, increase only if needed
4. **Tune npc**: Balance dimension reduction against information loss
5. **Use CV**: Let cross-validation find best lambda for your data
6. **Batch Predictions**: Vectorized operations faster than loop

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py::TestSinglePcghal::test_single_fit -v

# Run with coverage
pytest tests/ --cov=hapc

# Run validation tests
python test_norm_routing.py
python test_comprehensive.py
```

## Support & Debugging

### Verbose Mode
```python
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.1,
                       verbose=True)
# Shows all intermediate steps and convergence info
```

### Check Dimensions
```python
print(f"X shape: {X.shape}")          # Should be (n, p)
print(f"Y shape: {Y.shape}")          # Should be (n,)
print(f"X is C-contiguous: {X.flags['C_CONTIGUOUS']}")
```

### Inspect Results
```python
result = hapc(X, Y, norm="sv", single_lambda=0.1, predict=X_test)
print(f"Alpha type: {type(result.alpha)}, shape: {result.alpha.shape}")
print(f"Alpha values: min={result.alpha.min()}, max={result.alpha.max()}")
print(f"Risk value: {result.risk}")
print(f"Iterations: {result.iter}")
```

---

**Last Updated**: 2024-12-20  
**Version**: 1.0  
**Status**: Production Ready
