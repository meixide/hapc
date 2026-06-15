# HAPC: Highly Adaptive Prinicipal Components

A fast and flexible machine learning library for nonparametric high-dimensional regression and classification with guarantees.

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler (g++, clang, or MSVC)
- CMake 3.15+
- Eigen3

### Quick Install

```bash
pip install hapc
```

Prebuilt wheels are published for Linux (manylinux2014, x86_64), macOS
(Intel + Apple Silicon) and Windows, for CPython 3.8–3.12. No compiler,
CMake or Eigen is needed when a wheel is available.

### Linux / HPC clusters

The Linux wheels use the **manylinux2014** baseline (glibc 2.17), so
`pip install hapc` works out of the box on HPC login/compute nodes —
no `conda` toolchain, `devtoolset`, or sysroot setup required:

```bash
pip install hapc
```

If you must build from the source distribution (niche architecture, very
old Python, or an air-gapped node), provide a C++17 compiler and either
let CMake fetch Eigen automatically (needs network) or install Eigen and
let `find_package(Eigen3)` find it:

```bash
# with conda compilers (recommended on HPC)
conda install -c conda-forge cxx-compiler cmake eigen
pip install hapc --no-binary hapc
```

### Install from GitHub (latest development version)

```bash
pip install git+https://github.com/meixide/hapc.git
```

Or with editable install for development:

```bash
git clone https://github.com/meixide/hapc.git
cd hapc
pip install -e .
```

### Install build dependencies

If installation fails, you may need to install build dependencies:

**macOS:**
```bash
brew install cmake eigen
```

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake libeigen3-dev build-essential
```

**Windows:**
```bash
pip install cmake
# Install Visual Studio Build Tools or use conda
conda install -c conda-forge eigen
```

## Quick Start

```python
import numpy as np
from hapc.single import single_pcghal
from hapc.cv import pcghal_cv

# Generate sample data
X = np.random.randn(100, 5)
Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

# Single fit with fixed lambda
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01)
print(f"Risk: {result.optimizer_output.risk:.6f}")

# Cross-validation to select lambda
lambdas = np.logspace(-4, 0, 10)
cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, nfolds=5)
print(f"Best lambda: {cv_result.best_lambda:.6f}")

# Make predictions
X_test = np.random.randn(20, 5)
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01, predict=X_test)
print(f"Predictions: {result.predictions}")
```

## Usage

### Regression

```python
from hapc.single import single_pcghal

result = single_pcghal(
    X, Y,
    maxdeg=2,        # Maximum degree of interactions
    npc=10,          # Number of principal components
    single_lambda=0.01,
    predict=X_test   # Optional: test data for predictions
)
```

### Classification

```python
from hapc.single import single_pcghal

result = single_pcghal(
    X, Y_binary,
    maxdeg=2,
    npc=10,
    single_lambda=0.01,
    predict=X_test
)
```

### Cross-Validation

```python
from hapc.cv import pcghal_cv

cv_result = pcghal_cv(
    X, Y,
    maxdeg=2,
    npc=10,
    lambdas=np.logspace(-4, 0, 20),
    nfolds=5
)
print(cv_result.best_lambda)
```

## API Reference

### `hapc.single.single_pcghal()`

Fit PC-GHAL with a single lambda value.

**Parameters:**
- `X` (ndarray, shape (n, p)): Input features
- `Y` (ndarray, shape (n,)): Response variable
- `maxdeg` (int): Maximum degree of interactions
- `npc` (int): Number of principal components
- `single_lambda` (float): Regularization parameter
- `max_iter` (int, default=100): Maximum iterations
- `tol` (float, default=1e-6): Convergence tolerance
- `verbose` (bool, default=False): Print progress
- `predict` (ndarray, optional): Test data for predictions
- `center` (bool, default=True): Center the design matrix

**Returns:**
- `result.optimizer_output.alpha`: Coefficients
- `result.optimizer_output.risk`: Final risk
- `result.optimizer_output.iter`: Iterations until convergence
- `result.predictions`: Predictions on test data (if provided)

### `hapc.cv.pcghal_cv()`

Cross-validation to select lambda.

**Parameters:**
- `lambdas` (ndarray): Grid of lambda values to test
- `nfolds` (int, default=5): Number of CV folds
- ...other parameters same as `single_pcghal`

**Returns:**
- `cv_result.best_lambda`: Optimal lambda
- `cv_result.mses`: CV errors for each lambda
- `cv_result.best_model`: Fitted model with best lambda
- `cv_result.predictions`: Predictions on test data (if provided)

## Contributing

Contributions welcome! The C++ core is shared between R and Python packages.

```bash
git clone https://github.com/meixide/hapc.git
cd hapc
pip install -e .
pytest
```

## License

MIT License - see LICENSE file
