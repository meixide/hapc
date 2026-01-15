# HAPC Copilot Instructions

## Project Overview
**HAPC** (Highly Adaptive Principal Components) is a dual-language ML library for regression/classification based on the Highly Adaptive Lasso. It features C++ core algorithms with Python (pybind11) and R (Rcpp) bindings.

## Architecture

### Core Components
- **`src/hapc_core.hpp`**: Defines core algorithm interfaces (`DesignOutput`, `OptimizerOutput`) and function signatures
- **C++ Implementations** (`src/*.cpp`):
  - `pchal_design.cpp`: Polynomial-Kernel HAL design matrix generation
  - `ridge_wrappers.cpp`, `mkernel.cpp`, `cross_kernel.cpp`: Linear algebra kernels
  - `pcghal_call.cpp`, `pcghal_classi_call.cpp`: Gradient-based optimizers (regression/classification)
  - `fast_pchal.cpp`: Fast ridge regression for single lambda
  - `cv_fast_pchal_python.cpp`: CV wrapper (Python-specific)
- **Bindings**:
  - `src/bindings.cpp`: pybind11 module (`hapc_core`) exposing C++ functions to Python
  - `src/r_bindings.cpp`, `src/init.c`: R/Rcpp bindings

### Python Package Structure
```
python/hapc/
├── __init__.py          # Exports high-level API
├── core.py              # Low-level C++ function wrappers + fallback import logic
├── single.py            # Single-lambda fitting (wraps pcghal_call)
└── cv.py                # Cross-validation (wraps CV functions)
```

## Build System

### Python Build (CMake + setuptools)
1. **CMakeLists.txt** defines pybind11 module compilation:
   - Finds Python, Eigen3, pybind11
   - Compiles `src/*.cpp` + `src/bindings.cpp` → `hapc_core` extension module
2. **setup.py** (root):
   - Uses custom `CMakeBuild` class to invoke CMake
   - Copies built `hapc_core.*` to `python/hapc/`
   - Entry point: `pip install -e .`

### Critical Caveat: Module Import
`python/hapc/core.py` contains complex fallback logic to locate the `hapc_core` module:
1. Direct import
2. Relative import (from parent package)
3. Search in build directories and sys.path

**Reason**: CMake builds into temporary directories; setup.py must copy the module after build.

### R Build
Uses standard R package build (`src/Makevars`, `DESCRIPTION`, `NAMESPACE`). Separate from Python build.

## Key Development Patterns

### Array Handling
Always use `_ensure_c_contiguous()` wrapper before passing NumPy arrays to C++:
```python
def ridge_regression(Y, U, D2, lambda_):
    Y = _ensure_c_contiguous(Y, np.float64)
    U = _ensure_c_contiguous(U, np.float64)
    D2 = _ensure_c_contiguous(D2, np.float64)
    return hapc_core.ridge_call(Y, U, D2, float(lambda_))
```
This ensures pybind11/Eigen integration works correctly.

### NamedTuples for Outputs
Wrap C++ struct returns as Python NamedTuples for clean API:
```python
class OptimizerOutput(NamedTuple):
    alpha: np.ndarray
    alphaiters: np.ndarray
    beta: np.ndarray
    risk: float
    iter: int
```

### Design-Then-Optimize Pattern
Most ML functions follow:
1. **Design phase** (`pchal_design`): Generate feature matrix H with PCA compression
2. **Kernel phase** (`mkernel`): Compute interaction kernel matrix
3. **Optimize phase** (`pcghal_call`): Iterative gradient-based optimization

See `python/hapc/single.py` for concrete example.

## Testing

**Location**: `tests/` directory  
**Framework**: pytest  
**Run**: `pytest -v` or `pytest tests/test_api.py::TestClassName::test_method`  
**Key patterns**:
- Use `@pytest.fixture` for test data (see `regression_data` fixture)
- Test both `center=True/False` variants (results should differ)
- Verify output shapes and finite values, not exact values

## Common Workflows

### Install Development Version
```bash
pip install -e .
```
Rebuilds C++ extension via CMake. Clean with `rm -rf build/ python/hapc/hapc_core.*` if needed.

### Debug Build Failures
1. Check CMake finds Eigen3: `cmake --version && pkg-config --modversion eigen3`
2. Verify pybind11: `python -c "import pybind11; print(pybind11.get_cmake_dir())"`
3. Check manual CMake build: `cd build && cmake .. && cmake --build .`

### Run Tests
```bash
pytest tests/ -v
```

## Known Issues

1. **Import Fragility**: If `hapc_core` import fails, manually verify the `.so/.pyd` file exists in `python/hapc/`
2. **Build Artifacts**: CMake leaves objects in `src/` on some systems; safe to delete
3. **Missing CV Functions**: Some R-specific CV wrappers (`single_pchar.cpp`, `cv_fast_pchal.cpp`) not exposed to Python—this is intentional (use `pcghal_cv` instead)

## Files to Know

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Python C++ build configuration |
| `src/bindings.cpp` | pybind11 module definition |
| `src/hapc_core.hpp` | Core algorithm signatures |
| `python/hapc/core.py` | Low-level Python wrappers + module loading logic |
| `python/hapc/single.py` | High-level single-lambda API |
| `python/hapc/cv.py` | Cross-validation API |
| `pyproject.toml` | Metadata, dependencies, build system |
| `tests/test_api.py` | High-level API tests |

## Before Making Changes

- **Modifying C++ interfaces**: Update both `src/hapc_core.hpp` and `src/bindings.cpp`
- **Adding new functions**: Expose in `src/bindings.cpp`, wrap in `python/hapc/core.py`, test in `tests/`
- **Changing Python API**: Check test compatibility; follow NamedTuple + `_ensure_c_contiguous()` patterns
- **Debugging**: Add `verbose=True` to optimizer calls; check `alphaiters` matrix for convergence
