# HAPC Project Analysis & Recommendations

## Current State Analysis

### Issues Identified

#### 1. **Multiple Build Systems & Duplicate Setup Files**
   - **Root level**: `setup.py`, `pyproject.toml`, `CMakeLists.txt`
   - **Python subdirectory**: `python/setup.py`, `python/pyproject.toml` (empty)
   - **Manual setup**: `setup_manual.py` (alternative pybind11 approach)
   - **R package**: `src/Makevars` (standard R package build)

   **Problem**: Confusing which build system to use. Multiple entry points for Python installation.

#### 2. **Inconsistent Source File Lists**
   - **CMakeLists.txt** includes:
     - `pchal_design.cpp`, `ridge_wrappers.cpp`, `mkernel.cpp`, `cross_kernel.cpp`
     - `pcghal_call.cpp`, `pcghal_classi_call.cpp`, `fast_pchal.cpp`
     - Missing: `single_pchar.cpp`, `cv_fast_pchal.cpp`
   
   - **src/Makevars** includes:
     - All above PLUS: `r_bindings.cpp`, `single_pchar.cpp`, `cv_fast_pchal.cpp`
   
   **Problem**: R package has functions that Python package doesn't expose (or uses different implementations).

#### 3. **Build Artifacts in Source Directory**
   - `.o` files in `src/` (should be in build directories)
   - `.so` files scattered in `src/` and `build/`
   - CMake build artifacts in `build/`

   **Problem**: Source directory is polluted with build artifacts.

#### 4. **Python Import Complexity**
   - `python/hapc/core.py` has complex fallback logic to find `hapc_core` module
   - Module name inconsistency: `hapc_core` vs `hapc._core` in setup.py
   - Hardcoded paths in `cv.py` looking for R's `.so` files

   **Problem**: Fragile import mechanism suggests build/install issues.

#### 5. **Missing Source Files in Python Build**
   - `single_pchar.cpp` - used by R but not in CMakeLists
   - `cv_fast_pchal.cpp` - used by R but not in CMakeLists
   - These may be needed for full Python functionality

#### 6. **R Package Structure**
   - Standard R package structure (good)
   - Uses `src/Makevars` correctly
   - Has `src/init.c` for R function registration
   - Has proper `DESCRIPTION` and `NAMESPACE`

## Recommended Solution

### Strategy: Unified Build with Separate Entry Points

The goal is to:
1. Keep C++ sources in `src/` (shared between R and Python)
2. Use CMake as the primary build system for Python
3. Keep R's standard build system (`src/Makevars`) for R package
4. Clean up duplicate setup files
5. Ensure both packages can be built independently

### Recommended Structure

```
hapc/
├── CMakeLists.txt          # Python build (pybind11)
├── DESCRIPTION             # R package metadata
├── NAMESPACE               # R package exports
├── pyproject.toml          # Python package metadata (single source of truth)
├── setup.py                # Python package installer (uses CMake)
├── src/                    # Shared C++ sources
│   ├── hapc_core.hpp       # Core header
│   ├── pchal_design.cpp    # Core implementation
│   ├── ridge_wrappers.cpp
│   ├── mkernel.cpp
│   ├── cross_kernel.cpp
│   ├── pcghal_call.cpp
│   ├── pcghal_classi_call.cpp
│   ├── fast_pchal.cpp
│   ├── single_pchar.cpp    # R-specific wrapper
│   ├── cv_fast_pchal.cpp  # R-specific wrapper
│   ├── bindings.cpp        # Python bindings (pybind11)
│   ├── r_bindings.cpp      # R bindings (Rcpp)
│   └── init.c              # R function registration
├── R/                      # R package functions
├── python/                 # Python package
│   └── hapc/
│       ├── __init__.py
│       ├── core.py
│       ├── single.py
│       └── cv.py
├── man/                    # R documentation
├── tests/                  # Python tests
└── build/                  # Build artifacts (gitignored)
```

### Action Plan

#### Phase 1: Cleanup
1. Remove duplicate setup files
2. Clean build artifacts from `src/`
3. Update `.gitignore` to exclude build artifacts
4. Consolidate to single `setup.py` at root

#### Phase 2: Fix CMakeLists.txt
1. Add missing source files if needed for Python
2. Ensure all core C++ sources are included
3. Verify Python bindings are complete

#### Phase 3: Simplify Python Setup
1. Use single `setup.py` at root with CMake build
2. Remove `python/setup.py`
3. Fix module import paths
4. Update `pyproject.toml` as single source of truth

#### Phase 4: Verify Both Packages
1. Test R package build: `R CMD build .` and `R CMD INSTALL .`
2. Test Python package build: `pip install -e .`
3. Ensure both work independently

## Next Steps

Would you like me to:
1. **Clean up the project structure** (remove duplicates, organize files)
2. **Fix the CMakeLists.txt** to include all necessary sources
3. **Consolidate setup files** to a single Python build system
4. **Create a proper .gitignore** to exclude build artifacts
5. **Test both R and Python builds** to ensure they work

Let me know which steps you'd like me to proceed with!
