"""Python interface to shared HAPC C++ library."""

import numpy as np
from typing import NamedTuple, Optional
import sys
from pathlib import Path

# Try to import hapc_core module
hapc_core = None

# Try direct import
try:
    import hapc_core
except ImportError:
    pass

# Try relative import
if hapc_core is None:
    try:
        from .. import hapc_core
    except ImportError:
        pass

# Try to find it in known locations
if hapc_core is None:
    search_paths = [
        Path(__file__).parent,  # Same directory as this file
        Path(__file__).parent.parent.parent / "build",  # Build directory
    ]
    
    for path in search_paths:
        if path.exists():
            sys.path.insert(0, str(path))
            try:
                import hapc_core
                break
            except ImportError:
                continue

if hapc_core is None:
    raise ImportError(
        "hapc_core module not found. The C++ extension may not be built.\n"
        "Try: pip install -e . --force-reinstall --no-cache-dir"
    )

class DesignOutput(NamedTuple):
    """Output from pchal_design."""
    H: np.ndarray
    U: np.ndarray
    d: np.ndarray
    V: np.ndarray

class OptimizerOutput(NamedTuple):
    """Output from optimizer functions."""
    alpha: np.ndarray
    alphaiters: np.ndarray
    beta: np.ndarray
    risk: float
    iter: int

def _ensure_c_contiguous(arr: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Ensure array is C-contiguous double."""
    arr = np.asarray(arr, dtype=dtype)
    return np.ascontiguousarray(arr) if not arr.flags['C_CONTIGUOUS'] else arr

def pchal_design(X: np.ndarray, maxdeg: int, npc: int, center: bool = True) -> DesignOutput:
    """Generate PC-HAL design components."""
    X = _ensure_c_contiguous(X)
    n = X.shape[0]
    
    # Cap npc at n-1 when center=True (rank reduction due to centering)
    # Cap npc at n when center=False
    max_npc = n - 1 if center else n
    npc = min(npc, max_npc)
    npc = max(1, npc)  # At least 1
    
    result = hapc_core.pchal_des(X, int(maxdeg), int(npc), bool(center))
    return DesignOutput(result.H, result.U, result.d, result.V)

def ridge_regression(Y: np.ndarray, U: np.ndarray, D2: np.ndarray, 
                     lambda_: float) -> np.ndarray:
    """Ridge regression solver."""
    Y = _ensure_c_contiguous(Y, np.float64)
    U = _ensure_c_contiguous(U, np.float64)
    D2 = _ensure_c_contiguous(D2, np.float64)
    return hapc_core.ridge_call(Y, U, D2, float(lambda_))

def mkernel(X: np.ndarray, m: int, center: bool = True) -> np.ndarray:
    """Compute Haar-like kernel matrix."""
    X = _ensure_c_contiguous(X)
    return hapc_core.mkernel_call(X, int(m), bool(center))

def kernel_cross(Xtr: np.ndarray, Xte: np.ndarray, m: int, 
                 center: bool = True) -> np.ndarray:
    """Compute cross-kernel between training and test data."""
    Xtr = _ensure_c_contiguous(Xtr)
    Xte = _ensure_c_contiguous(Xte)
    return hapc_core.kernel_cross_call(Xtr, Xte, int(m), bool(center))

def pcghal(Y: np.ndarray, Xtilde: np.ndarray, ENn: np.ndarray, 
           alpha0: np.ndarray, max_iter: int = 100, tol: float = 1e-6,
           step_factor: float = 1.0, verbose: bool = False,
           crit: str = "grad") -> OptimizerOutput:
    """PC-GHAL optimizer (regression)."""
    Y = _ensure_c_contiguous(Y)
    Xtilde = _ensure_c_contiguous(Xtilde)
    ENn = _ensure_c_contiguous(ENn)
    alpha0 = _ensure_c_contiguous(alpha0)
    
    result = hapc_core.pcghal_call(Y, Xtilde, ENn, alpha0, int(max_iter), 
                          float(tol), float(step_factor), bool(verbose), str(crit))
    return OptimizerOutput(result.alpha, result.alphaiters, result.beta, 
                          result.risk, result.iter)

def pcghal_classification(Y: np.ndarray, Xtilde: np.ndarray, ENn: np.ndarray,
                          alpha0: np.ndarray, max_iter: int = 100, 
                          tol: float = 1e-6, step_factor: float = 1.0,
                          verbose: bool = False) -> OptimizerOutput:
    """PC-GHAL optimizer (classification)."""
    Y = _ensure_c_contiguous(Y)
    Xtilde = _ensure_c_contiguous(Xtilde)
    ENn = _ensure_c_contiguous(ENn)
    alpha0 = _ensure_c_contiguous(alpha0)
    
    result = hapc_core.pcghal_classi_call(Y, Xtilde, ENn, alpha0, int(max_iter),
                                 float(tol), float(step_factor), bool(verbose))
    return OptimizerOutput(result.alpha, result.alphaiters, result.beta,
                          result.risk, result.iter)

def fast_pchal(U: np.ndarray, D2: np.ndarray, Y: np.ndarray, 
               lambda_: float) -> np.ndarray:
    """Fast LASSO-type solver."""
    U = _ensure_c_contiguous(U)
    D2 = _ensure_c_contiguous(D2)
    Y = _ensure_c_contiguous(Y)
    return hapc_core.fast_pchal_call(U, D2, Y, float(lambda_))

__all__ = [
    "DesignOutput",
    "OptimizerOutput",
    "pchal_design",
    "ridge_regression",
    "mkernel",
    "kernel_cross",
    "pcghal",
    "pcghal_classification",
    "fast_pchal",
]
