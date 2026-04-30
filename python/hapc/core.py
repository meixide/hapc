"""
Low-level Python bindings to the HAPC C++ core.

These functions are 1:1 with the C++ kernels exposed by :mod:`hapc_core`
(``pchal_des``, ``ridge_call``, ``mkernel_call``, ``kernel_cross_call``,
``pcghal_call``, ``pcghal_classi_call``, ``fast_pchal_call``).  They are the
Python counterparts of the R C-level entry points (``pchal_des``,
``ridge_call``, ``mkernel_call``, ``kernel_cross_call``, ``pcghal_call``,
``pcghal_classi_call``, ``fast_pchal_call``) and produce numerically identical
output for the same inputs.

For high-level model fitting use :func:`hapc.single.hapc` and
:func:`hapc.cv.cv_hapc` instead.
"""

from typing import NamedTuple

import numpy as np

from . import hapc_core


class DesignOutput(NamedTuple):
    """Output of :func:`design_hapc`.

    Attributes
    ----------
    H : np.ndarray, shape (n, B)
        HAL design matrix (sparse 0/1 indicators) — only stored for small
        problems; large problems may set this to an empty array.
    U : np.ndarray, shape (n, npcs)
        Left singular vectors (principal components) of ``H``.
    d : np.ndarray, shape (npcs,)
        Singular values of ``H``.
    V : np.ndarray, shape (B, npcs)
        Right singular vectors of ``H``.
    """

    H: np.ndarray
    U: np.ndarray
    d: np.ndarray
    V: np.ndarray


class OptimizerOutput(NamedTuple):
    """Output of :func:`pcghal` and :func:`pcghal_classification`.

    Attributes
    ----------
    alpha : np.ndarray, shape (npcs,)
        Coefficients on the principal-component basis.
    alphaiters : np.ndarray
        History of α iterates (rows = iterations).
    beta : np.ndarray
        Coefficients on the original HAL basis (``β = V α``).
    risk : float
        Empirical risk at the final iterate.
    iter : int
        Number of iterations executed.
    """

    alpha: np.ndarray
    alphaiters: np.ndarray
    beta: np.ndarray
    risk: float
    iter: int


def _C(x: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Return a C-contiguous, ``dtype``-typed copy / view of ``x``."""
    a = np.asarray(x, dtype=dtype)
    return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)


def design_hapc(X: np.ndarray, max_degree: int, npcs: int,
                center: bool = True) -> DesignOutput:
    """Compute the PC-HAL design ingredients ``(H, U, d, V)``.

    Counterpart to R ``design.hapc()``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix.
    max_degree : int
        Maximum interaction order for the HAL basis.
    npcs : int
        Number of principal components to keep.
    center : bool, default True
        Whether to centre ``H`` before SVD.

    Returns
    -------
    DesignOutput
        Named tuple ``(H, U, d, V)``.

    Examples
    --------
    >>> import numpy as np
    >>> from hapc import design_hapc
    >>> X = np.random.randn(50, 3)
    >>> des = design_hapc(X, max_degree=2, npcs=20)
    >>> des.U.shape
    (50, 20)
    """
    X = _C(X)
    n = X.shape[0]
    cap = n - 1 if center else n
    npcs = max(1, min(int(npcs), cap))
    out = hapc_core.pchal_des(X, int(max_degree), int(npcs), bool(center))
    return DesignOutput(out.H, out.U, out.d, out.V)


def ridge_regression(Y: np.ndarray, U: np.ndarray, D2: np.ndarray,
                     lambda_: float) -> np.ndarray:
    """Closed-form ridge in the rotated basis: ``α = (D² + λ)⁻¹ D Uᵀ Y``.

    Counterpart to R ``ridge_call()``.

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Response.
    U : np.ndarray, shape (n, k)
        Left singular vectors.
    D2 : np.ndarray, shape (k,)
        Squared singular values.
    lambda_ : float
        Regularisation parameter.

    Returns
    -------
    np.ndarray, shape (k,)
        Ridge coefficients on the PC basis.
    """
    return hapc_core.ridge_call(_C(Y), _C(U), _C(D2), float(lambda_))


def fast_pchal(U: np.ndarray, D2: np.ndarray, Y: np.ndarray,
               lambda_: float) -> np.ndarray:
    """Closed-form LASSO in the rotated basis (soft-thresholding).

    Counterpart to R ``fast_pchal_call()``.

    Parameters
    ----------
    U : np.ndarray, shape (n, k)
        Left singular vectors.
    D2 : np.ndarray, shape (k,)
        Squared singular values.
    Y : np.ndarray, shape (n,)
        Response.
    lambda_ : float
        Regularisation parameter.

    Returns
    -------
    np.ndarray, shape (k,)
        LASSO coefficients on the PC basis.
    """
    return hapc_core.fast_pchal_call(_C(U), _C(D2), _C(Y), float(lambda_))


def kernel_hapc(X: np.ndarray, max_degree: int = 1,
                center: bool = True) -> np.ndarray:
    """HAL kernel matrix ``K = H Hᵀ`` (computed implicitly).

    Counterpart to R ``kernel.hapc()``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Feature matrix.
    max_degree : int, default 1
        Maximum interaction order.
    center : bool, default True
        Whether to centre ``H``.

    Returns
    -------
    np.ndarray, shape (n, n)
        Kernel matrix.
    """
    return hapc_core.mkernel_call(_C(X), int(max_degree), bool(center))


def cross_kernel_hapc(X: np.ndarray, Xte: np.ndarray, max_degree: int = 1,
                      center: bool = True) -> np.ndarray:
    """Cross-kernel ``K = H_te H_trᵀ`` for prediction.

    Counterpart to R ``cross_kernel.hapc()``.

    Parameters
    ----------
    X : np.ndarray, shape (n_tr, p)
        Training features (defines the basis).
    Xte : np.ndarray, shape (n_te, p)
        Test features.
    max_degree : int, default 1
        Maximum interaction order.
    center : bool, default True
        Whether to apply the same column centring as in training.

    Returns
    -------
    np.ndarray, shape (n_te, n_tr)
        Cross-kernel matrix.
    """
    return hapc_core.kernel_cross_call(_C(X), _C(Xte), int(max_degree),
                                       bool(center))


def pcghal(Y: np.ndarray, Xtilde: np.ndarray, ENn: np.ndarray,
           alpha0: np.ndarray, max_iter: int = 5000, tol: float = 1e-3,
           step_factor: float = 0.8, verbose: bool = False,
           crit: str = "grad") -> OptimizerOutput:
    """Projected gradient descent for ``norm="sv"``, gaussian loss.

    Counterpart to R ``pcghal_call()`` (called by ``hapc(norm="sv")``).

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Centred response.
    Xtilde : np.ndarray, shape (n, k)
        ``U @ diag(d)``.
    ENn : np.ndarray, shape (B, k)
        Right singular vectors ``V``.
    alpha0 : np.ndarray, shape (k,)
        Initial α (e.g. from :func:`fast_pchal` or :func:`ridge_regression`).
    max_iter : int, default 5000
        Maximum number of PGD iterations.
    tol : float, default 1e-3
        Stopping tolerance.
    step_factor : float, default 0.8
        Step-size factor ``δ = step_factor / max_j |h*(j)|``.
    verbose : bool, default False
        Print iteration progress.
    crit : str, default "grad"
        Stopping criterion: ``"grad"`` (gradient inf-norm) or ``"risk"``
        (relative risk decrease).

    Returns
    -------
    OptimizerOutput
    """
    res = hapc_core.pcghal_call(_C(Y), _C(Xtilde), _C(ENn), _C(alpha0),
                                int(max_iter), float(tol),
                                float(step_factor), bool(verbose), str(crit))
    return OptimizerOutput(res.alpha, res.alphaiters, res.beta,
                           res.risk, res.iter)


def pcghal_classification(Y: np.ndarray, Xtilde: np.ndarray, ENn: np.ndarray,
                          alpha0: np.ndarray, max_iter: int = 5000,
                          tol: float = 1e-3, step_factor: float = 0.8,
                          verbose: bool = False) -> OptimizerOutput:
    """Projected gradient descent for ``norm="sv"``, logistic loss.

    Counterpart to R ``pcghal_classi_call()`` /
    ``pc_hal_classi_cpp()``.

    Parameters
    ----------
    Y : np.ndarray, shape (n,)
        Response in {-1, +1}.
    Xtilde : np.ndarray, shape (n, k)
        ``U @ diag(d)``.
    ENn : np.ndarray, shape (B, k)
        Right singular vectors ``V``.
    alpha0 : np.ndarray, shape (k,)
        Initial α (typically logistic ridge).
    max_iter, tol, step_factor, verbose
        See :func:`pcghal`.

    Returns
    -------
    OptimizerOutput
    """
    res = hapc_core.pcghal_classi_call(_C(Y), _C(Xtilde), _C(ENn), _C(alpha0),
                                       int(max_iter), float(tol),
                                       float(step_factor), bool(verbose))
    return OptimizerOutput(res.alpha, res.alphaiters, res.beta,
                           res.risk, res.iter)


def pc_hal_classi(Y: np.ndarray, Xtilde: np.ndarray, ENn: np.ndarray,
                  alpha: np.ndarray, **kwargs) -> OptimizerOutput:
    """Alias of :func:`pcghal_classification` (R name ``pc_hal_classi_cpp``).

    Parameters match ``pcghal_classification`` with ``alpha`` as the initial
    coefficient vector (same role as ``alpha0`` in :func:`pcghal_classification`).
    """
    return pcghal_classification(Y, Xtilde, ENn, alpha, **kwargs)


# --- Backward-compatible aliases (pre-0.3.0 names) -------------------------
pchal_design = design_hapc
mkernel = kernel_hapc
kernel_cross = cross_kernel_hapc


__all__ = [
    "DesignOutput",
    "OptimizerOutput",
    "design_hapc",
    "kernel_hapc",
    "cross_kernel_hapc",
    "ridge_regression",
    "fast_pchal",
    "pcghal",
    "pcghal_classification",
    "pc_hal_classi",
    # aliases
    "pchal_design",
    "mkernel",
    "kernel_cross",
]
