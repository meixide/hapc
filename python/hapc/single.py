"""Single lambda model fitting - wraps C++ single_lambda_pchar."""

import numpy as np
from typing import Optional, NamedTuple
from .core import (pchal_design, ridge_regression, kernel_cross, mkernel, 
                   pcghal, pcghal_classification)

class SingleLambdaResult(NamedTuple):
    """Result from single_lambda_pchar."""
    alpha: np.ndarray
    predictions: Optional[np.ndarray] = None
    lambda_: float = None
    optimizer_output: Optional[NamedTuple] = None  # Full optimizer output for norm="sv"

class SinglePcghalResult(NamedTuple):
    """Result from single_pcghal (gradient descent optimizer)."""
    alpha: np.ndarray
    predictions: Optional[np.ndarray] = None
    lambda_: float = None
    optimizer_output: Optional[NamedTuple] = None
    risk: float = None
    iter: int = None

def single_lambda_fit(X: np.ndarray, Y: np.ndarray, maxdeg: int, npc: int,
                      single_lambda: float, predict: Optional[np.ndarray] = None,
                      center: bool = True, approx: bool = False, l1: bool = False) -> SingleLambdaResult:
    """
    Fit model with single lambda using either L1 or L2 penalty.
    Mirrors C++ single_lambda_pchar implementation.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Input features
    Y : np.ndarray, shape (n,)
        Response variable
    maxdeg : int
        Maximum degree of interactions
    npc : int
        Number of principal components
    single_lambda : float
        Regularization parameter
    predict : np.ndarray, optional
        Test data for predictions
    center : bool, default=True
        Center the design matrix
    approx : bool, default=False
        Use approximate eigendecomposition
    l1 : bool, default=False
        Use L1 penalty (LASSO), otherwise L2 (Ridge)
    
    Returns
    -------
    SingleLambdaResult
        Named tuple with alpha coefficients and optional predictions
    """
    X = np.asarray(X, dtype=np.float64, order='C')
    Y = np.asarray(Y, dtype=np.float64, order='C')
    n, p = X.shape
    
    # Generate design
    des = pchal_design(X, maxdeg, npc, center=center)
    final_npc = des.d.shape[0]
    
    # Kernel matrix
    K = mkernel(X, maxdeg, center=center)
    
    # Eigendecomposition
    evals, evecs = np.linalg.eigh(K)
    idx = np.argsort(-evals)[:final_npc]
    U = evecs[:, idx]
    D = np.sqrt(evals[idx])  # Singular values (square root of eigenvalues)
    D2 = evals[idx]  # Eigenvalues for ridge_call
    
    # Xtilde = U * D (singular values)
    Xtilde = U @ np.diag(D)
    
    # Center Y
    ymean = Y.mean() if center else 0.0
    Y_centered = Y - ymean if center else Y
    
    # Solve
    if l1:
        # LASSO: use fast_pchal_call logic
        from .core import fast_pchal
        alpha = fast_pchal(U, D2, Y_centered, single_lambda)
    else:
        # Ridge: use ridge_call logic
        from .core import ridge_regression
        alpha = ridge_regression(Y_centered, U, D2, single_lambda)
    
    # Predictions
    predictions_out = None
    if predict is not None:
        predict = np.asarray(predict, dtype=np.float64, order='C')
        if predict.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        
        Ktest = kernel_cross(X, predict, maxdeg, center=center)
        D2_inv_sqrt = np.diag(1.0 / np.sqrt(D2 + 1e-12))
        predictions_out = Ktest @ U @ D2_inv_sqrt @ alpha
        
        if center:
            predictions_out += ymean
    
    return SingleLambdaResult(alpha=alpha, predictions=predictions_out, lambda_=single_lambda)


def single_pcghal(X: np.ndarray, Y: np.ndarray, maxdeg: int, npc: int,
                  single_lambda: float, predict: Optional[np.ndarray] = None,
                  center: bool = True, approx: bool = False, verbose: bool = False,
                  max_iter: int = 100, tol: float = 1e-6) -> SinglePcghalResult:
    """
    Fit model with single lambda using gradient descent optimizer (PC-GHAL).
    This is the high-level interface matching R's hapc() with norm="sv".
    **Calls the C++ single_pcghal_fit function directly.**
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Input features
    Y : np.ndarray, shape (n,)
        Response variable
    maxdeg : int
        Maximum degree of interactions
    npc : int
        Number of principal components
    single_lambda : float
        Regularization parameter
    predict : np.ndarray, optional
        Test data for predictions
    center : bool, default=True
        Center the design matrix
    approx : bool, default=False
        Use approximate eigendecomposition
    verbose : bool, default=False
        Print iteration details
    max_iter : int, default=100
        Maximum iterations for gradient descent
    tol : float, default=1e-6
        Convergence tolerance
    
    Returns
    -------
    SinglePcghalResult
        Named tuple with alpha, predictions, optimizer output, and convergence info
    """
    from .core import _ensure_c_contiguous, hapc_core
    
    X = _ensure_c_contiguous(X)
    Y = _ensure_c_contiguous(Y)
    n, p = X.shape
    
    # Prepare prediction data
    if predict is not None:
        predict = _ensure_c_contiguous(predict)
        if predict.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        predict_data = predict
    else:
        # Empty matrix for no predictions
        predict_data = np.array([], dtype=np.float64).reshape(0, p)
    
    if verbose:
        print("=" * 60)
        print("PC-GHAL Single Lambda Optimization (C++ Implementation)")
        print("=" * 60)
        print()
    
    # Call C++ single_pcghal_fit directly
    result_cpp = hapc_core.single_pcghal_fit(
        X, Y, maxdeg, npc, single_lambda, predict_data,
        max_iter, tol, 1.0, verbose, "grad", center, approx
    )
    
    # Extract predictions
    predictions_out = None
    if predict is not None and result_cpp.predictions.size > 0:
        predictions_out = result_cpp.predictions
    
    return SinglePcghalResult(
        alpha=result_cpp.alpha,
        predictions=predictions_out,
        lambda_=single_lambda,
        optimizer_output=None,  # Not available from C++ output
        risk=result_cpp.risk,
        iter=result_cpp.iter
    )



def hapc(X: np.ndarray, Y: np.ndarray, maxdeg: int, npc: int,
         single_lambda: float, norm: str = "sv", predict: Optional[np.ndarray] = None,
         center: bool = True, approx: bool = False, verbose: bool = False,
         max_iter: int = 100, tol: float = 1e-6) -> SinglePcghalResult:
    """
    High-level interface matching R's hapc() function.
    Dispatches to appropriate solver based on norm parameter.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Input features
    Y : np.ndarray, shape (n,)
        Response variable
    maxdeg : int
        Maximum degree of interactions
    npc : int
        Number of principal components
    single_lambda : float
        Regularization parameter
    norm : str, default="sv"
        Normalization/solver type:
        - "1": L1 penalty (LASSO soft-thresholding)
        - "2": L2 penalty (Ridge regression, closed-form)
        - "sv": Supervised (gradient descent optimizer PC-GHAL)
    predict : np.ndarray, optional
        Test data for predictions
    center : bool, default=True
        Center the design matrix
    approx : bool, default=False
        Use approximate eigendecomposition
    verbose : bool, default=False
        Print iteration details
    max_iter : int, default=100
        Maximum iterations (for norm="sv")
    tol : float, default=1e-6
        Convergence tolerance (for norm="sv")
    
    Returns
    -------
    SinglePcghalResult or SingleLambdaResult
        Fit results with predictions
    """
    if verbose:
        print(f"HAPC with norm='{norm}'")
    
    if norm == "1":
        # L1 (LASSO)
        if verbose:
            print("Using L1 penalty (soft-thresholding)")
        return single_lambda_fit(X, Y, maxdeg, npc, single_lambda, predict=predict,
                                center=center, approx=approx, l1=True)
    elif norm == "2":
        # L2 (Ridge)
        if verbose:
            print("Using L2 penalty (ridge regression)")
        return single_lambda_fit(X, Y, maxdeg, npc, single_lambda, predict=predict,
                                center=center, approx=approx, l1=False)
    elif norm == "sv":
        # Supervised (Gradient Descent)
        if verbose:
            print("Using gradient descent optimizer (PC-GHAL)")
        return single_pcghal(X, Y, maxdeg, npc, single_lambda, predict=predict,
                            center=center, approx=approx, verbose=verbose,
                            max_iter=max_iter, tol=tol)
    else:
        raise ValueError(f"Unknown norm='{norm}'. Must be '1', '2', or 'sv'")

