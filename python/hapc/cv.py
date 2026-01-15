"""Cross-validation functions - calls C++ fasthal_cv_call."""

import numpy as np
from typing import Optional, NamedTuple
import ctypes
from .core import kernel_cross, mkernel

class CVResult(NamedTuple):
    """Cross-validation result."""
    mses: np.ndarray
    lambdas: np.ndarray
    best_lambda: float
    best_model_alpha: np.ndarray
    predictions: Optional[np.ndarray] = None

def pcghal_cv(X: np.ndarray, Y: np.ndarray, maxdeg: int, npc: int,
              lambdas: Optional[np.ndarray] = None,
              log_lambda_min: float = -5, 
              log_lambda_max: float = -3,
              grid_length: int = 10,
              nfolds: int = 5,
              predict: Optional[np.ndarray] = None,
              center: bool = True, verbose: bool = False,
              max_iter: int = 100, tol: float = 1e-6) -> CVResult:
    """
    Cross-validation for PC-GHAL with gradient descent optimizer.
    Calls C++ pcghal_cv_fit directly (matches R pchal_cv_call).
    
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
    lambdas : np.ndarray, optional
        Array of lambda regularization parameters to test.
        If None, generates grid from log_lambda_min to log_lambda_max.
    log_lambda_min : float, default=-5
        Minimum log(lambda) for grid generation (if lambdas is None)
    log_lambda_max : float, default=-3
        Maximum log(lambda) for grid generation (if lambdas is None)
    grid_length : int, default=10
        Number of lambda values to generate (if lambdas is None)
    nfolds : int, default=5
        Number of folds for cross-validation
    predict : np.ndarray, optional
        Test data for predictions
    center : bool, default=True
        Center the design matrix
    verbose : bool, default=False
        Print progress information
    max_iter : int, default=100
        Maximum iterations for optimizer
    tol : float, default=1e-6
        Convergence tolerance
    
    Returns
    -------
    CVResult
        Cross-validation results with best lambda and predictions
    """
    from .core import _ensure_c_contiguous, hapc_core
    
    X = _ensure_c_contiguous(X)
    Y = _ensure_c_contiguous(Y)
    
    # Generate lambda grid if not provided
    if lambdas is None:
        log_lambdas = np.linspace(log_lambda_min, log_lambda_max, grid_length)
        lambdas = np.exp(log_lambdas)
    
    lambdas = np.asarray(lambdas, dtype=np.float64)
    n, p = X.shape
    
    if predict is not None:
        predict = _ensure_c_contiguous(predict)
    else:
        predict = np.array([], dtype=np.float64).reshape(0, p)
    
    if verbose:
        print("=" * 60)
        print("PC-GHAL Cross-Validation (C++ Implementation)")
        print("=" * 60)
        print(f"Lambda grid: {len(lambdas)} values from {lambdas.min():.6f} to {lambdas.max():.6f}")
    
    # Call C++ pcghal_cv_fit directly
    result_cpp = hapc_core.pcghal_cv_fit(
        X, Y, maxdeg, npc, lambdas.tolist(), nfolds,
        predict,
        max_iter, tol, 1.0, verbose, "risk", center, False
    )
    
    # Extract predictions
    predictions_out = None
    if predict.shape[0] > 0 and result_cpp.predictions.size > 0:
        predictions_out = result_cpp.predictions
    
    return CVResult(
        mses=np.array(result_cpp.mses),
        lambdas=np.array(result_cpp.lambdas),
        best_lambda=result_cpp.best_lambda,
        best_model_alpha=result_cpp.best_alpha,
        predictions=predictions_out
    )


def fasthal_cv(X: np.ndarray, Y: np.ndarray, npc: int,
               lambdas: np.ndarray, nfolds: int = 5,
               predict: Optional[np.ndarray] = None,
               maxdeg: int = 1, center: bool = True,
               approx: bool = False, l1: bool = False) -> CVResult:
    """
    Fast cross-validation with L1 (LASSO) or L2 (Ridge) penalties.
    Matches R cv.hapc with norm="1" or norm="2".
    
    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Input features
    Y : np.ndarray, shape (n,)
        Response variable
    npc : int
        Number of principal components
    lambdas : np.ndarray
        Array of lambda regularization parameters to test
    nfolds : int, default=5
        Number of folds for cross-validation
    predict : np.ndarray, optional
        Test data for predictions
    maxdeg : int, default=1
        Maximum degree of interactions
    center : bool, default=True
        Center the design matrix
    approx : bool, default=False
        Use approximate eigendecomposition
    l1 : bool, default=False
        Use L1 penalty (LASSO), otherwise L2 (Ridge)
    
    Returns
    -------
    CVResult
        Cross-validation results with best lambda and predictions
    """
    from .single import single_lambda_fit
    from .core import _ensure_c_contiguous, pchal_design
    from sklearn.model_selection import KFold
    
    X = _ensure_c_contiguous(X)
    Y = _ensure_c_contiguous(Y)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    n, p = X.shape
    
    if predict is not None:
        predict = _ensure_c_contiguous(predict)
    else:
        predict = np.array([], dtype=np.float64).reshape(0, p)
    
    # CV loop
    cv = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    cv_mses = np.zeros((nfolds, len(lambdas)))
    
    fold_idx = 0
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        for j, lam in enumerate(lambdas):
            # Fit on train
            result = single_lambda_fit(X_train, Y_train, maxdeg=maxdeg, 
                                       npc=npc, single_lambda=lam, 
                                       center=center, approx=approx, l1=l1)
            
            # Predict on test
            if X_test.shape[0] > 0 and result.alpha is not None:
                # Make predictions on test set using predict parameter
                result_test = single_lambda_fit(X_train, Y_train, maxdeg=maxdeg, 
                                                npc=npc, single_lambda=lam, 
                                                predict=X_test,
                                                center=center, approx=approx, l1=l1)
                
                if result_test.predictions is not None:
                    y_pred = result_test.predictions
                    cv_mses[fold_idx, j] = np.mean((Y_test - y_pred) ** 2)
                else:
                    cv_mses[fold_idx, j] = np.inf
            else:
                cv_mses[fold_idx, j] = np.inf
        
        fold_idx += 1
    
    # Average CV MSE
    mean_mses = np.nanmean(cv_mses, axis=0)
    best_idx = np.nanargmin(mean_mses)
    best_lambda = lambdas[best_idx]
    
    # Refit on full data with best lambda
    result_final = single_lambda_fit(X, Y, maxdeg=maxdeg, npc=npc,
                                     single_lambda=best_lambda, center=center,
                                     approx=approx, l1=l1)
    
    # Predictions on test set if provided
    predictions_out = None
    if predict is not None and predict.shape[0] > 0:
        K_pred = kernel_cross(X, predict, m=maxdeg, center=center)
        K = mkernel(X, m=maxdeg, center=center)
        
        evals, evecs = np.linalg.eigh(K)
        des = pchal_design(X, maxdeg=maxdeg, npc=npc, center=center)
        final_npc = des.d.shape[0]
        
        idx = np.argsort(-evals)[:final_npc]
        U = evecs[:, idx]
        D = np.sqrt(evals[idx])
        D_inv = np.diag(1.0 / (D + 1e-12))
        
        predictions_out = K_pred @ U @ D_inv @ result_final.alpha
        
        if center:
            predictions_out += Y.mean()
    
    return CVResult(
        mses=mean_mses,
        lambdas=lambdas,
        best_lambda=best_lambda,
        best_model_alpha=result_final.alpha,
        predictions=predictions_out
    )


def cv_hapc(X: np.ndarray, Y: np.ndarray, maxdeg: int, npc: int,
            log_lambda_min: float = -5, log_lambda_max: float = -3,
            grid_length: int = 10, nfolds: int = 5,
            norm: str = "sv", predict: Optional[np.ndarray] = None,
            center: bool = True, approx: bool = False,
            verbose: bool = False, max_iter: int = 100, 
            tol: float = 1e-6) -> CVResult:
    """
    High-level cross-validation dispatcher matching R cv.hapc().
    
    Automatically generates lambda grid and routes to appropriate solver
    based on norm parameter.
    
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
    log_lambda_min : float, default=-5
        Minimum log(lambda) for grid generation
    log_lambda_max : float, default=-3
        Maximum log(lambda) for grid generation
    grid_length : int, default=10
        Number of lambda values to generate
    nfolds : int, default=5
        Number of CV folds
    norm : str, default="sv"
        Normalization/solver type:
        - "sv": Gradient descent (PC-GHAL) via pcghal_cv
        - "1": L1 penalty (LASSO) via fasthal_cv with l1=True
        - "2": L2 penalty (Ridge) via fasthal_cv with l1=False
    predict : np.ndarray, optional
        Test data for predictions (shape: (m, p))
    center : bool, default=True
        Center the design matrix
    approx : bool, default=False
        Use approximate eigendecomposition (for norm="1" or "2")
    verbose : bool, default=False
        Print progress information
    max_iter : int, default=100
        Maximum iterations for optimizer (norm="sv" only)
    tol : float, default=1e-6
        Convergence tolerance (norm="sv" only)
    
    Returns
    -------
    CVResult
        Cross-validation results with fields:
        - mses: MSE for each lambda
        - lambdas: Lambda values tested
        - best_lambda: Optimal lambda
        - best_model_alpha: Coefficients for best model
        - predictions: Predictions on test set (if predict provided)
    
    Examples
    --------
    >>> # Gradient descent (PC-GHAL)
    >>> cv_sv = cv_hapc(X, Y, maxdeg=2, npc=10, norm="sv")
    
    >>> # Ridge regression
    >>> cv_l2 = cv_hapc(X, Y, maxdeg=2, npc=10, norm="2")
    
    >>> # LASSO
    >>> cv_l1 = cv_hapc(X, Y, maxdeg=2, npc=10, norm="1")
    
    >>> # With predictions
    >>> cv_sv = cv_hapc(X, Y, maxdeg=2, npc=10, norm="sv", predict=Xnew)
    """
    # Generate lambda grid from log scale
    log_lambdas = np.linspace(log_lambda_min, log_lambda_max, grid_length)
    lambdas = np.exp(log_lambdas)
    
    if verbose:
        print(f"CV with norm='{norm}'")
        print(f"Lambda grid: {len(lambdas)} values from {lambdas.min():.6f} to {lambdas.max():.6f}")
    
    if norm == "sv":
        # Gradient descent optimizer (PC-GHAL)
        if verbose:
            print("Using PC-GHAL gradient descent optimizer")
        return pcghal_cv(X, Y, maxdeg, npc, lambdas=lambdas, nfolds=nfolds,
                        predict=predict, center=center, verbose=verbose,
                        max_iter=max_iter, tol=tol)
    
    elif norm == "1":
        # L1 penalty (LASSO)
        if verbose:
            print("Using L1 penalty (LASSO soft-thresholding)")
        return fasthal_cv(X, Y, npc, lambdas, nfolds=nfolds,
                         predict=predict, maxdeg=maxdeg, center=center,
                         approx=approx, l1=True)
    
    elif norm == "2":
        # L2 penalty (Ridge)
        if verbose:
            print("Using L2 penalty (Ridge regression)")
        return fasthal_cv(X, Y, npc, lambdas, nfolds=nfolds,
                         predict=predict, maxdeg=maxdeg, center=center,
                         approx=approx, l1=False)
    
    else:
        raise ValueError(f"Unknown norm='{norm}'. Must be 'sv', '1', or '2'")
