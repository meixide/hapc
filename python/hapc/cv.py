"""
Cross-validated HAPC fitting (high-level Python API).

Provides :func:`cv_hapc` (counterpart of R ``cv.hapc()``) and the lower-level
helpers :func:`pcghal_cv` and :func:`fasthal_cv`.

All three Python CV variants delegate to a pure-C++ implementation that uses
the **same** fold partitioning as R (block partition with last fold absorbing
the remainder, then shuffled with ``std::mt19937(12345)``).  This guarantees
that ``cv.hapc()`` and ``cv_hapc()`` produce identical CV scores / best-λ / α
for the same inputs across languages (gaussian and binomial ``norm="sv"``;
binomial ``norm`` in ``{"1","2"}`` uses the same squared-error CV as R).
"""

from typing import NamedTuple, Optional

import numpy as np

from . import hapc_core
from .core import _C, cross_kernel_hapc, design_hapc
from .single import (
    _check_binomial_labels,
    _to_soft01,
    single_pcghal_classification_lasso,
)


class CVResult(NamedTuple):
    """Output of all CV variants.

    Attributes
    ----------
    mses : np.ndarray, shape (L,)
        Mean cross-validated risk per λ. For binomial CV this is logistic
        deviance (the field is still called ``mses`` for API uniformity).
    lambdas : np.ndarray, shape (L,)
        λ grid that was searched.
    best_lambda : float
        λ minimising ``mses``.
    best_model_alpha : np.ndarray, shape (k,)
        α refit on the full training data at ``best_lambda``.
    predictions : np.ndarray or None
        Predictions on ``predict`` (if supplied). For ``pcghal_cv_classi`` these
        are probabilities; for ``pcghal_cv`` / ``fasthal_cv`` they match the
        underlying regression head (continuous).
    """

    mses: np.ndarray
    lambdas: np.ndarray
    best_lambda: float
    best_model_alpha: np.ndarray
    predictions: Optional[np.ndarray]


def _grid(lambdas: Optional[np.ndarray], log_lambda_min: float,
          log_lambda_max: float, grid_length: int) -> np.ndarray:
    if lambdas is not None:
        return np.asarray(lambdas, dtype=np.float64)
    return np.exp(np.linspace(log_lambda_min, log_lambda_max, grid_length))


# ---------------------------------------------------------------------------
# norm="sv" (gaussian)  →  C++ pcghal_cv_fit (init + PGD per fold)
# ---------------------------------------------------------------------------

def pcghal_cv(X: np.ndarray, Y: np.ndarray,
              max_degree: int, npcs: int,
              lambdas: Optional[np.ndarray] = None,
              log_lambda_min: float = -5,
              log_lambda_max: float = -3,
              grid_length: int = 10,
              nfolds: int = 5,
              predict: Optional[np.ndarray] = None,
              center: bool = True, approx: bool = False,
              verbose: bool = False,
              max_iter: int = 5000, tol: float = 1e-3,
              step_factor: float = 0.8,
              crit: str = "grad",
              ini: str = "1") -> CVResult:
    """k-fold CV for gaussian HAPC with ``norm="sv"``.

    Per fold: initialise α with LASSO (``ini="1"``, default) or ridge
    (``ini="2"``), then projected gradient descent on squared error — same
    pipeline as R ``cv.hapc(family="gaussian", norm="sv")``.

    Delegates to C++ ``pcghal_cv_fit`` (shared with R), including fold IDs.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Continuous response.
    max_degree : int
        HAL interaction order.
    npcs : int
        Number of PCs.
    lambdas : np.ndarray, optional
        Explicit λ grid. If ``None``, built from ``log_*`` and ``grid_length``.
    log_lambda_min, log_lambda_max : float
        Log-space grid endpoints when ``lambdas`` is ``None``.
    grid_length : int, default 10
        Number of grid points when ``lambdas`` is ``None``.
    nfolds : int, default 5
        CV folds.
    predict : np.ndarray, optional, shape (m, p)
        Test matrix for predictions at the winning λ.
    center, approx : bool
        Passed through to design / kernel (``approx`` reserved).
    verbose : bool
        Print PGD progress.
    max_iter, tol, step_factor : float / int
        PGD budget and step-size factor.
    crit : {"grad", "risk"}, default "grad"
        PGD stopping rule (matches R default).
    ini : {"1", "2"}, default "1"
        ``"1"`` = LASSO initialisation, ``"2"`` = ridge.

    Returns
    -------
    CVResult
        ``mses`` holds mean CV MSE per λ; ``best_model_alpha`` is α refit on
        all data at ``best_lambda``.
    """
    if ini not in {"1", "2"}:
        raise ValueError(f"ini must be '1' or '2'; got '{ini}'")
    if crit not in {"grad", "risk"}:
        raise ValueError(f"crit must be 'grad' or 'risk'; got '{crit}'")

    X = _C(X)
    Y = _C(Y).ravel()
    n, p = X.shape
    if Y.size != n:
        raise ValueError(f"Y must have {n} elements, got {Y.size}")

    lams = _grid(lambdas, log_lambda_min, log_lambda_max, grid_length)
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
    else:
        Xte = np.empty((0, p), dtype=np.float64)

    res = hapc_core.pcghal_cv_fit(
        X, Y, int(max_degree), int(npcs), lams.tolist(), int(nfolds),
        Xte, int(max_iter), float(tol), float(step_factor),
        bool(verbose), str(crit), bool(center), bool(approx), str(ini),
    )

    predictions = (np.asarray(res.predictions)
                   if predict is not None and res.predictions.size > 0
                   else None)

    return CVResult(
        mses=np.asarray(res.mses),
        lambdas=np.asarray(res.lambdas),
        best_lambda=float(res.best_lambda),
        best_model_alpha=np.asarray(res.best_alpha),
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# norm in {"1","2"} (gaussian)  →  C++ fasthal_cv_fit
# ---------------------------------------------------------------------------

def fasthal_cv(X: np.ndarray, Y: np.ndarray,
               npcs: int, lambdas: np.ndarray,
               nfolds: int = 5,
               predict: Optional[np.ndarray] = None,
               max_degree: int = 1,
               center: bool = True,
               approx: bool = False,
               l1: bool = True) -> CVResult:
    """k-fold CV with closed-form LASSO or ridge in the PC basis per fold.

    Used for gaussian ``norm="1"`` / ``norm="2"`` and, in R, also for
    ``family="binomial"`` with those norms (squared-error CV on ``Y``).

    Calls the same C++ routine as R ``fasthal_cv_call`` with identical fold
    construction (``std::mt19937(12345)`` shuffle), so CV scores match R.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Training features.
    Y : np.ndarray, shape (n,)
        Response (continuous, or binary labels if using the binomial branch in
        :func:`cv_hapc`).
    npcs : int
        Number of principal components.
    lambdas : np.ndarray, shape (L,)
        Penalty grid (must be supplied explicitly; :func:`cv_hapc` builds it).
    nfolds : int, default 5
        Number of CV folds.
    predict : np.ndarray, optional, shape (m, p)
        Test features for out-of-sample predictions at the CV-winning λ.
    max_degree : int, default 1
        HAL interaction order.
    center : bool, default True
        Centre kernel / response like training.
    approx : bool, default False
        If True, approximate top eigenvectors (same flag as R).
    l1 : bool, default True
        ``True`` for LASSO (``norm="1"``), ``False`` for ridge (``norm="2"``).

    Returns
    -------
    CVResult
        ``mses`` are mean squared errors on held-out folds.
    """
    X = _C(X)
    Y = _C(Y).ravel()
    n, p = X.shape
    if Y.size != n:
        raise ValueError(f"Y must have {n} elements, got {Y.size}")

    lams = np.asarray(lambdas, dtype=np.float64)
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
    else:
        Xte = np.empty((0, p), dtype=np.float64)

    res = hapc_core.fasthal_cv_fit(
        X, Y, int(npcs), lams.tolist(), int(nfolds),
        Xte, int(max_degree),
        bool(center), bool(approx), bool(l1),
    )

    predictions = (np.asarray(res.predictions)
                   if predict is not None and res.predictions.size > 0
                   else None)

    return CVResult(
        mses=np.asarray(res.mses),
        lambdas=np.asarray(res.lambdas),
        best_lambda=float(res.best_lambda),
        best_model_alpha=np.asarray(res.best_alpha),
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# norm="sv" (binomial)  →  C++ pcghal_cv_classi_fit
# ---------------------------------------------------------------------------

def pcghal_cv_classi(X: np.ndarray, Y: np.ndarray,
                     max_degree: int, npcs: int,
                     lambdas: Optional[np.ndarray] = None,
                     log_lambda_min: float = -5,
                     log_lambda_max: float = -3,
                     grid_length: int = 10,
                     nfolds: int = 5,
                     predict: Optional[np.ndarray] = None,
                     center: bool = True,
                     verbose: bool = False,
                     max_iter: int = 5000, tol: float = 1e-3,
                     step_factor: float = 0.8,
                     with_pgd: bool = True) -> CVResult:
    """k-fold logistic-loss CV for binomial HAPC.

    Calls the shared C++ logistic CV used by R's ``cv.hapc(family="binomial")``.
    Risk is **always logistic deviance**; we never use squared error on a binary
    response.

    Parameters
    ----------
    X, Y : np.ndarray
        Features and 0/1 response.
    with_pgd : bool, default True
        ``True`` → ``norm="sv"`` (logistic ridge initialiser + projected
        gradient descent on logistic loss, per fold). ``False`` → ``norm="2"``
        (logistic ridge only, no PGD), still scored with logistic deviance.
    max_degree, npcs, lambdas, log_lambda_min, log_lambda_max, grid_length,\
        nfolds, predict, center, verbose, max_iter, tol, step_factor :
        See :func:`cv_hapc`.

    Returns
    -------
    CVResult
        ``mses`` holds mean per-fold logistic deviance. ``predictions`` are
        probabilities on ``predict``.
    """
    X = _C(X)
    Y = _C(Y).ravel()
    n, p = X.shape
    if Y.size != n:
        raise ValueError(f"Y must have {n} elements, got {Y.size}")

    lams = _grid(lambdas, log_lambda_min, log_lambda_max, grid_length)
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
    else:
        Xte = np.empty((0, p), dtype=np.float64)

    res = hapc_core.pcghal_cv_classi_fit(
        X, Y, int(max_degree), int(npcs), lams.tolist(), int(nfolds),
        Xte, int(max_iter), float(tol), float(step_factor),
        bool(verbose), bool(center), bool(with_pgd),
    )

    predictions = (np.asarray(res.predictions)
                   if predict is not None and res.predictions.size > 0
                   else None)

    return CVResult(
        mses=np.asarray(res.deviances),
        lambdas=np.asarray(res.lambdas),
        best_lambda=float(res.best_lambda),
        best_model_alpha=np.asarray(res.best_alpha),
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# norm="1" (binomial)  →  glmnet-style logistic LASSO CV (sklearn liblinear)
# ---------------------------------------------------------------------------

def _native_folds(n: int, K: int, seed: int = 12345) -> np.ndarray:
    """Block partition with last fold absorbing the remainder, then shuffled
    deterministically with a numpy MT19937 seeded by ``seed``.

    The shuffle algorithm is numpy's, not libstdc++'s, so this does **not**
    match the C++ ``make_folds`` output bit-for-bit — but it is deterministic
    and gives equally-sized folds. Used only by the binomial+norm="1" path,
    where the solver itself differs from the C++ paths anyway.
    """
    fold_size = n // K
    folds = np.empty(n, dtype=np.int64)
    for i in range(n):
        folds[i] = (i // fold_size) + 1
    for i in range(fold_size * K, n):
        folds[i] = K
    rng = np.random.default_rng(seed)
    rng.shuffle(folds)
    return folds


def pcghal_cv_classi_lasso(X: np.ndarray, Y: np.ndarray,
                            max_degree: int, npcs: int,
                            lambdas: Optional[np.ndarray] = None,
                            log_lambda_min: float = -5,
                            log_lambda_max: float = -3,
                            grid_length: int = 10,
                            nfolds: int = 5,
                            predict: Optional[np.ndarray] = None,
                            center: bool = True,
                            verbose: bool = False,
                            max_iter: int = 1000) -> CVResult:
    """k-fold logistic-LASSO CV for binomial HAPC, ``norm="1"``.

    Per fold, fits :func:`hapc.single_pcghal_classification_lasso` (sklearn
    LogisticRegression with ``penalty="l1"``, ``solver="liblinear"``,
    ``fit_intercept=False``) on the training portion and scores the held-out
    portion with logistic deviance. Mirrors R ``cv.hapc(family="binomial",
    norm="1")``, which uses ``glmnet`` per fold.

    Parameters
    ----------
    See :func:`cv_hapc`.

    Returns
    -------
    CVResult
        ``mses`` holds mean per-fold logistic deviance. ``best_model_alpha``
        is the LASSO α refit on the full data at ``best_lambda``.
        ``predictions`` are probabilities at ``best_lambda`` on ``predict``.
    """
    X = _C(X)
    Y = _C(Y).ravel()
    n, p = X.shape
    if Y.size != n:
        raise ValueError(f"Y must have {n} elements, got {Y.size}")

    lams = _grid(lambdas, log_lambda_min, log_lambda_max, grid_length)
    if not np.all(lams > 0):
        raise ValueError("All lambdas must be > 0 for logistic LASSO.")

    # Soft target in [0,1] used for the held-out cross-entropy deviance
    # (accepts hard {0,1}/{-1,+1} or fractional EM-HAL posteriors).
    q = _to_soft01(Y)
    folds = _native_folds(n, int(nfolds))
    L = lams.size
    fold_dev = np.full((int(nfolds), L), np.nan)

    for k in range(1, int(nfolds) + 1):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        if te.size == 0 or tr.size == 0:
            continue
        Xtr, Ytr = X[tr], Y[tr]
        Xte, Yte = X[te], q[te]

        for j, lam in enumerate(lams):
            res = single_pcghal_classification_lasso(
                Xtr, Ytr, max_degree=int(max_degree), npcs=int(npcs),
                lambda_=float(lam), predict=Xte, center=bool(center),
                verbose=bool(verbose), max_iter=int(max_iter),
            )
            probs = np.clip(res.probabilities, 1e-15, 1 - 1e-15)
            dev = -(Yte * np.log(probs) + (1 - Yte) * np.log(1 - probs))
            fold_dev[k - 1, j] = float(dev.mean())

    deviances = np.nanmean(fold_dev, axis=0)
    best_idx = int(np.nanargmin(deviances))
    best_lambda = float(lams[best_idx])

    full = single_pcghal_classification_lasso(
        X, Y, max_degree=int(max_degree), npcs=int(npcs),
        lambda_=best_lambda, predict=predict, center=bool(center),
        verbose=bool(verbose), max_iter=int(max_iter),
    )

    predictions = full.probabilities if predict is not None else None

    return CVResult(
        mses=np.asarray(deviances),
        lambdas=np.asarray(lams),
        best_lambda=best_lambda,
        best_model_alpha=np.asarray(full.alpha),
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def cv_hapc(X: np.ndarray, Y: np.ndarray,
            family: str = "gaussian",
            max_degree: int = 1,
            npcs: Optional[int] = None,
            log_lambda_min: float = -5,
            log_lambda_max: float = -3,
            grid_length: int = 10,
            nfolds: int = 5,
            norm: str = "sv",
            predict: Optional[np.ndarray] = None,
            max_iter: int = 5000,
            tol: float = 1e-3,
            step_factor: float = 0.8,
            verbose: bool = False,
            crit: str = "grad",
            center: bool = True,
            approx: bool = False,
            ini: str = "1") -> CVResult:
    """k-fold cross-validated HAPC fit (Python counterpart of R ``cv.hapc()``).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Response.
    family : {"gaussian", "binomial"}, default "gaussian"
        Loss family. For ``"binomial"`` the loss is **always logistic** (deviance
        CV, not MSE). ``norm="sv"`` runs logistic ridge + PGD per fold;
        ``norm="2"`` runs logistic ridge only; ``norm="1"`` runs logistic
        LASSO via ``sklearn.linear_model.LogisticRegression`` with
        ``solver="liblinear"`` (mirrors R ``glmnet`` for the binomial+norm="1"
        path).
    max_degree : int, default 1
        Maximum interaction order.
    npcs : int, optional
        Number of PCs. Defaults to ``n``.
    log_lambda_min, log_lambda_max : float
        Bounds of the log-λ grid (defaults ``-5``, ``-3``).
    grid_length : int, default 10
        Number of grid points.
    nfolds : int, default 5
        Number of CV folds. Identical fold partitions to R.
    norm : {"sv", "1", "2"}, default "sv"
        Norm constraint. See :func:`hapc.single.hapc`.
    predict : np.ndarray, optional
        Test features.
    max_iter, tol, step_factor, crit : optional
        PGD parameters used when ``norm="sv"``.
    center, approx, verbose, ini : optional
        See :func:`hapc.single.hapc`.

    Returns
    -------
    CVResult

    Examples
    --------
    >>> import numpy as np
    >>> from hapc import cv_hapc
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 3))
    >>> Y = np.sin(np.pi * X[:, 0]) + rng.standard_normal(100) * 0.1
    >>> cv = cv_hapc(X, Y, max_degree=2, npcs=99, nfolds=5)
    >>> bool(cv.best_lambda > 0)
    True
    """
    if family not in {"gaussian", "binomial"}:
        raise ValueError(f"family must be 'gaussian' or 'binomial'; got '{family}'")
    if npcs is None:
        npcs = int(X.shape[0])

    lams = _grid(None, log_lambda_min, log_lambda_max, grid_length)

    if family == "binomial":
        # Validate labels; allow soft labels in [0,1] only for norm in {"1","2"}.
        _check_binomial_labels(Y, norm)
        if norm in {"sv", "2"}:
            return pcghal_cv_classi(
                X, Y, max_degree=max_degree, npcs=npcs,
                lambdas=lams, nfolds=nfolds, predict=predict,
                center=center, verbose=verbose,
                max_iter=max_iter, tol=tol, step_factor=step_factor,
                with_pgd=(norm == "sv"),
            )
        if norm == "1":
            return pcghal_cv_classi_lasso(
                X, Y, max_degree=max_degree, npcs=npcs,
                lambdas=lams, nfolds=nfolds, predict=predict,
                center=center, verbose=verbose,
            )
        raise ValueError(
            f"family='binomial' supports norm in {{'sv','1','2'}}; got '{norm}'"
        )

    if norm == "sv":
        return pcghal_cv(
            X, Y, max_degree=max_degree, npcs=npcs,
            lambdas=lams, nfolds=nfolds, predict=predict,
            center=center, approx=approx, verbose=verbose,
            max_iter=max_iter, tol=tol, step_factor=step_factor,
            crit=crit, ini=ini,
        )
    if norm in {"1", "2"}:
        return fasthal_cv(
            X, Y, npcs=npcs, lambdas=lams, nfolds=nfolds,
            predict=predict, max_degree=max_degree,
            center=center, approx=approx, l1=(norm == "1"),
        )
    raise ValueError(f"norm must be one of {{'sv','1','2'}}; got '{norm}'")


__all__ = [
    "CVResult",
    "pcghal_cv",
    "pcghal_cv_classi",
    "pcghal_cv_classi_lasso",
    "fasthal_cv",
    "cv_hapc",
]
