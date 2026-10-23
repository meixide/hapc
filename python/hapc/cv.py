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
    _check_intercept,
    _lasso_basis,
    _lasso_path,
    _lasso_predictor,
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
    # binomial norm="1" only (None elsewhere):
    best_index: Optional[int] = None
    lambda_position: Optional[str] = None  # interior / lower_edge / upper_edge / null_model
    n_extend: Optional[int] = None
    lambda_max: Optional[float] = None
    intercept: Optional[float] = None


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
# norm="1" (binomial)  →  glmnet-style logistic LASSO CV
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
                            max_iter: int = 100,
                            extend_grid: int = 0) -> CVResult:
    """k-fold logistic-LASSO CV for binomial HAPC, ``norm="1"``.

    Per fold, the kernel PC basis is computed once and the L1 logistic
    regression with unpenalised intercept is fitted along the whole λ grid
    with warm starts; held-out folds are scored with cross-entropy. Mirrors R
    ``cv.hapc(family="binomial", norm="1")`` (glmnet per fold), including
    ``extend_grid`` and the ``lambda_position`` report.

    Parameters
    ----------
    See :func:`cv_hapc`.

    Returns
    -------
    CVResult
        ``mses`` holds mean per-fold cross-entropy. ``best_model_alpha`` is the
        α refit on the full data at ``best_lambda`` (``intercept`` its
        intercept). ``predictions`` are probabilities at ``best_lambda`` on
        ``predict``. ``lambda_position`` is ``"interior"``, ``"lower_edge"``,
        ``"upper_edge"`` or ``"null_model"`` (optimum at ``lambda_max``).
    """
    X = _C(X)
    Y = _C(Y).ravel()
    n, p = X.shape
    if Y.size != n:
        raise ValueError(f"Y must have {n} elements, got {Y.size}")

    lams = np.sort(_grid(lambdas, log_lambda_min, log_lambda_max, grid_length))
    if not np.all(lams > 0):
        raise ValueError("All lambdas must be > 0 for logistic LASSO.")
    L0 = lams.size

    # Soft target in [0,1] for the fit and the held-out cross-entropy.
    q = _to_soft01(Y)
    folds = _native_folds(n, int(nfolds))

    fold_data = []
    for k in range(1, int(nfolds) + 1):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        if te.size == 0 or tr.size == 0:
            fold_data.append(None)
            continue
        basis = _lasso_basis(X[tr], int(max_degree), int(npcs), bool(center))
        fold_data.append(dict(Xt=basis.xtilde, q=q[tr], q_te=q[te],
                              P=_lasso_predictor(basis, X[te])))

    def cv_dev(ls: np.ndarray) -> np.ndarray:
        dev = np.full((len(fold_data), ls.size), np.nan)
        for f, fd in enumerate(fold_data):
            if fd is None:
                continue
            B, b0, _ = _lasso_path(fd["Xt"], fd["q"], ls, max_iter)
            eta = fd["P"] @ B + b0[None, :]
            pr = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1 - 1e-15)
            qt = fd["q_te"][:, None]
            dev[f] = (-(qt * np.log(pr) + (1 - qt) * np.log(1 - pr))).mean(axis=0)
        return np.nanmean(dev, axis=0)

    deviances = cv_dev(lams)

    # lambda_max: at the intercept-only fit the gradient of the mean loss
    # w.r.t. alpha is Xt'(q - mean(q)) / n; the largest over folds.
    lambda_max = max(
        float(np.abs(fd["Xt"].T @ (fd["q"] - fd["q"].mean())).max()) / fd["q"].size
        for fd in fold_data if fd is not None
    )
    step = float(np.log(lams[1]) - np.log(lams[0])) if L0 > 1 else float(np.log(10))
    n_extend = 0
    while n_extend < int(extend_grid):
        b = int(np.nanargmin(deviances))
        if b == 0:
            extra = np.exp(np.log(lams[0]) - step * np.arange(L0, 0, -1))
            deviances = np.concatenate([cv_dev(extra), deviances])
            lams = np.concatenate([extra, lams])
        elif b == lams.size - 1 and lams.size > 1 and lams[-1] < lambda_max:
            extra = np.exp(np.log(lams[-1]) + step * np.arange(1, L0 + 1))
            extra = np.append(extra[extra < lambda_max], lambda_max)
            deviances = np.concatenate([deviances, cv_dev(extra)])
            lams = np.concatenate([lams, extra])
        else:
            break
        n_extend += 1

    best_idx = int(np.nanargmin(deviances))
    best_lambda = float(lams[best_idx])
    L = lams.size
    if L == 1:
        position = None
    elif best_lambda >= lambda_max * (1 - 1e-8):
        position = "null_model"
    elif best_idx == 0:
        position = "lower_edge"
    elif best_idx == L - 1:
        position = "upper_edge"
    else:
        position = "interior"
    if int(extend_grid) > 0 and position in ("lower_edge", "upper_edge"):
        import warnings
        warnings.warn(
            f"cv_hapc(family='binomial', norm='1'): the CV-optimal lambda is still "
            f"on the {position.split('_')[0]} edge of the grid after {n_extend} "
            "extension(s).", stacklevel=2)

    full = single_pcghal_classification_lasso(
        X, Y, max_degree=int(max_degree), npcs=int(npcs),
        lambda_=best_lambda, predict=predict, center=bool(center),
        verbose=bool(verbose), max_iter=int(max_iter),
    )

    return CVResult(
        mses=np.asarray(deviances),
        lambdas=np.asarray(lams),
        best_lambda=best_lambda,
        best_model_alpha=np.asarray(full.alpha),
        predictions=full.predictions if predict is not None else None,
        best_index=best_idx,
        lambda_position=position,
        n_extend=n_extend,
        lambda_max=float(lambda_max),
        intercept=full.intercept,
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
            ini: str = "1",
            Delta: Optional[np.ndarray] = None,
            time_grid: Optional[np.ndarray] = None,
            intercept: Optional[bool] = None,
            extend_grid: int = 0) -> CVResult:
    """k-fold cross-validated HAPC fit (Python counterpart of R ``cv.hapc()``).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Response.
    family : {"gaussian", "binomial", "logit-hazard"}, default "gaussian"
        Loss family. For ``"binomial"`` the loss is **always logistic** (deviance
        CV, not MSE). ``norm="sv"`` runs logistic ridge + PGD per fold;
        ``norm="2"`` runs logistic ridge only; ``norm="1"`` runs logistic
        LASSO with an unpenalised intercept by glmnet's algorithm (mirrors R
        ``glmnet`` for the binomial+norm="1" path). Every binomial fit has an
        unpenalised intercept. ``"logit-hazard"`` is a discrete-time logistic hazard model: ``Y``
        is then the observed time ``T = min(T_event, C)``, ``Delta`` (event
        indicator) is required, and the call forwards to
        :func:`hapc.hazard_hapc` (returning a :class:`hapc.hazard.HazardResult`,
        not a :class:`CVResult`). ``norm="sv"`` is not supported for this family.
    Delta : np.ndarray, optional
        Event indicator (0/1), required only for ``family="logit-hazard"``.
    time_grid : np.ndarray, optional
        Discrete time grid for ``family="logit-hazard"`` (see
        :func:`hapc.hazard_hapc`).
    intercept : bool or None, default None
        Every binomial fit includes an unpenalised intercept estimated jointly
        with the PC coefficients, so ``intercept=False`` raises for
        ``family="binomial"`` / ``"logit-hazard"``. For ``family="gaussian"``
        the intercept is fitted exactly when ``center=True``; an explicit value
        contradicting ``center`` raises. ``None`` means "the intercept the
        model fits".
    extend_grid : int, default 0
        ``family="binomial"``, ``norm="1"`` only: if the CV-optimal λ is on an
        edge of the grid, extend the grid on that side (same log spacing,
        another ``grid_length`` points) up to ``extend_grid`` times; upwards
        never past ``lambda_max`` (the intercept-only fit). Mirrors R.
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
    _check_intercept(intercept, family, center)
    # family = "logit-hazard": discrete-time logistic hazard. The second
    # positional argument Y carries the observed times T = min(T^event, C); the
    # event indicator Delta must be supplied. norm = "sv" is flagged inside
    # hazard_hapc. See hapc.hazard_hapc.
    if family == "logit-hazard":
        if Delta is None:
            raise ValueError(
                "family='logit-hazard' requires 'Delta' (event indicator); pass "
                "the observed times T as Y and Delta separately. See hazard_hapc."
            )
        from .hazard import hazard_hapc  # local import avoids circular import
        return hazard_hapc(
            X, T=Y, Delta=Delta, norm=norm,
            max_degree=max_degree, npcs=npcs, time_grid=time_grid,
            log_lambda_min=log_lambda_min, log_lambda_max=log_lambda_max,
            grid_length=grid_length, nfolds=nfolds, predict=predict,
            center=center, verbose=verbose,
            max_iter=max_iter, tol=tol, step_factor=step_factor,
        )

    if family not in {"gaussian", "binomial"}:
        raise ValueError(
            f"family must be 'gaussian', 'binomial', or 'logit-hazard'; got '{family}'"
        )
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
                center=center, verbose=verbose, extend_grid=extend_grid,
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
