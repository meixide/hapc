"""
Single-λ HAPC fitting (high-level Python API).

This module exposes :func:`hapc` (the Python counterpart of R ``hapc()``)
together with the lower-level building blocks
:func:`single_pcghal`,
:func:`single_lambda_fit`,
:func:`single_pcghal_classification` and
:func:`single_pcghal_classification_ridge_only`.

Argument names and defaults mirror the R wrapper exactly (the only language
difference is that Python uses ``lambda_`` since ``lambda`` is a reserved
keyword).
"""

from typing import NamedTuple, Optional

import numpy as np

from . import hapc_core
from .core import (
    _C,
    OptimizerOutput,
    cross_kernel_hapc,
    design_hapc,
    fast_pchal,
    pcghal_classification,
    ridge_regression,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class SinglePcghalResult(NamedTuple):
    """Output of :func:`single_pcghal` / :func:`hapc` (gaussian, sv)."""

    alpha: np.ndarray
    predictions: Optional[np.ndarray]
    lambda_: float
    optimizer_output: OptimizerOutput
    risk: float
    iter: int


class SingleLambdaResult(NamedTuple):
    """Output of :func:`single_lambda_fit` (gaussian, norm="1"/"2")."""

    alpha: np.ndarray
    predictions: Optional[np.ndarray]
    lambda_: float


class SinglePcghalClassificationResult(NamedTuple):
    """Output of :func:`single_pcghal_classification` (binomial)."""

    alpha: np.ndarray
    predictions: Optional[np.ndarray]      # log-odds
    probabilities: Optional[np.ndarray]    # σ(log-odds)
    predicted_classes: Optional[np.ndarray]  # in {-1, +1}
    lambda_: float
    risk: float
    iter: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_xy(X: np.ndarray, Y: np.ndarray):
    X = _C(X)
    Y = _C(Y).ravel()
    n, p = X.shape
    if Y.size != n:
        raise ValueError(f"Y must have {n} elements, got {Y.size}")
    return X, Y, n, p


def _to_pm1(Y: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Convert binary {0,1} or {-1,+1} response to {-1, +1}."""
    u = np.unique(Y)
    if len(u) != 2:
        raise ValueError(
            f"Y must be binary (2 unique values); found {len(u)}: {u}"
        )
    if set(u.tolist()) == {0.0, 1.0}:
        if verbose:
            print("[hapc] Converting Y from {0,1} to {-1,+1}")
        return np.where(Y == 0, -1.0, 1.0)
    if set(u.tolist()) == {-1.0, 1.0}:
        return Y.astype(float, copy=True)
    raise ValueError(
        f"Y must contain only {{0,1}} or {{-1,+1}}; found {u}"
    )


def _calibrate_logistic_intercept(y01: np.ndarray, eta: np.ndarray) -> float:
    """Newton calibration for intercept with fixed linear predictor ``eta``."""
    y01 = np.asarray(y01, dtype=np.float64).ravel()
    eta = np.asarray(eta, dtype=np.float64).ravel()
    if y01.shape != eta.shape:
        raise ValueError("y01 and eta must have the same shape")
    b0 = 0.0
    for _ in range(50):
        z = eta + b0
        p = 1.0 / (1.0 + np.exp(-z))
        g = float(np.sum(p - y01))
        h = float(np.sum(p * (1.0 - p)))
        if abs(g) < 1e-10 or h < 1e-12:
            break
        b0 -= g / h
    return float(b0)


# ---------------------------------------------------------------------------
# Single λ — gaussian, norm in {"1", "2"} (closed-form)
# ---------------------------------------------------------------------------

def single_lambda_fit(X: np.ndarray, Y: np.ndarray,
                      max_degree: int, npcs: int,
                      lambda_: float,
                      predict: Optional[np.ndarray] = None,
                      center: bool = True, approx: bool = False,
                      l1: bool = False) -> SingleLambdaResult:
    """Closed-form gaussian fit for ``norm="1"`` (l1=True) or ``norm="2"``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Response.
    max_degree : int
        Max interaction order.
    npcs : int
        Number of PCs.
    lambda_ : float
        Regularisation parameter.
    predict : np.ndarray, optional
        Test features. If supplied, predictions are returned.
    center : bool, default True
        Centre H and Y.
    approx : bool, default False
        (Currently ignored; reserved for future power-iteration eigensolver.)
    l1 : bool, default False
        ``True`` for LASSO (norm="1"), ``False`` for ridge (norm="2").

    Returns
    -------
    SingleLambdaResult
    """
    X, Y, n, p = _check_xy(X, Y)
    cap = n - 1 if center else n
    npcs = max(1, min(int(npcs), cap))

    des = design_hapc(X, max_degree, npcs, center=center)
    final_npc = des.d.shape[0]

    # Filter near-zero singular values for numerical stability
    eps = 1e-12
    mask = des.d ** 2 > eps
    final_npc = int(min(final_npc, mask.sum()))
    U = des.U[:, :final_npc]
    D = des.d[:final_npc]
    D2 = D ** 2

    ymean = Y.mean() if center else 0.0
    Yc = Y - ymean if center else Y

    if l1:
        alpha = fast_pchal(U, D2, Yc, lambda_)
    else:
        alpha = ridge_regression(Yc, U, D2, lambda_)

    predictions = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        Ktest = cross_kernel_hapc(X, Xte, max_degree, center=center)
        v = U @ ((1.0 / (D + eps)) * alpha)
        predictions = Ktest @ v
        if center:
            predictions = predictions + ymean

    return SingleLambdaResult(alpha=alpha, predictions=predictions,
                              lambda_=float(lambda_))


# ---------------------------------------------------------------------------
# Single λ — gaussian, norm="sv"  (calls C++ single_pcghal_fit)
# ---------------------------------------------------------------------------

def single_pcghal(X: np.ndarray, Y: np.ndarray,
                  max_degree: int, npcs: int, lambda_: float,
                  predict: Optional[np.ndarray] = None,
                  center: bool = True, approx: bool = False,
                  verbose: bool = False, max_iter: int = 5000,
                  tol: float = 1e-3, step_factor: float = 0.8,
                  crit: str = "grad",
                  ini: str = "1") -> SinglePcghalResult:
    """Single-λ gaussian fit with PGD (``norm="sv"``).

    Calls C++ ``single_pcghal_fit``, which initialises α with LASSO
    (``ini="1"``, default) or ridge (``ini="2"``), then runs projected gradient
    descent.

    Parameters
    ----------
    X, Y, max_degree, npcs, lambda_, predict, center, verbose, max_iter,
    tol, step_factor, crit, ini :
        See :func:`hapc`.
    approx : bool, default False
        Currently ignored (the C++ path always uses exact eigendecomposition).
    """
    if ini not in {"1", "2"}:
        raise ValueError(f"ini must be '1' (LASSO) or '2' (ridge), got '{ini}'")
    if crit not in {"grad", "risk"}:
        raise ValueError(f"crit must be 'grad' or 'risk', got '{crit}'")

    X, Y, n, p = _check_xy(X, Y)
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
    else:
        Xte = np.empty((0, p), dtype=np.float64)

    res = hapc_core.single_pcghal_fit(
        X, Y, int(max_degree), int(npcs), float(lambda_), Xte,
        int(max_iter), float(tol), float(step_factor), bool(verbose),
        str(crit), bool(center), bool(approx), str(ini),
    )

    # Reconstruct β = V α (exposed for symmetry with the R output)
    des = design_hapc(X, max_degree, npcs, center=center)
    final_npc = des.d.shape[0]
    beta = des.V[:, :final_npc] @ res.alpha

    predictions = res.predictions if (predict is not None
                                      and res.predictions.size > 0) else None

    opt = OptimizerOutput(
        alpha=res.alpha, alphaiters=res.alpha.reshape(1, -1),
        beta=beta, risk=res.risk, iter=res.iter,
    )
    return SinglePcghalResult(
        alpha=res.alpha, predictions=predictions,
        lambda_=float(lambda_), optimizer_output=opt,
        risk=res.risk, iter=res.iter,
    )


# ---------------------------------------------------------------------------
# Single λ — binomial, norm="sv"
# ---------------------------------------------------------------------------

def single_pcghal_classification(
    X: np.ndarray, Y: np.ndarray,
    max_degree: int, npcs: int, lambda_: float,
    predict: Optional[np.ndarray] = None,
    center: bool = True, verbose: bool = False,
    max_iter: int = 5000, tol: float = 1e-3, step_factor: float = 0.8,
) -> SinglePcghalClassificationResult:
    """Single-λ binomial HAPC with ``norm="sv"`` (logistic loss + PGD).

    Matches R ``hapc(..., family="binomial", norm="sv")``: logistic ridge
    initialisation for α, then the same projected-gradient optimiser as
    ``pcghal_classi_call``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Binary labels in ``{0,1}`` or ``{-1,+1}``.
    max_degree : int
        HAL interaction order.
    npcs : int
        Number of PCs (same role as in :func:`design_hapc`).
    lambda_ : float
        Regularisation strength (passed to the ridge initialiser and CV-style
        full-data refit used in the shared C++ layer).
    predict : np.ndarray, optional, shape (m, p)
        Test features. When given, returns log-odds, probabilities, and
        thresholded classes in ``{-1,+1}``.
    center : bool, default True
        Centre the HAL design.
    verbose : bool, default False
        Print PGD diagnostics.
    max_iter : int, default 5000
        Maximum PGD iterations.
    tol : float, default 1e-3
        Convergence tolerance for PGD.
    step_factor : float, default 0.8
        PGD step-size factor.

    Returns
    -------
    SinglePcghalClassificationResult
        Fields ``alpha``, ``risk``, ``iter``, and optional ``predictions`` /
        ``probabilities`` / ``predicted_classes``.
    """

    X, Y, n, p = _check_xy(X, Y)
    Y_pm1 = _to_pm1(Y, verbose=verbose)

    des = design_hapc(X, max_degree, npcs, center=center)
    final_npc = des.d.shape[0]
    Xtilde = des.U[:, :final_npc] * des.d[:final_npc]
    ENn = des.V[:, :final_npc]

    alpha0 = np.asarray(
        hapc_core.logistic_ridge_init(_C(Y_pm1), _C(Xtilde), float(lambda_))
    ).ravel()
    res = pcghal_classification(Y_pm1, Xtilde, ENn, alpha0,
                                max_iter=max_iter, tol=tol,
                                step_factor=step_factor, verbose=verbose)
    y01 = (Y_pm1 > 0).astype(np.float64)
    eta_train = Xtilde @ np.asarray(res.alpha).ravel()
    b0 = _calibrate_logistic_intercept(y01, eta_train)
    ymu = Y_pm1 * (eta_train + b0)
    risk = float(
        np.where(ymu > 0, np.log1p(np.exp(-ymu)), -ymu + np.log1p(np.exp(ymu)))
        .mean()
    )

    predictions = probabilities = predicted_classes = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        Ktest = cross_kernel_hapc(X, Xte, max_degree, center=center)
        v = des.U[:, :final_npc] @ ((1.0 / (des.d[:final_npc] + 1e-12)) * res.alpha)
        log_odds = Ktest @ v + b0
        predictions = log_odds
        probabilities = 1.0 / (1.0 + np.exp(-log_odds))
        predicted_classes = np.where(probabilities > 0.5, 1.0, -1.0)

    return SinglePcghalClassificationResult(
        alpha=res.alpha, predictions=predictions,
        probabilities=probabilities, predicted_classes=predicted_classes,
        lambda_=float(lambda_), risk=risk, iter=res.iter,
    )


def single_pcghal_classification_ridge_only(
    X: np.ndarray, Y: np.ndarray,
    max_degree: int, npcs: int, lambda_: float,
    predict: Optional[np.ndarray] = None,
    center: bool = True, verbose: bool = False,
) -> SinglePcghalClassificationResult:
    """Binomial ``norm="2"``: logistic ridge initialiser only (no PGD).

    Mirrors R ``hapc(..., family="binomial", norm="2")`` and is intended as a
    fast linear-in-kernel baseline.

    Parameters
    ----------
    X, Y, max_degree, npcs, lambda_, predict, center, verbose
        Same meaning as in :func:`single_pcghal_classification`, except there is
        no iterative refinement and ``iter`` is always ``0``.

    Returns
    -------
    SinglePcghalClassificationResult
    """
    X, Y, n, p = _check_xy(X, Y)
    Y_pm1 = _to_pm1(Y, verbose=verbose)

    des = design_hapc(X, max_degree, npcs, center=center)
    final_npc = des.d.shape[0]
    Xtilde = des.U[:, :final_npc] * des.d[:final_npc]

    alpha = np.asarray(
        hapc_core.logistic_ridge_init(_C(Y_pm1), _C(Xtilde), float(lambda_))
    ).ravel()

    eta = Xtilde @ alpha
    y01 = (Y_pm1 > 0).astype(np.float64)
    b0 = _calibrate_logistic_intercept(y01, eta)
    ymu = Y_pm1 * (eta + b0)
    risk = float(
        np.where(ymu > 0, np.log1p(np.exp(-ymu)), -ymu + np.log1p(np.exp(ymu)))
        .mean()
    )

    predictions = probabilities = predicted_classes = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        Ktest = cross_kernel_hapc(X, Xte, max_degree, center=center)
        v = des.U[:, :final_npc] @ ((1.0 / (des.d[:final_npc] + 1e-12)) * alpha)
        log_odds = Ktest @ v + b0
        predictions = log_odds
        probabilities = 1.0 / (1.0 + np.exp(-log_odds))
        predicted_classes = np.where(probabilities > 0.5, 1.0, -1.0)

    return SinglePcghalClassificationResult(
        alpha=alpha, predictions=predictions,
        probabilities=probabilities, predicted_classes=predicted_classes,
        lambda_=float(lambda_), risk=risk, iter=0,
    )


# ---------------------------------------------------------------------------
# Single λ — binomial, norm="1"  (logistic LASSO via sklearn / liblinear)
# ---------------------------------------------------------------------------

def single_pcghal_classification_lasso(
    X: np.ndarray, Y: np.ndarray,
    max_degree: int, npcs: int, lambda_: float,
    predict: Optional[np.ndarray] = None,
    center: bool = True, verbose: bool = False,
    max_iter: int = 1000,
) -> SinglePcghalClassificationResult:
    """Single-λ binomial HAPC with ``norm="1"`` — logistic LASSO.

    Solves the L1-penalised logistic regression in the PC basis using
    :class:`sklearn.linear_model.LogisticRegression` with the
    ``liblinear`` solver. Mirrors R ``hapc(family="binomial", norm="1")``,
    which uses ``glmnet(..., family="binomial", alpha=1, intercept=FALSE)``
    on the same ``Xtilde = U * d`` design.

    Lambda parameterisation
    -----------------------
    Sklearn's L1 logistic objective with ``liblinear`` is

        min_w   ||w||_1  +  C * sum_i log(1 + exp(-y_i x_i^T w))      (1)

    while the package convention (matching ``glmnet``'s binomial loss) is

        min_w   (1/n) * sum_i log(1 + exp(-y_i x_i^T w))  +  lambda * ||w||_1   (2)

    Equating (1) and (2) up to scaling yields  ``C = 1 / (n * lambda)``,
    which is what we use here so that the same ``lambda`` produces matching
    coefficients in R (glmnet) and Python (sklearn).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Binary labels in ``{0,1}`` or ``{-1,+1}``.
    max_degree : int
        HAL interaction order.
    npcs : int
        Number of PCs.
    lambda_ : float
        L1 regularisation strength on the per-sample loss; must be > 0.
    predict : np.ndarray, optional, shape (m, p)
        Test features.
    center : bool, default True
        Centre the HAL design.
    verbose : bool, default False
        Reserved for parity with sibling routines (sklearn is silent).
    max_iter : int, default 1000
        Maximum iterations for ``liblinear``.

    Returns
    -------
    SinglePcghalClassificationResult
        ``alpha`` is the LASSO coefficient on the PC basis,
        ``risk`` is the training logistic loss,
        ``iter`` is the iteration count returned by ``liblinear``,
        and ``predictions`` / ``probabilities`` / ``predicted_classes`` are
        populated when ``predict`` is supplied.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for binomial + norm='1' (logistic LASSO). "
            "Install with: pip install scikit-learn"
        ) from e

    if not (lambda_ > 0):
        raise ValueError(f"lambda_ must be > 0 for LASSO; got {lambda_}")

    X, Y, n, p = _check_xy(X, Y)
    Y_pm1 = _to_pm1(Y, verbose=verbose)
    Y_01 = (Y_pm1 > 0).astype(np.int64)

    des = design_hapc(X, max_degree, npcs, center=center)
    final_npc = des.d.shape[0]
    Xtilde = des.U[:, :final_npc] * des.d[:final_npc]

    C = 1.0 / (n * float(lambda_))
    # sklearn>=1.8 deprecated penalty="l1" in favour of l1_ratio=1 with the
    # liblinear solver; older versions still need penalty="l1". Try the new
    # API first and fall back gracefully.
    import inspect
    sig_params = inspect.signature(LogisticRegression).parameters
    common_kw = dict(solver="liblinear", C=C, fit_intercept=False,
                     max_iter=int(max_iter))
    if "l1_ratio" in sig_params and "penalty" in sig_params:
        try:
            model = LogisticRegression(l1_ratio=1.0, **common_kw)
            model.fit(_C(Xtilde), Y_01)
        except (TypeError, ValueError):
            model = LogisticRegression(penalty="l1", **common_kw)
            model.fit(_C(Xtilde), Y_01)
    else:  # pragma: no cover  (very old sklearn)
        model = LogisticRegression(penalty="l1", **common_kw)
        model.fit(_C(Xtilde), Y_01)
    alpha = np.asarray(model.coef_, dtype=np.float64).ravel()
    b0 = _calibrate_logistic_intercept(Y_01.astype(np.float64), Xtilde @ alpha)

    eta = Xtilde @ alpha + b0
    ymu = Y_pm1 * eta
    risk = float(
        np.where(ymu > 0, np.log1p(np.exp(-ymu)), -ymu + np.log1p(np.exp(ymu))).mean()
    )

    predictions = probabilities = predicted_classes = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        Ktest = cross_kernel_hapc(X, Xte, max_degree, center=center)
        v = des.U[:, :final_npc] @ ((1.0 / (des.d[:final_npc] + 1e-12)) * alpha)
        log_odds = Ktest @ v + b0
        predictions = log_odds
        probabilities = 1.0 / (1.0 + np.exp(-log_odds))
        predicted_classes = np.where(probabilities > 0.5, 1.0, -1.0)

    n_iter = int(np.asarray(model.n_iter_).ravel()[0])
    return SinglePcghalClassificationResult(
        alpha=alpha, predictions=predictions,
        probabilities=probabilities, predicted_classes=predicted_classes,
        lambda_=float(lambda_), risk=risk, iter=n_iter,
    )


# ---------------------------------------------------------------------------
# High-level dispatcher (Python counterpart of R `hapc()`)
# ---------------------------------------------------------------------------

def hapc(X: np.ndarray, Y: np.ndarray,
         family: str = "gaussian",
         max_degree: int = 1,
         npcs: Optional[int] = None,
         lambda_: float = 0.01,
         norm: str = "sv",
         predict: Optional[np.ndarray] = None,
         max_iter: int = 5000,
         tol: float = 1e-3,
         step_factor: float = 0.8,
         verbose: bool = False,
         crit: str = "grad",
         center: bool = True,
         approx: bool = False,
         ini: str = "1"):
    """Single-λ HAPC fit (Python counterpart of R ``hapc()``).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Response. For ``family="binomial"`` must contain only ``{0,1}`` or
        ``{-1,+1}``.
    family : {"gaussian", "binomial"}, default "gaussian"
        Loss family.
    max_degree : int, default 1
        Maximum interaction order for the HAL basis.
    npcs : int, optional
        Number of principal components. Defaults to ``n``.
    lambda_ : float, default 0.01
        Regularisation parameter.
    norm : {"sv", "1", "2"}, default "sv"
        Norm constraint. ``"sv"`` runs PGD, ``"1"`` is closed-form LASSO,
        ``"2"`` is closed-form ridge.
    predict : np.ndarray, optional
        Test features for prediction.
    max_iter : int, default 5000
        Maximum PGD iterations (``norm="sv"`` only).
    tol : float, default 1e-3
        Convergence tolerance (``norm="sv"`` only).
    step_factor : float, default 0.8
        PGD step factor (``norm="sv"`` only).
    verbose : bool, default False
        Print progress (``norm="sv"`` only).
    crit : {"grad", "risk"}, default "grad"
        PGD stopping criterion.
    center : bool, default True
        Centre H and Y.
    approx : bool, default False
        Use approximate eigendecomposition for ``norm in {"1","2"}`` (currently
        ignored — exact eigendecomposition is always used).
    ini : {"1", "2"}, default "1"
        PGD initialiser for ``norm="sv"``: ``"1"`` = LASSO (default, matches R),
        ``"2"`` = ridge.

    Returns
    -------
    object
        One of
        :class:`SinglePcghalResult` (gaussian ``norm="sv"``),
        :class:`SingleLambdaResult` (gaussian ``norm`` in ``{"1","2"}``), or
        :class:`SinglePcghalClassificationResult` (binomial).

    Examples
    --------
    >>> import numpy as np
    >>> from hapc import hapc
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 3))
    >>> Y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(50) * 0.1
    >>> fit = hapc(X, Y, max_degree=2, npcs=20, lambda_=0.05, norm="sv")
    >>> fit.alpha.shape  # doctest: +ELLIPSIS
    (...,)
    """
    if npcs is None:
        npcs = int(X.shape[0])

    if family == "binomial":
        if norm == "sv":
            return single_pcghal_classification(
                X, Y, max_degree, npcs, lambda_,
                predict=predict, center=center, verbose=verbose,
                max_iter=max_iter, tol=tol, step_factor=step_factor,
            )
        if norm == "2":
            return single_pcghal_classification_ridge_only(
                X, Y, max_degree, npcs, lambda_,
                predict=predict, center=center, verbose=verbose,
            )
        if norm == "1":
            return single_pcghal_classification_lasso(
                X, Y, max_degree, npcs, lambda_,
                predict=predict, center=center, verbose=verbose,
            )
        raise ValueError(
            f"family='binomial' supports norm in {{'sv','1','2'}}; got '{norm}'"
        )

    if family != "gaussian":
        raise ValueError(f"family must be 'gaussian' or 'binomial'; got '{family}'")

    if norm == "sv":
        return single_pcghal(X, Y, max_degree, npcs, lambda_,
                             predict=predict, center=center, approx=approx,
                             verbose=verbose, max_iter=max_iter, tol=tol,
                             step_factor=step_factor, crit=crit, ini=ini)
    if norm in {"1", "2"}:
        return single_lambda_fit(X, Y, max_degree, npcs, lambda_,
                                 predict=predict, center=center,
                                 approx=approx, l1=(norm == "1"))
    raise ValueError(f"norm must be one of {{'sv','1','2'}}; got '{norm}'")


__all__ = [
    "SinglePcghalResult",
    "SingleLambdaResult",
    "SinglePcghalClassificationResult",
    "single_lambda_fit",
    "single_pcghal",
    "single_pcghal_classification",
    "single_pcghal_classification_ridge_only",
    "single_pcghal_classification_lasso",
    "hapc",
]
