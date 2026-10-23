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
    kernel_hapc,
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
    predictions: Optional[np.ndarray]      # probabilities on ``predict``
    log_odds: Optional[np.ndarray]         # linear predictor on ``predict``
    lambda_: float
    risk: float
    iter: int
    intercept: Optional[float] = None      # unpenalised intercept b0


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


def _label_kind(Y: np.ndarray) -> str:
    """Classify a binomial response vector.

    Returns ``"01"`` (hard labels in ``{0,1}``), ``"pm1"`` (hard labels in
    ``{-1,+1}``), or ``"soft"`` (fractional labels in ``[0,1]``, e.g. EM-HAL
    E-step posteriors). Raises ``ValueError`` if any value falls outside
    ``[0,1]`` and the set is not exactly ``{-1,+1}``.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    u = np.unique(Y[~np.isnan(Y)])
    s = set(u.tolist())
    if s.issubset({0.0, 1.0}):
        return "01"
    if s == {-1.0, 1.0}:
        return "pm1"
    if u.size and u.min() >= 0.0 and u.max() <= 1.0:
        return "soft"
    raise ValueError(
        "family='binomial' requires Y in {0,1}, {-1,+1}, or soft labels in "
        "[0,1]; found values outside [0,1]."
    )


def _to_soft01(Y: np.ndarray) -> np.ndarray:
    """Map a binomial response to a soft cross-entropy target in ``[0,1]``."""
    Y = np.asarray(Y, dtype=np.float64).ravel()
    return (Y + 1.0) / 2.0 if _label_kind(Y) == "pm1" else Y


def _check_binomial_labels(Y: np.ndarray, norm: str) -> str:
    """Validate labels and enforce the soft-label norm restriction.

    Soft labels (any value strictly inside ``(0,1)``) are supported only for
    ``norm`` in ``{"1","2"}``; ``norm="sv"`` raises ``NotImplementedError``.
    A warning is emitted whenever soft labels are detected. Returns the label
    kind from :func:`_label_kind`.
    """
    import warnings

    kind = _label_kind(Y)
    if kind == "soft":
        if norm == "sv":
            raise NotImplementedError(
                "Soft labels (Y in (0,1)) are not implemented for norm='sv'; "
                "use norm='1' or norm='2'."
            )
        warnings.warn(
            "Non-binary labels detected in Y: treating them as soft labels in "
            "[0,1] (cross-entropy target). Supported only for norm='1' and "
            "norm='2'.",
            stacklevel=2,
        )
    return kind


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
    X, Y, max_degree, npcs, lambda_, predict, center, verbose, max_iter, tol, step_factor, crit, ini :
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
        Test features. When given, returns probabilities in ``predictions``
        and the linear predictor in ``log_odds``.
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
        Fields ``alpha``, ``risk``, ``iter``, ``intercept`` and, when
        ``predict`` is given, ``predictions`` (probabilities) and ``log_odds``.
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

    predictions = log_odds = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        Ktest = cross_kernel_hapc(X, Xte, max_degree, center=center)
        v = des.U[:, :final_npc] @ ((1.0 / (des.d[:final_npc] + 1e-12)) * res.alpha)
        log_odds = Ktest @ v + b0
        predictions = 1.0 / (1.0 + np.exp(-log_odds))

    return SinglePcghalClassificationResult(
        alpha=res.alpha, predictions=predictions, log_odds=log_odds,
        lambda_=float(lambda_), risk=risk, iter=res.iter, intercept=float(b0),
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
    # Accept hard {0,1}/{-1,+1} or soft [0,1] labels (cross-entropy target).
    y01 = _to_soft01(Y)

    des = design_hapc(X, max_degree, npcs, center=center)
    final_npc = des.d.shape[0]
    Xtilde = des.U[:, :final_npc] * des.d[:final_npc]

    alpha = np.asarray(
        hapc_core.logistic_ridge_init_y01(_C(y01), _C(Xtilde), float(lambda_))
    ).ravel()

    eta = Xtilde @ alpha
    b0 = _calibrate_logistic_intercept(y01, eta)
    phat = np.clip(1.0 / (1.0 + np.exp(-(eta + b0))), 1e-15, 1 - 1e-15)
    risk = float((-(y01 * np.log(phat) + (1 - y01) * np.log(1 - phat))).mean())

    predictions = log_odds = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        Ktest = cross_kernel_hapc(X, Xte, max_degree, center=center)
        v = des.U[:, :final_npc] @ ((1.0 / (des.d[:final_npc] + 1e-12)) * alpha)
        log_odds = Ktest @ v + b0
        predictions = 1.0 / (1.0 + np.exp(-log_odds))

    return SinglePcghalClassificationResult(
        alpha=alpha, predictions=predictions, log_odds=log_odds,
        lambda_=float(lambda_), risk=risk, iter=0, intercept=float(b0),
    )


# ---------------------------------------------------------------------------
# Single λ — binomial, norm="1"  (logistic LASSO, glmnet's algorithm)
# ---------------------------------------------------------------------------

def _check_intercept(intercept: Optional[bool], family: str, center: bool) -> None:
    """Validate ``intercept`` (mirrors R ``.hapc_check_intercept``).

    Every binomial fit has an unpenalised intercept estimated jointly with the
    PC coefficients; a gaussian fit has an intercept exactly when
    ``center=True`` (the response is centred). ``None`` means "whatever the
    model fits"; an explicit value the model cannot honour raises instead of
    being silently ignored.
    """
    if intercept is None:
        return
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError("intercept must be None, True or False")
    if family in ("binomial", "logit-hazard") and not intercept:
        raise ValueError(
            f"intercept=False is not supported for family='{family}': the "
            "intercept is always estimated jointly and left unpenalised "
            "(fitting with the intercept fixed at 0 centres the logistic fit at "
            "p = 1/2 instead of the prevalence). Leave 'intercept' unset."
        )
    if family == "gaussian" and bool(intercept) != bool(center):
        raise ValueError(
            "intercept=True requires center=True for family='gaussian': "
            "without centring no intercept is fitted."
            if intercept else
            "intercept=False is incompatible with center=True for "
            "family='gaussian': centring fits the intercept (the mean of Y). "
            "Use center=False for a fit without intercept."
        )


class _LassoBasis(NamedTuple):
    """PC basis of the HAL design built from the n x n kernel (norm="1")."""

    X: np.ndarray
    U: np.ndarray          # (n, k) eigenvectors, sign-normalised
    d: np.ndarray          # (k,) singular values of H
    keep: np.ndarray       # (k,) bool, numerically non-null directions
    colmean: np.ndarray    # column means of the uncentred kernel
    grand: float           # grand mean of the uncentred kernel
    max_degree: int
    center: bool

    @property
    def xtilde(self) -> np.ndarray:
        return self.U[:, self.keep] * self.d[self.keep]


def _lasso_basis(X: np.ndarray, max_degree: int, npcs: int,
                 center: bool) -> _LassoBasis:
    """Kernel PC basis, as R ``.hapc_kernel_basis``.

    U and d are the left singular vectors / singular values of the (centred)
    HAL design H, from the eigendecomposition of K = H H^T, so H is never
    materialised. Same npcs clamp as :func:`design_hapc` (n - 1 if centred,
    else n). Eigenvector signs are solver dependent: each column's
    largest-|.| entry is made positive (sklearn's svd_flip convention), the
    rule R applies too. Directions with d <= 1e-7 * d[0] are excluded. The
    uncentred kernel's column means are kept so that predictions can centre
    ``cross_kernel_hapc(center=False)`` directly instead of recomputing the
    training kernel.
    """
    n = X.shape[0]
    K = kernel_hapc(X, max_degree, center=False)
    colmean = K.mean(axis=0)
    grand = float(colmean.mean())
    if center:
        K = K - colmean[:, None] - colmean[None, :] + grand
    evals, evecs = np.linalg.eigh(K)
    del K
    order = np.argsort(evals)[::-1]
    k = max(1, min(int(npcs), n - 1 if center else n))
    idx = order[:k]
    d = np.sqrt(np.clip(evals[idx], 0.0, None))
    U = np.array(evecs[:, idx], dtype=np.float64)
    signs = np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(k)])
    signs[signs == 0] = 1.0
    U *= signs
    return _LassoBasis(X=X, U=U, d=d, keep=d > 1e-7 * d[0], colmean=colmean,
                       grand=grand, max_degree=int(max_degree),
                       center=bool(center))


def _lasso_predictor(basis: _LassoBasis, Xte: np.ndarray) -> np.ndarray:
    """Map from kept PC coefficients to log-odds at ``Xte`` (minus b0)."""
    Kt = cross_kernel_hapc(basis.X, Xte, basis.max_degree, center=False)
    if basis.center:
        Kt = (Kt - Kt.mean(axis=1, keepdims=True)
              - basis.colmean[None, :] + basis.grand)
    return Kt @ (basis.U[:, basis.keep] / basis.d[basis.keep])


def _logistic_lasso_cd(Xt: np.ndarray, q: np.ndarray, lam: float,
                       beta: np.ndarray, b0: float,
                       tol: float = 1e-10, max_outer: int = 100,
                       max_sweeps: int = 1000):
    """L1 logistic regression with unpenalised intercept (glmnet's algorithm).

    Minimises ``(1/n) sum_i CE(q_i, b0 + x_i' beta) + lam * ||beta||_1`` for
    soft targets ``q`` in [0, 1] by proximal Newton: each outer step builds the
    IRLS quadratic approximation and solves the weighted LASSO by cyclic
    coordinate descent (soft-thresholding on beta, plain update on b0).
    Warm-started from ``(beta, b0)``. Returns ``(beta, b0, n_outer)``.
    """
    n, k = Xt.shape
    beta = beta.astype(np.float64).copy()
    for outer in range(1, max_outer + 1):
        eta = b0 + Xt @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-5)
        z = eta + (q - p) / w
        r = z - eta                          # working residual
        wsum = w.sum()
        v = (w[:, None] * Xt * Xt).sum(axis=0) / n
        beta_old, b0_old = beta.copy(), b0
        for _ in range(max_sweeps):
            dmax = 0.0
            db = float((w * r).sum() / wsum)
            b0 += db
            r -= db
            dmax = max(dmax, abs(db))
            for j in range(k):
                if v[j] <= 0.0:
                    continue
                xj = Xt[:, j]
                g = float((w * xj) @ r) / n + v[j] * beta[j]
                new = np.sign(g) * max(abs(g) - lam, 0.0) / v[j]
                d = new - beta[j]
                if d != 0.0:
                    r -= d * xj
                    beta[j] = new
                    dmax = max(dmax, abs(d) * np.sqrt(v[j]))
            if dmax < tol:
                break
        change = max(abs(b0 - b0_old),
                     float(np.max(np.abs(beta - beta_old) * np.sqrt(v))) if k else 0.0)
        if change < 1e-8:
            break
    return beta, b0, outer


def _lasso_path(Xt: np.ndarray, q: np.ndarray, lambdas: np.ndarray,
                max_iter: int = 100):
    """L1 logistic fits with unpenalised intercept along ``lambdas``.

    Returns ``(B, b0, n_iter)`` with B of shape (ncol(Xt), len(lambdas)),
    columns in the order of ``lambdas``. Objective (glmnet's):
    ``(1/n) sum_i CE(q_i, eta_i) + lambda ||alpha||_1``; soft labels q in
    [0, 1] enter the cross-entropy directly. Fits run from the largest lambda
    down, each warm-started from the previous one, starting from the
    intercept-only fit ``b0 = logit(mean(q))``.
    """
    lambdas = np.asarray(lambdas, dtype=np.float64)
    L = lambdas.size
    B = np.zeros((Xt.shape[1], L))
    b0s = np.zeros(L)
    n_iter = np.zeros(L, dtype=int)
    qbar = float(np.clip(q.mean(), 1e-10, 1 - 1e-10))
    beta, b0 = np.zeros(Xt.shape[1]), float(np.log(qbar / (1 - qbar)))
    for j in np.argsort(-lambdas):
        beta, b0, it = _logistic_lasso_cd(Xt, q, float(lambdas[j]), beta, b0,
                                          max_outer=int(max_iter))
        B[:, j], b0s[j], n_iter[j] = beta, b0, it
    return B, b0s, n_iter


def single_pcghal_classification_lasso(
    X: np.ndarray, Y: np.ndarray,
    max_degree: int, npcs: int, lambda_: float,
    predict: Optional[np.ndarray] = None,
    center: bool = True, verbose: bool = False,
    max_iter: int = 100,
) -> SinglePcghalClassificationResult:
    """Single-λ binomial HAPC with ``norm="1"`` — logistic LASSO.

    Solves the L1-penalised logistic regression in the PC basis with an
    unpenalised intercept by glmnet's algorithm (IRLS + coordinate descent,
    :func:`_logistic_lasso_cd`). Mirrors R ``hapc(family="binomial", norm="1")``,
    which calls ``glmnet(..., family="binomial", alpha=1)`` on the same
    ``Xtilde = U * d`` design.

    ``U`` and ``d`` come from the eigendecomposition of the n x n HAL kernel
    (:func:`kernel_hapc`), so the HAL design itself is never materialised.

    **Intercept.** The intercept is estimated jointly and never penalised.
    Centring makes the PC columns orthogonal to 1, which removes the
    intercept from least squares but not from logistic regression: fitting
    with the intercept fixed at 0 (hapc <= 2.6.0) centres the fit at
    p = 1/2 instead of the prevalence.

    **Lambda parameterisation.** The package convention matches ``glmnet``:

    .. code-block:: text

        min_{b0, w}   (1/n) * sum_i CE(y_i, b0 + x_i^T w)  +  lambda * ||w||_1



    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Labels in ``{0,1}`` or ``{-1,+1}``, or soft labels in ``[0,1]``.
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
        Reserved for parity with sibling routines.
    max_iter : int, default 100
        Maximum outer (IRLS) iterations.

    Returns
    -------
    SinglePcghalClassificationResult
        ``alpha`` (PC coefficients), ``intercept``, ``risk`` (training
        cross-entropy), ``iter`` (IRLS iterations) and, when ``predict`` is
        supplied, ``predictions`` (probabilities) and ``log_odds``.
    """
    if not (lambda_ > 0):
        raise ValueError(f"lambda_ must be > 0 for LASSO; got {lambda_}")

    X, Y, n, p = _check_xy(X, Y)
    # Accept hard {0,1}/{-1,+1} or soft [0,1] labels (cross-entropy target).
    q = _to_soft01(Y)

    basis = _lasso_basis(X, max_degree, npcs, center)
    Xt = basis.xtilde
    B, b0s, n_iter = _lasso_path(Xt, q, np.array([float(lambda_)]), max_iter)
    alpha_keep, b0 = B[:, 0], float(b0s[0])
    alpha = np.zeros(basis.d.shape[0])
    alpha[basis.keep] = alpha_keep

    eta = Xt @ alpha_keep + b0
    phat = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1 - 1e-15)
    risk = float((-(q * np.log(phat) + (1 - q) * np.log(1 - phat))).mean())

    predictions = log_odds = None
    if predict is not None:
        Xte = _C(predict)
        if Xte.shape[1] != p:
            raise ValueError(f"predict must have {p} columns")
        log_odds = _lasso_predictor(basis, Xte) @ alpha_keep + b0
        predictions = 1.0 / (1.0 + np.exp(-log_odds))

    return SinglePcghalClassificationResult(
        alpha=alpha, predictions=predictions, log_odds=log_odds,
        lambda_=float(lambda_), risk=risk, iter=int(n_iter[0]), intercept=b0,
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
         ini: str = "1",
         intercept: Optional[bool] = None):
    """Single-λ HAPC fit (Python counterpart of R ``hapc()``).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Features.
    Y : np.ndarray, shape (n,)
        Response. For ``family="binomial"``: hard labels in ``{0,1}`` or
        ``{-1,+1}``, or soft labels in ``[0,1]`` (e.g. EM-HAL E-step
        posteriors). Soft labels are supported only for ``norm`` in
        ``{"1","2"}``; ``norm="sv"`` requires hard labels.
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
    intercept : bool or None, default None
        Every binomial fit includes an unpenalised intercept estimated jointly
        with the PC coefficients, so ``intercept=False`` raises for
        ``family="binomial"``. For ``family="gaussian"`` the intercept is
        fitted exactly when ``center=True``; an explicit value contradicting
        ``center`` raises. ``None`` means "the intercept the model fits".

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
    _check_intercept(intercept, family, center)
    if family == "logit-hazard":
        raise ValueError(
            "family='logit-hazard' is a cross-validated discrete-time hazard "
            "model; use hazard_hapc(X, T, Delta, ...) or "
            "cv_hapc(..., family='logit-hazard') instead of the single-lambda hapc()."
        )

    if npcs is None:
        npcs = int(X.shape[0])

    if family == "binomial":
        # Validate labels; allow soft labels in [0,1] only for norm in {"1","2"}.
        _check_binomial_labels(Y, norm)
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
