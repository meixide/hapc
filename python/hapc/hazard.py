"""Discrete-time logistic hazard via HAPC (``family="logit-hazard"``).

Provides :func:`hazard_hapc`, the Python counterpart of R ``hazard.hapc()``.
It is a thin wrapper around :func:`hapc.cv_hapc` with ``family="binomial"``:
right-censored survival data ``(X, T, Delta)`` are expanded into a
person-period table (one row per subject-per-interval-at-risk) whose binary
response is the discrete hazard indicator, with the visit time prepended as the
first HAL covariate.

Data contract
-------------
For subject ``i`` let ``T_event_i`` be the latent event time and ``C_i`` the
censoring time. The user supplies only the observed quantities

    T_i     = min(T_event_i, C_i)        # observed time
    Delta_i = 1(T_event_i <= C_i)        # event indicator

together with baseline covariates ``X``. Times are assumed *discrete* (interval
/ visit indices). Each subject contributes one person-period row for every grid
time ``g <= T_i``; the hazard label is ``1`` only at the event interval
(``g == T_i`` and ``Delta_i == 1``) and ``0`` otherwise (including the last
interval of censored subjects).

``norm="sv"`` is **not implemented** for this family and raises
``NotImplementedError``.
"""

from typing import NamedTuple, Optional

import numpy as np

from .core import _C, design_hapc
from .cv import CVResult, cv_hapc
from .single import hapc as _hapc


class HazardResult(NamedTuple):
    """Output of :func:`hazard_hapc`.

    Attributes
    ----------
    hazard : np.ndarray, shape (N,)
        Estimated discrete hazard for each person-period row (aligned with
        ``ids``/``times``); the cross-validated predictions at the winning λ.
    ids : np.ndarray, shape (N,)
        Subject index (0-based) for each person-period row.
    times : np.ndarray, shape (N,)
        Grid time for each person-period row.
    Y : np.ndarray, shape (N,)
        Binary hazard label for each person-period row.
    time_grid : np.ndarray, shape (K,)
        The discrete time grid used.
    lambdas : np.ndarray, shape (L,)
        CV λ grid.
    risk : np.ndarray, shape (L,)
        Mean cross-validated logistic deviance per λ.
    best_lambda : float
        Deviance-minimising λ.
    interior : bool
        ``True`` when ``best_lambda`` is strictly inside the grid (not at either
        endpoint) — a basic check that the grid brackets the optimum.
    cv : CVResult
        The full underlying :func:`hapc.cv_hapc` result.
    predict_hazard : np.ndarray or None, shape (m, K)
        Hazard surface for new subjects (only when ``predict`` is supplied).
    predict_survival : np.ndarray or None, shape (m, K)
        Survival curves ``S(t|x) = prod_{g<=t}(1 - hazard(g|x))`` for new
        subjects (only when ``predict`` is supplied).
    """

    hazard: np.ndarray
    ids: np.ndarray
    times: np.ndarray
    Y: np.ndarray
    time_grid: np.ndarray
    lambdas: np.ndarray
    risk: np.ndarray
    best_lambda: float
    interior: bool
    cv: CVResult
    predict_hazard: Optional[np.ndarray]
    predict_survival: Optional[np.ndarray]


def _infer_time_grid(T: np.ndarray) -> np.ndarray:
    """Default discrete grid: ``min:max`` for integer times, else unique values."""
    T = np.asarray(T, dtype=np.float64).ravel()
    if np.all(np.abs(T - np.round(T)) < 1e-9):
        return np.arange(int(round(T.min())), int(round(T.max())) + 1,
                         dtype=np.float64)
    return np.unique(T)


def _person_period(X: np.ndarray, T: np.ndarray, Delta: np.ndarray,
                   grid: np.ndarray):
    """Expand ``(X, T, Delta)`` into the person-period design.

    Returns ``ids``, ``times``, the ``[time, X]`` design matrix ``Xpp`` and the
    binary hazard response ``Y``. Subject ``i`` contributes one row per grid
    time ``g <= T_i``; ``Y = 1`` only at the event interval.
    """
    n = X.shape[0]
    ids, times, Xrows, Y = [], [], [], []
    for i in range(n):
        gi = grid[grid <= T[i]]
        if gi.size == 0:
            gi = grid[:1]
        k = gi.size
        ids.append(np.full(k, i, dtype=np.int64))
        times.append(gi)
        Xrows.append(np.repeat(X[i][None, :], k, axis=0))
        Y.append(((gi == T[i]) & (Delta[i] == 1)).astype(np.float64))
    ids = np.concatenate(ids)
    times = np.concatenate(times)
    Xpp = np.column_stack([times, np.vstack(Xrows)])
    Yv = np.concatenate(Y)
    return ids, times, Xpp, Yv


def hazard_hapc(X: np.ndarray, T: np.ndarray, Delta: np.ndarray,
                norm: str = "1",
                max_degree: int = 1,
                npcs: Optional[int] = None,
                time_grid: Optional[np.ndarray] = None,
                log_lambda_min: float = -4,
                log_lambda_max: float = -1,
                grid_length: int = 15,
                nfolds: int = 5,
                predict: Optional[np.ndarray] = None,
                center: bool = True,
                verbose: bool = False,
                max_iter: int = 5000,
                tol: float = 1e-3,
                step_factor: float = 0.8) -> HazardResult:
    """Discrete-time logistic hazard HAPC fit (``family="logit-hazard"``).

    Convenience wrapper around :func:`hapc.cv_hapc` with ``family="binomial"``.
    The right-censored survival data ``(X, T, Delta)`` are expanded into a
    person-period table (one row per subject-per-interval-at-risk) whose binary
    response is the discrete hazard indicator, the visit time is prepended as the
    first HAL covariate, and the regularisation parameter ``lambda`` is chosen by
    cross-validated logistic deviance. R counterpart: ``hapc::hazard.hapc()``.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Baseline covariates, one row per subject.
    T : np.ndarray, shape (n,)
        Observed times ``T_i = min(T_event_i, C_i)``. Assumed discrete.
    Delta : np.ndarray, shape (n,)
        Event indicators ``Delta_i in {0,1}`` (1 = event, 0 = right-censored).
    norm : {"1", "2"}, default "1"
        ``"1"`` = logistic LASSO, ``"2"`` = logistic ridge. ``"sv"`` raises
        ``NotImplementedError``.
    max_degree : int, default 1
        HAL interaction order over ``[time, X]``.
    npcs : int, optional
        Number of principal components. Defaults to the number of person-period
        rows (capped internally as in :func:`hapc.cv_hapc`).
    time_grid : np.ndarray, optional
        Discrete time grid (risk-set grid). Defaults to ``min(T):max(T)`` when
        ``T`` is integer-valued, else ``np.unique(T)``. Subjects are assumed at
        risk from ``min(time_grid)`` onward.
    log_lambda_min, log_lambda_max, grid_length : float / int
        Log-λ CV grid (defaults ``-4``, ``-1``, ``15``).
    nfolds : int, default 5
        Number of CV folds.
    predict : np.ndarray, optional, shape (m, p)
        Baseline covariates for *new subjects*. When supplied the model (refit
        at the CV-selected λ) is evaluated on the full time grid for each new
        subject, returning the hazard surface and the implied survival curves.
    center, verbose, max_iter, tol, step_factor :
        Passed through to :func:`hapc.cv_hapc` / :func:`hapc.hapc`.

    Returns
    -------
    HazardResult
        See :class:`HazardResult`. ``hazard`` holds the cross-validated discrete
        hazard for each person-period row; ``predict_hazard`` /
        ``predict_survival`` are populated when ``predict`` is supplied.

    Notes
    -----
    **Model.** The discrete hazard is the conditional event probability in
    interval ``t`` given survival up to ``t``,

        lambda(t | x) = P(T_event = t | T_event >= t, X = x),

    modelled on the logit scale by a Highly Adaptive Principal Components fit
    ``f`` of the augmented covariate ``(t, x)``:

        logit lambda(t | x) = f(t, x).

    The HAL basis spans indicator (and, for ``max_degree > 1``, interaction)
    tensor products in ``(t, x)``, so the time effect, the covariate effects and
    their interactions are estimated nonparametrically; the L1/L2 penalty
    (``norm``) controls smoothness and is tuned by cross-validation.

    **Person-period likelihood.** Under independent right-censoring the
    observed-data likelihood factorises over the at-risk intervals,

        prod_i prod_{t <= T_i}
            lambda(t | x_i) ** Y_it * (1 - lambda(t | x_i)) ** (1 - Y_it),

    with ``Y_it = 1(T_event_i = t)``. This is exactly the Bernoulli (logistic)
    likelihood of the expanded person-period table, so fitting a binomial HAPC
    model to ``Y_it`` against ``(t, x_i)`` estimates the discrete hazard
    (Cox 1972; Brown 1975; Allison 1982).

    **Survival.** The conditional survival function follows from the estimated
    hazard by the product-limit relation

        S(t | x) = P(T_event > t | x) = prod_{s <= t} (1 - lambda(s | x)),

    returned in ``predict_survival`` for new subjects when ``predict`` is given.

    References
    ----------
    Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2),
    187-220.

    Brown, C. C. (1975). On the use of indicator variables for studying the
    time-dependence of parameters in a response-time model. *Biometrics*,
    31(4), 863-872.

    Allison, P. D. (1982). Discrete-time methods for the analysis of event
    histories. *Sociological Methodology*, 13, 61-98.

    Singer, J. D. and Willett, J. B. (2003). *Applied Longitudinal Data
    Analysis*. Oxford University Press.

    Benkeser, D. and van der Laan, M. (2016). The Highly Adaptive Lasso
    estimator. *IEEE DSAA*, 689-696.

    Examples
    --------
    >>> import numpy as np
    >>> from hapc import hazard_hapc
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> X = np.column_stack([rng.uniform(size=n), rng.integers(0, 2, n)]).astype(float)
    >>> grid = np.arange(1, 7)
    >>> def haz(t, x): return 1 / (1 + np.exp(-(-2.6 + 0.3 * t + 1.3 * x[0] - 0.9 * x[1])))
    >>> Tev = np.full(n, grid.max())
    >>> for i in range(n):
    ...     for t in grid:
    ...         if rng.random() < haz(t, X[i]):
    ...             Tev[i] = t; break
    >>> C = rng.choice(grid, n)
    >>> Tobs = np.minimum(Tev, C); Delta = (Tev <= C).astype(float)
    >>> fit = hazard_hapc(X, Tobs, Delta, norm="1", max_degree=2, time_grid=grid)
    >>> bool(fit.best_lambda > 0)
    True
    """
    norm = str(norm)
    if norm == "sv":
        raise NotImplementedError(
            "family='logit-hazard' (discrete-time logistic hazard) is not "
            "implemented for norm='sv'; use norm='1' or norm='2'."
        )
    if norm not in {"1", "2"}:
        raise ValueError(
            f"family='logit-hazard' supports norm in {{'1','2'}}; got '{norm}'"
        )

    X = _C(X)
    T = np.asarray(T, dtype=np.float64).ravel()
    Delta = np.asarray(Delta, dtype=np.float64).ravel()
    n, p = X.shape
    if T.size != n:
        raise ValueError(f"T must have {n} elements, got {T.size}")
    if Delta.size != n:
        raise ValueError(f"Delta must have {n} elements, got {Delta.size}")
    if not np.all(np.isin(Delta, (0.0, 1.0))):
        raise ValueError("Delta must be a 0/1 event indicator.")

    if time_grid is None:
        grid = _infer_time_grid(T)
    else:
        grid = np.unique(np.asarray(time_grid, dtype=np.float64).ravel())
    if grid.size < 2:
        raise ValueError(f"Need at least two distinct time points; got {grid.size}.")
    if T.max() > grid.max():
        raise ValueError("Some observed times exceed max(time_grid); supply a "
                         "wider 'time_grid'.")

    ids, times, Xpp, Yv = _person_period(X, T, Delta, grid)
    if npcs is None:
        # Default npcs = numerical rank of the person-period HAL design. The
        # expansion creates many duplicate rows (discrete time x repeated
        # baseline covariates), so the kernel is rank-deficient; keeping the
        # null-space directions makes the 1/d prediction reconstruction blow up
        # (notably for norm="2"). Dropping the near-zero singular values is both
        # numerically safe and a sensible amount of regularisation.
        des0 = design_hapc(Xpp, int(max_degree), int(Xpp.shape[0]), center=center)
        d0 = np.asarray(des0.d, dtype=np.float64)
        npcs = int(max(1, np.sum(d0 > 1e-7 * d0[0])))

    cv = cv_hapc(
        Xpp, Yv, family="binomial", norm=norm,
        max_degree=max_degree, npcs=npcs,
        log_lambda_min=log_lambda_min, log_lambda_max=log_lambda_max,
        grid_length=grid_length, nfolds=nfolds, predict=Xpp,
        center=center, verbose=verbose,
        max_iter=max_iter, tol=tol, step_factor=step_factor,
    )

    hazard_train = np.asarray(cv.predictions, dtype=np.float64).ravel()
    lambdas = np.asarray(cv.lambdas, dtype=np.float64)
    risk = np.asarray(cv.mses, dtype=np.float64)
    best_lambda = float(cv.best_lambda)
    best_idx = int(np.argmin(risk))
    interior = bool(0 < best_idx < lambdas.size - 1)

    predict_hazard = predict_survival = None
    if predict is not None:
        Xnew = _C(predict)
        if Xnew.shape[1] != p:
            raise ValueError("predict must have the same number of columns as X.")
        m, K = Xnew.shape[0], grid.size
        # Full-grid expansion: row order = (subject 0: all times), (subject 1: ...)
        new_times = np.tile(grid, m)
        new_X = np.repeat(Xnew, K, axis=0)
        newXpp = np.column_stack([new_times, new_X])
        fit = _hapc(
            Xpp, Yv, family="binomial", norm=norm,
            max_degree=max_degree, npcs=npcs, lambda_=best_lambda,
            predict=newXpp, center=center, verbose=verbose,
            max_iter=max_iter, tol=tol, step_factor=step_factor,
        )
        haz = np.asarray(fit.probabilities, dtype=np.float64).reshape(m, K)
        predict_hazard = haz
        predict_survival = np.cumprod(1.0 - haz, axis=1)

    return HazardResult(
        hazard=hazard_train,
        ids=ids,
        times=times,
        Y=Yv,
        time_grid=grid,
        lambdas=lambdas,
        risk=risk,
        best_lambda=best_lambda,
        interior=interior,
        cv=cv,
        predict_hazard=predict_hazard,
        predict_survival=predict_survival,
    )


__all__ = ["HazardResult", "hazard_hapc"]
