"""Average Treatment Effect estimation with HAPC + undersmoothing.

Provides :func:`ate_hapc`, a high-level convenience wrapper that:

1. Cross-validates the **propensity** model (binomial, ``A ~ W``) on a
   log-spaced grid
   ``(log_lambda_prop_min, log_lambda_prop_max, grid_length_prop)``
   and the **outcome** model (gaussian, ``Y ~ (A, W)``) on a separate grid
   ``(log_lambda_out_min, log_lambda_out_max, grid_length_out)``
   (each built like :func:`hapc.cv_hapc`).
2. Fixes the propensity score at its CV-best λ.
3. Computes σ = std of the ATE efficient influence function (EIF) at the
   CV configuration ``(π̂_CV, μ̂_CV)``.
4. Restricts to the **undersmoothing region** ``λ ≤ λ_CV`` (decreasing λ from
   the CV value = a more flexible outcome fit) and selects the **smallest** λ
   for which ``|mean(EIF)| ≤ σ / (√n · log n)`` — the most undersmoothed fit
   still inside the band (matching §8.3 of the manuscript).  If no λ in the
   region meets the threshold, the λ minimising ``|mean(EIF)|`` is used and a
   warning is emitted.  Warnings also fire when the CV fit already satisfies
   the band (no undersmoothing happens) or when the selection is pinned to the
   grid edge (possible over-undersmoothing).
5. Returns a **doubly robust** ATE point estimate at the undersmoothed outcome
   model and a ``(1 - alpha)`` Wald confidence interval from the EIF evaluated
   at that estimate (see Notes).

This ``method="undersmooth"`` path does not implement sample splitting /
cross-fitting: nuisances are fit on the full sample and the EIF is evaluated on
the same sample, with bias control delegated to the undersmoothing step.

Because the empirical score is evaluated **in-sample**, the undersmoothing gate
is fragile: reducing λ drives the score toward zero through in-sample
overfitting (or floors it via PC truncation) rather than through genuine bias
removal, and ``std(EIF)`` underestimates the sampling variance. The result is a
biased point estimate and/or a Wald interval that under-covers.

``method="crossfit"`` (DML-style K-fold cross-fitting) avoids this: both
nuisances are CV-selected and fit on the training folds and the EIF is evaluated
only on the held-out fold, so its mean and variance are honest and no
undersmoothing is needed. The default remains ``"undersmooth"`` for backward
compatibility; prefer ``"crossfit"`` for valid point estimates and coverage.
"""

import warnings
from typing import NamedTuple, Optional

import numpy as np

try:
    from scipy.stats import norm as _normal
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "scipy is required for ate_hapc (used for normal quantiles). "
        "It ships transitively with scikit-learn; run `pip install scipy`."
    ) from _e

from .cv import CVResult, cv_hapc
from .single import hapc as _hapc


class ATEResult(NamedTuple):
    """Output of :func:`ate_hapc`.

    Attributes
    ----------
    estimate : float
        Doubly robust (AIPW-style) ATE at the undersmoothed outcome model:
        ``mean(A/π̂·(Y-μ̂₁)+μ̂₁ - (1-A)/(1-π̂)·(Y-μ̂₀) - μ̂₀)``, matching the
        efficient influence function used for the Wald interval (see Notes).
    lower : float
        Lower endpoint of the ``(1 - alpha)`` Wald confidence interval.
    upper : float
        Upper endpoint of the ``(1 - alpha)`` Wald confidence interval.
    """

    estimate: float
    lower: float
    upper: float


def _plot_ate_diagnostics(
    cv_prop: CVResult,
    cv_out: CVResult,
    traj_lambdas: np.ndarray,
    traj_abs_mean_eif: np.ndarray,
    lam_prop_cv: float,
    lam_out_cv: float,
    lam_undersmooth: float,
    threshold: float,
) -> None:
    """Raise ImportError if matplotlib is missing; otherwise show diagnostic figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_diagnostics=True requires matplotlib. "
            "Install with: pip install matplotlib"
        ) from e

    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.35, wspace=0.3)
    ax_prop = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[1, :])

    lp = np.asarray(cv_prop.lambdas, dtype=float)
    sp = np.asarray(cv_prop.mses, dtype=float)
    lo = np.asarray(cv_out.lambdas, dtype=float)
    so = np.asarray(cv_out.mses, dtype=float)

    ax_prop.semilogx(lp, sp, "o-", color="C1", lw=1.5, ms=5)
    ax_prop.axvline(lam_prop_cv, color="C3", ls="--", lw=1.5,
                    label=f"CV λ = {lam_prop_cv:.4g}")
    ax_prop.set_xlabel("λ (propensity)")
    ax_prop.set_ylabel("Mean CV logistic deviance")
    ax_prop.set_title("Propensity CV (A ~ W, binomial)")
    ax_prop.legend(loc="best", fontsize=8)
    ax_prop.grid(True, alpha=0.3)

    ax_out.semilogx(lo, so, "o-", color="C2", lw=1.5, ms=5)
    ax_out.axvline(lam_out_cv, color="C3", ls="--", lw=1.5,
                   label=f"CV λ = {lam_out_cv:.4g}")
    ax_out.set_xlabel("λ (outcome)")
    ax_out.set_ylabel("Mean CV MSE")
    ax_out.set_title("Outcome CV (Y ~ (A,W), gaussian)")
    ax_out.legend(loc="best", fontsize=8)
    ax_out.grid(True, alpha=0.3)

    tv = np.asarray(traj_lambdas, dtype=float)
    yv = np.asarray(traj_abs_mean_eif, dtype=float)
    ok = np.isfinite(tv) & np.isfinite(yv) & (tv > 0)
    tv, yv = tv[ok], yv[ok]
    order = np.argsort(tv)
    tv, yv = tv[order], yv[order]

    if tv.size:
        ax_traj.semilogx(tv, yv, "o-", color="C0", lw=2, ms=6,
                         label=r"$|\mathrm{mean}(\mathrm{EIF}_{\mathrm{ATE}})|$")
        ax_traj.fill_between(tv, 0, threshold, alpha=0.12, color="gray")
    else:
        ax_traj.text(
            0.5, 0.5, "No valid outcome fits on λ grid",
            transform=ax_traj.transAxes, ha="center", va="center",
        )
    ax_traj.axhline(threshold, color="gray", lw=2, alpha=0.85,
                    label=r"Threshold $\sigma_{\mathrm{CV}}/(\sqrt{n}\log n)$")
    ax_traj.axvline(lam_out_cv, color="C3", ls="--", lw=1.8,
                    label=f"Outcome CV λ = {lam_out_cv:.4g}")
    ax_traj.axvline(lam_undersmooth, color="C4", ls="-", lw=2.0,
                    label=f"Undersmoothed λ = {lam_undersmooth:.4g}")
    ax_traj.set_xlabel("Outcome λ (undersmoothing grid)")
    ax_traj.set_ylabel(r"$|\mathrm{mean}(\mathrm{EIF})|$")
    ax_traj.set_title("Undersmoothing trajectory (fixed propensity at its CV-λ)")
    ax_traj.legend(loc="best", fontsize=9, ncol=2)
    ax_traj.grid(True, alpha=0.3)

    fig.suptitle("ate_hapc diagnostics", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
    plt.show()


def _print_undersmoothing_table(
    path_abs: dict, threshold: float, sigma_cv: float, n: int,
    lam_out_cv: float, lam_und: float,
) -> None:
    """Print a ``|mean(EIF)|`` vs ``λ`` table for the undersmoothing scan (stdout).

    Lets the user confirm the undersmoothing gate fired correctly: every outcome
    ``λ`` in the undersmoothing region (``λ ≤ λ_CV``, listed from λ_CV downward =
    increasing flexibility), its in-sample ``|mean(EIF)|``, and whether it lies
    within the threshold ``τ = σ_CV / (√n · log n)`` (σ_CV is the std of the EIF
    at the CV outcome fit). The CV λ and the selected undersmoothed λ are flagged.
    """
    lams = sorted(path_abs, reverse=True)  # λ_CV → smallest λ (scan direction)
    print("\nate_hapc undersmoothing report (method='undersmooth')")
    print(f"  n = {n}   sigma_CV = {sigma_cv:.6g}   "
          f"tau = sigma_CV / (sqrt(n) * log(n)) = {threshold:.6g}")
    print(f"  CV outcome lambda    = {lam_out_cv:.4e}")
    print(f"  undersmoothed lambda = {lam_und:.4e}")
    print(f"  {'lambda':>13}  {'|mean EIF|':>12}  {'<= tau':>7}   note")
    print("  " + "-" * 54)
    for lam in lams:
        a = path_abs[lam]
        ok = "yes" if a <= threshold else "no"
        flags = []
        if abs(lam - lam_out_cv) <= lam_out_cv * 1e-9:
            flags.append("CV lambda")
        if abs(lam - lam_und) <= lam_und * 1e-9:
            flags.append("<-- selected")
        print(f"  {lam:13.4e}  {a:12.6f}  {ok:>7}   {', '.join(flags)}")
    print()


def _warn_if_cv_edge(best_lambda: float, lambdas, label: str, prefix: str) -> None:
    """Warn if the CV-selected λ sits on a grid edge (optimum likely outside).

    A boundary CV λ means the MSE-optimal penalty is outside the searched
    range, so the nuisance is mis-regularised (and, for the outcome, the
    undersmoothing region starts from the wrong place). ``prefix`` is the
    parameter stem (``"prop"`` or ``"out"``) used in the remedial hint.
    """
    lams = np.asarray(lambdas, dtype=float)
    if lams.size < 2:
        return
    lo, hi = float(lams.min()), float(lams.max())
    if abs(np.log(best_lambda) - np.log(hi)) < 1e-6:
        side, knob = "upper", f"raise log_lambda_{prefix}_max"
    elif abs(np.log(best_lambda) - np.log(lo)) < 1e-6:
        side, knob = "lower", f"lower log_lambda_{prefix}_min"
    else:
        return
    warnings.warn(
        f"ate_hapc: {label} CV λ is at the {side} edge of its grid "
        f"(λ={best_lambda:.3g}); the CV optimum is likely outside the range — "
        f"{knob} so CV selects an interior λ.",
        stacklevel=3,
    )


def _coerce_binary(A: np.ndarray) -> np.ndarray:
    """Return ``A`` re-encoded as floats in ``{0,1}``.

    Accepts ``{0,1}``, ``{-1,+1}`` (or any pair where one value is non-positive
    and one positive — falls back to the sign).
    """
    A = np.asarray(A).ravel()
    u = set(np.unique(A).tolist())
    if u.issubset({0, 1, 0.0, 1.0}):
        return A.astype(np.float64)
    if u.issubset({-1, 1, -1.0, 1.0}):
        return ((A > 0).astype(np.float64))
    raise ValueError(
        f"A must be binary in {{0,1}} or {{-1,+1}}; found {sorted(u)}"
    )


def _make_folds(n: int, K: int, strat: np.ndarray, seed: int) -> np.ndarray:
    """Stratified K-fold assignment in ``{0,…,K-1}`` (balanced within each arm).

    Stratifying on the treatment ``strat`` keeps both arms present in every
    training split, which the propensity/outcome fits and positivity require.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    folds = np.empty(n, dtype=np.int64)
    for cls in (0.0, 1.0):
        idx = np.where(strat == cls)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(idx.size) % K
    return folds


def _ate_crossfit(
    X: np.ndarray, Y: np.ndarray, A01: np.ndarray, n: int, *,
    alpha: float, max_degree: int, npcs: int,
    log_lambda_prop_min: float, log_lambda_prop_max: float, grid_length_prop: int,
    log_lambda_out_min: float, log_lambda_out_max: float, grid_length_out: int,
    nfolds: int, norm: str, max_iter: int, tol: float, step_factor: float,
    verbose: bool, crit: str, center: bool, approx: bool, ini: str,
    cf_folds: int, cf_seed: int,
) -> "ATEResult":
    """Cross-fitted (DML-style) AIPW ATE; counterpart to the undersmoothing path.

    Both nuisances (propensity ``A~W`` and outcome ``Y~(A,W)``) are CV-selected
    **and** fit on the training folds, then evaluated on the held-out fold, so
    the efficient influence function is an honest out-of-sample quantity. This
    removes the in-sample bias the undersmoothing path tries to control, and the
    EIF variance is no longer underestimated.
    """
    if cf_folds < 2:
        raise ValueError(f"cf_folds must be >= 2; got {cf_folds}")
    if min(int(A01.sum()), int(n - A01.sum())) < cf_folds:
        raise ValueError(
            f"cf_folds={cf_folds} exceeds the rarer treatment arm's count; "
            "use fewer folds or more data."
        )

    folds = _make_folds(n, cf_folds, A01, cf_seed)
    pi1 = np.empty(n, dtype=np.float64)
    mu1 = np.empty(n, dtype=np.float64)
    mu0 = np.empty(n, dtype=np.float64)

    cv_kwargs = dict(
        max_degree=max_degree, nfolds=nfolds, norm=norm,
        max_iter=max_iter, tol=tol, step_factor=step_factor,
        verbose=verbose, crit=crit, center=center, approx=approx, ini=ini,
    )
    fit_kwargs = dict(
        max_iter=max_iter, tol=tol, step_factor=step_factor,
        verbose=verbose, crit=crit, center=center, approx=approx, ini=ini,
    )

    for k in range(cf_folds):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        if te.size == 0 or tr.size == 0:
            raise ValueError(f"Empty cross-fitting fold {k}; reduce cf_folds.")
        # Cap PCs at the training-fold rank budget.
        npcs_k = max(1, min(int(npcs), tr.size - 1))
        Xtr, Xte = X[tr], X[te]

        # --- propensity: CV + refit on training folds, predict on held-out ----
        cv_prop = cv_hapc(
            Xtr, A01[tr], family="binomial",
            log_lambda_min=log_lambda_prop_min,
            log_lambda_max=log_lambda_prop_max,
            grid_length=grid_length_prop, npcs=npcs_k, **cv_kwargs,
        )
        prop = _hapc(
            Xtr, A01[tr], family="binomial", max_degree=max_degree, npcs=npcs_k,
            lambda_=float(cv_prop.best_lambda), norm=norm, predict=Xte,
            **fit_kwargs,
        )
        pi1[te] = np.clip(np.asarray(prop.probabilities).ravel(), 1e-8, 1 - 1e-8)

        # --- outcome: CV + refit on training folds, predict both arms ---------
        Xout_tr = np.column_stack([A01[tr], Xtr])
        cv_out = cv_hapc(
            Xout_tr, Y[tr], family="gaussian",
            log_lambda_min=log_lambda_out_min,
            log_lambda_max=log_lambda_out_max,
            grid_length=grid_length_out, npcs=npcs_k, **cv_kwargs,
        )
        m = te.size
        Xeval = np.vstack([
            np.column_stack([np.ones(m), Xte]),
            np.column_stack([np.zeros(m), Xte]),
        ])
        res = _hapc(
            Xout_tr, Y[tr], family="gaussian", max_degree=max_degree, npcs=npcs_k,
            lambda_=float(cv_out.best_lambda), norm=norm, predict=Xeval,
            **fit_kwargs,
        )
        p = np.asarray(res.predictions).ravel()
        if p.size != 2 * m:
            raise RuntimeError(
                f"Outcome predict returned {p.size} values, expected {2 * m}."
            )
        mu1[te], mu0[te] = p[:m], p[m:]

    # --- cross-fitted AIPW point estimate + Wald CI -------------------------
    phi = (
        (A01 / pi1) * (Y - mu1) + mu1
        - ((1.0 - A01) / (1.0 - pi1)) * (Y - mu0) - mu0
    )
    psi = float(phi.mean())
    sigma = float(np.std(phi, ddof=0))
    z = float(_normal.ppf(1.0 - alpha / 2.0))
    half = z * sigma / np.sqrt(n)
    return ATEResult(estimate=psi, lower=psi - half, upper=psi + half)


def ate_hapc(X: np.ndarray, Y: np.ndarray, A: np.ndarray,
             alpha: float = 0.05,
             max_degree: int = 1,
             npcs: Optional[int] = None,
             log_lambda_prop_min: float = -5,
             log_lambda_prop_max: float = 2,
             grid_length_prop: int = 14,
             log_lambda_out_min: float = -10,
             log_lambda_out_max: float = 2,
             grid_length_out: int = 22,
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
             plot_diagnostics: bool = False,
             method: str = "undersmooth",
             cf_folds: int = 5,
             cf_seed: int = 0,
             report_undersmoothing: bool = False) -> ATEResult:
    """ATE estimate with HAPC nuisances and outcome undersmoothing.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Covariate matrix ``W`` (do NOT include the treatment column).
    Y : np.ndarray, shape (n,)
        Continuous outcome.
    A : np.ndarray, shape (n,)
        Binary treatment in ``{0,1}`` or ``{-1,+1}``.
    alpha : float, default 0.05
        Significance level.  The returned interval has confidence
        ``1 - alpha``.
    max_degree, npcs, nfolds, norm, predict, max_iter, tol, step_factor,\
        verbose, crit, center, approx, ini :
        Same meaning and defaults as in :func:`hapc.cv_hapc` (except λ grids,
        see below).
        ``predict`` is accepted for signature parity with :func:`cv_hapc` and
        is currently ignored (``ate_hapc`` always evaluates the EIF on the
        training sample).
    log_lambda_prop_min, log_lambda_prop_max, grid_length_prop :
        Equally spaced log-λ grid for **propensity** cross-validation
        (``A ~ W``, binomial), same rule as :func:`cv_hapc`.
    log_lambda_out_min, log_lambda_out_max, grid_length_out :
        Log-λ grid (defaults ``-10``, ``2``, ``22``) for **outcome**
        cross-validation ``Y ~ (A, W)`` (gaussian) and for the
        **undersmoothing** scan over ``λ ≤ λ_CV``.  The grid deliberately
        reaches very small λ: for ``method="undersmooth"`` the (nearly)
        unbiased regime typically sits well below the CV-optimal λ, so a grid
        that stops too high (e.g. the old ``[e^{-5}, e^{-3}]``) never reaches it
        and no undersmoothing effectively occurs.  Undersmoothing only removes
        bias when the PC basis is rich enough to represent the outcome surface,
        so use the **full basis** (``npcs = n``, the default); a truncated basis
        floors the bias at an approximation level that no λ can remove.
    plot_diagnostics : bool, default False
        If True, open a matplotlib figure with (1) propensity CV curve
        (logistic deviance vs λ), (2) outcome CV curve (MSE vs λ), and (3)
        the undersmoothing path: ``|mean(EIF)|`` vs outcome λ with the
        threshold line and vertical markers for the CV and selected
        undersmoothed λ. Requires ``matplotlib`` (``pip install matplotlib``).
        Only meaningful for ``method="undersmooth"`` (ignored otherwise).
    method : {"undersmooth", "crossfit"}, default "undersmooth"
        Bias-control strategy.

        * ``"undersmooth"`` — the original single-sample procedure: fit the
          nuisances on the full sample, evaluate the EIF in-sample, and pick the
          outcome λ by the undersmoothing gate (Notes). Kept as the default for
          backward compatibility.
        * ``"crossfit"`` — DML-style K-fold **cross-fitting**: CV-select and fit
          both nuisances on the training folds and evaluate the EIF only on the
          held-out fold. No undersmoothing is performed (CV-optimal λ is used
          per fold); the outcome λ grid is still used for the per-fold CV. This
          makes the EIF mean and variance honest, so the AIPW estimate is
          √n-unbiased and the Wald CI attains nominal coverage without relying
          on the undersmoothing gate. Prefer this unless you specifically need
          the single-sample undersmoothed estimator.
    cf_folds : int, default 5
        Number of cross-fitting folds (only used when ``method="crossfit"``).
        Folds are stratified by treatment so both arms appear in every split.
    cf_seed : int, default 0
        Seed for the (deterministic) cross-fitting fold assignment.
    report_undersmoothing : bool, default False
        If True (and ``method="undersmooth"``), print a table to stdout of
        ``|mean(EIF)|`` versus outcome λ over the undersmoothing region
        (``λ ≤ λ_CV``), with the threshold ``τ = σ_CV / (√n · log n)``, so the
        user can confirm the undersmoothing gate fired correctly. Ignored for
        ``method="crossfit"`` (which evaluates the EIF out-of-fold).

    Returns
    -------
    ATEResult
        Named tuple with three fields ``(estimate, lower, upper)``.

    Notes
    -----
    The procedure is:

    1. Cross-validate the propensity ``A ~ W`` (binomial) on its grid and the
       outcome ``Y ~ (A, W)`` (gaussian) on the outcome grid (independently
       specified).
    2. Fix the propensity at its CV-best λ; refit on the full sample to
       obtain ``π̂(W_i) = P(A=1 | W_i)``.
    3. At the CV-best outcome λ, compute a **plugin-centered** influence vector
       (same mean as the DR EIF at :math:`\\psi=\\overline{\\mu}_1-\\overline{\\mu}_0`)
       and let ``σ = std(·)``.
    4. Threshold ``τ = σ / (√n · log n)``.
    5. Among the **outcome** λ in the undersmoothing region ``λ ≤ λ_CV``, pick
       the **smallest** λ for which ``|mean(EIF_diff)| ≤ τ`` (the most
       undersmoothed fit still inside the band) — call it ``λ_u``.  If none
       qualifies, ``λ_u`` minimises ``|mean(EIF_diff)|`` (with a warning).
    6. **Doubly robust** point estimate at the **undersmoothed** outcome fit
       ``(π̂, μ̂₁^{λ_u}, μ̂₀^{λ_u})``:
       ``ψ̂ = mean(A/π̂·(Y-μ̂₁)+μ̂₁ - (1-A)/(1-π̂)·(Y-μ̂₀) - μ̂₀)``.
       The CI uses the one-step influence function evaluated at the **CV**
       outcome fit ``(μ̂₁^{λ_CV}, μ̂₀^{λ_CV})`` (centered at ``ψ̂``):
       ``φ_i = A_i/π̂_i·(Y_i-μ̂_{1i}) + μ̂_{1i} - (1-A_i)/(1-π̂_i)·(Y_i-μ̂_{0i})
       - μ̂_{0i} - ψ̂``, and ``CI: ψ̂ ± z_{1-α/2} · std(φ) / √n``.
       The CV (rather than undersmoothed) fit is used for ``std(φ)`` because a
       heavily undersmoothed, near-interpolating fit has artificially small
       in-sample residuals that collapse the SE and destroy coverage; the CV
       fit yields honest residuals. (For ``method="crossfit"`` no such issue
       arises — the EIF is evaluated out-of-fold.)

       This contrasts with **plug-in** G-computation ``mean(μ̂₁(W)-μ̂₀(W))``,
       which can be materially biased when both nuisances are estimated on the
       same sample and the outcome regressions are regularized.  The DR
       ``ψ̂`` is consistent if **either** the propensity **or** the pair
       ``(μ̂₁, μ̂₀)`` is correctly specified (standard double robustness).

    Examples
    --------
    >>> import numpy as np
    >>> from hapc import ate_hapc
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> W = np.column_stack([rng.uniform(-2, 2, n), rng.normal(0, 0.5, n)])
    >>> p = 1.0 / (1.0 + np.exp(-(W[:, 0] + 0.5 * W[:, 1])))
    >>> A = rng.binomial(1, p, n)
    >>> Y = 2 * W[:, 0] + 0.5 + rng.normal(0, 0.5, n)  # truth: ATE=0
    >>> res = ate_hapc(W, Y, A, alpha=0.05, max_degree=2, npcs=50,
    ...                grid_length_prop=4, grid_length_out=4, nfolds=3,
    ...                norm="2")
    >>> bool(res.lower <= res.estimate <= res.upper)
    True
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    if method not in {"undersmooth", "crossfit"}:
        raise ValueError(
            f"method must be 'undersmooth' or 'crossfit'; got '{method}'"
        )

    # --- Coerce inputs ------------------------------------------------------
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A01 = _coerce_binary(A)
    n, _p = X.shape
    if Y.size != n or A01.size != n:
        raise ValueError("X, Y, A must all have the same number of rows.")

    if npcs is None:
        npcs = int(n)

    if method == "crossfit":
        if plot_diagnostics:
            warnings.warn(
                "plot_diagnostics is only available for method='undersmooth'; "
                "ignoring it for method='crossfit'.",
                stacklevel=2,
            )
        return _ate_crossfit(
            X, Y, A01, n, alpha=alpha, max_degree=max_degree, npcs=npcs,
            log_lambda_prop_min=log_lambda_prop_min,
            log_lambda_prop_max=log_lambda_prop_max,
            grid_length_prop=grid_length_prop,
            log_lambda_out_min=log_lambda_out_min,
            log_lambda_out_max=log_lambda_out_max,
            grid_length_out=grid_length_out,
            nfolds=nfolds, norm=norm, max_iter=max_iter, tol=tol,
            step_factor=step_factor, verbose=verbose, crit=crit,
            center=center, approx=approx, ini=ini,
            cf_folds=cf_folds, cf_seed=cf_seed,
        )

    lambdas_out = np.exp(
        np.linspace(log_lambda_out_min, log_lambda_out_max, grid_length_out))

    cv_kwargs_base = dict(
        max_degree=max_degree, npcs=npcs, nfolds=nfolds, norm=norm,
        max_iter=max_iter, tol=tol, step_factor=step_factor,
        verbose=verbose, crit=crit, center=center, approx=approx, ini=ini,
    )

    # --- 1. CV propensity (binomial) ---------------------------------------
    cv_prop = cv_hapc(
        X, A01, family="binomial",
        log_lambda_min=log_lambda_prop_min,
        log_lambda_max=log_lambda_prop_max,
        grid_length=grid_length_prop,
        **cv_kwargs_base,
    )
    lam_prop_cv = float(cv_prop.best_lambda)

    # Refit propensity at CV λ on full data, predict in-sample probabilities.
    prop = _hapc(
        X, A01, family="binomial", max_degree=max_degree, npcs=npcs,
        lambda_=lam_prop_cv, norm=norm, predict=X,
        max_iter=max_iter, tol=tol, step_factor=step_factor,
        verbose=verbose, crit=crit, center=center, approx=approx, ini=ini,
    )
    pi1 = np.clip(np.asarray(prop.probabilities).ravel(), 1e-8, 1 - 1e-8)

    # --- 2. CV outcome (gaussian on [A,W]) ---------------------------------
    Xout = np.column_stack([A01, X])
    cv_out = cv_hapc(
        Xout, Y, family="gaussian",
        log_lambda_min=log_lambda_out_min,
        log_lambda_max=log_lambda_out_max,
        grid_length=grid_length_out,
        **cv_kwargs_base,
    )
    lam_out_cv = float(cv_out.best_lambda)

    # Stacked design for one-shot prediction at both arms.
    Xmu1 = np.column_stack([np.ones(n), X])
    Xmu0 = np.column_stack([np.zeros(n), X])
    Xeval = np.vstack([Xmu1, Xmu0])

    def _mu_pair(lam: float):
        """Refit outcome at λ on full data, return (μ̂_1, μ̂_0) on training W."""
        res = _hapc(
            Xout, Y, family="gaussian", max_degree=max_degree, npcs=npcs,
            lambda_=float(lam), norm=norm, predict=Xeval,
            max_iter=max_iter, tol=tol, step_factor=step_factor,
            verbose=verbose, crit=crit, center=center, approx=approx, ini=ini,
        )
        p = np.asarray(res.predictions).ravel()
        if p.size != 2 * n:
            raise RuntimeError(
                f"Outcome predict returned {p.size} values, expected {2 * n}."
            )
        return p[:n], p[n:]

    def _eif_plugin_centered(mu1: np.ndarray, mu0: np.ndarray) -> np.ndarray:
        """Plugin-centered influence vector (undersmoothing gate only).

        Its mean matches the DR EIF evaluated at plug-in
        :math:`\\psi=\\overline{\\mu}_1-\\overline{\\mu}_0`. The returned ATE
        uses ``_psi_dr`` / ``_eif_dr`` instead.
        """
        eif1 = (A01 / pi1) * (Y - mu1) - (mu1 - mu1.mean())
        eif0 = ((1.0 - A01) / (1.0 - pi1)) * (Y - mu0) - (mu0 - mu0.mean())
        return eif1 - eif0

    def _psi_dr(mu1: np.ndarray, mu0: np.ndarray) -> float:
        return float(
            np.mean(
                (A01 / pi1) * (Y - mu1)
                + mu1
                - ((1.0 - A01) / (1.0 - pi1)) * (Y - mu0)
                - mu0
            )
        )

    def _eif_dr(mu1: np.ndarray, mu0: np.ndarray, psi: float) -> np.ndarray:
        return (
            (A01 / pi1) * (Y - mu1)
            + mu1
            - ((1.0 - A01) / (1.0 - pi1)) * (Y - mu0)
            - mu0
            - psi
        )

    # --- 3. σ at CV configuration → threshold τ ----------------------------
    mu1_cv, mu0_cv = _mu_pair(lam_out_cv)
    eif_cv = _eif_plugin_centered(mu1_cv, mu0_cv)
    sigma_cv = float(np.std(eif_cv, ddof=0))
    threshold = sigma_cv / (np.sqrt(n) * np.log(n))

    # --- 4. Undersmoothing sweep over the undersmoothing region λ ≤ λ_CV ----
    # Faithful to §8.3: starting at the CV λ, decrease λ (more flexible outcome
    # fit) and select the *smallest* λ whose |mean(EIF)| ≤ τ — the most
    # undersmoothed fit still inside the band.  Only λ ≤ λ_CV are eligible;
    # larger λ would be *over*-smoothing, not undersmoothing.  The EIF is
    # evaluated in-sample, so the gate is fragile: the warnings below flag the
    # degenerate regimes (no undersmoothing region, band never met, the CV fit
    # already inside the band, or undersmoothing pinned to the grid edge).
    und_lams = np.sort(lambdas_out[lambdas_out <= lam_out_cv * (1.0 + 1e-9)])
    if und_lams.size == 0:  # numerical safety; CV λ is always a grid point
        und_lams = np.array([lam_out_cv], dtype=float)

    path_abs: dict = {}
    path_mu: dict = {}
    for lam in und_lams:
        try:
            mu1, mu0 = _mu_pair(float(lam))
        except Exception:
            continue
        path_mu[float(lam)] = (mu1, mu0)
        path_abs[float(lam)] = float(abs(_eif_plugin_centered(mu1, mu0).mean()))

    if not path_mu:
        raise RuntimeError("Outcome model failed to fit on the entire λ grid.")

    fitted = sorted(path_mu)  # ascending λ
    within = [lam for lam in fitted if path_abs[lam] <= threshold]
    if within:
        lam_und = float(min(within))  # smallest λ inside the band (§8.3 rule)
        if lam_und >= lam_out_cv * (1.0 - 1e-9):
            warnings.warn(
                "ate_hapc(method='undersmooth'): the CV outcome fit already "
                "satisfies the EIF threshold τ, so no undersmoothing occurred "
                "and any in-sample plug-in/DR bias is left uncorrected. Prefer "
                "method='crossfit', or lower log_lambda_out_min.",
                stacklevel=2,
            )
        elif len(fitted) > 1 and lam_und <= float(fitted[0]) * (1.0 + 1e-9):
            warnings.warn(
                "ate_hapc(method='undersmooth'): undersmoothing reached the "
                "smallest outcome λ in the grid (the band stays satisfied down "
                "to the grid edge), which may over-undersmooth toward "
                "interpolation. Inspect plot_diagnostics or prefer "
                "method='crossfit'.",
                stacklevel=2,
            )
    else:
        # Band never reached in the undersmoothing region → best effort.
        lam_und = float(min(path_abs, key=path_abs.get))
        warnings.warn(
            "ate_hapc(method='undersmooth'): undersmoothing never brought "
            "|mean(EIF)| within the threshold τ along the outcome λ grid; using "
            "the λ that minimises it. The in-sample EIF gate is uninformative "
            "here — prefer method='crossfit'.",
            stacklevel=2,
        )
    mu1_und, mu0_und = path_mu[lam_und]

    if report_undersmoothing:
        _print_undersmoothing_table(
            path_abs, threshold, sigma_cv, n, lam_out_cv, lam_und,
        )

    if plot_diagnostics:
        t_lams: list[float] = []
        t_abs: list[float] = []
        for lam in np.sort(lambdas_out):
            try:
                mu1, mu0 = _mu_pair(float(lam))
            except Exception:
                continue
            eif = _eif_plugin_centered(mu1, mu0)
            t_lams.append(float(lam))
            t_abs.append(float(np.abs(eif.mean())))
        _plot_ate_diagnostics(
            cv_prop, cv_out,
            np.asarray(t_lams), np.asarray(t_abs),
            lam_prop_cv, lam_out_cv, lam_und, threshold,
        )

    # --- 5. Doubly robust point estimate + (1 - alpha) Wald CI --------------
    # Point estimate at the undersmoothed fit (low bias), but the SE is the
    # std of the EIF at the **CV** outcome fit, not the undersmoothed one: a
    # heavily undersmoothed (near-interpolating) fit has artificially small
    # in-sample residuals (Y − μ̂) that collapse std(EIF) and destroy coverage.
    # The CV fit gives honest residuals, so std(EIF) is a consistent estimate
    # of the efficient SE. (When no undersmoothing occurs the two coincide.)
    psi = _psi_dr(mu1_und, mu0_und)
    eif_se = _eif_dr(mu1_cv, mu0_cv, psi)
    sigma = float(np.std(eif_se, ddof=0))
    z = float(_normal.ppf(1.0 - alpha / 2.0))
    half = z * sigma / np.sqrt(n)

    return ATEResult(estimate=psi, lower=psi - half, upper=psi + half)


__all__ = ["ATEResult", "ate_hapc"]
