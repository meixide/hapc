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
4. Sweeps the **outcome** λ grid in **decreasing**
   order (most smoothing → least smoothing) and stops at the first λ for
   which ``|mean(EIF)| ≤ σ / (√n · log n)``.  This is the **undersmoothed**
   outcome model.  If no λ in the grid meets the threshold, the smallest λ
   is used.
5. Returns a **doubly robust** ATE point estimate at the undersmoothed outcome
   model and a ``(1 - alpha)`` Wald confidence interval from the EIF evaluated
   at that estimate (see Notes).

The function does not implement sample splitting / cross-fitting:
nuisances are fit on the full sample and the EIF is evaluated on the same
sample.  Bias control is provided by the undersmoothing step instead.
"""

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


def ate_hapc(X: np.ndarray, Y: np.ndarray, A: np.ndarray,
             alpha: float = 0.05,
             max_degree: int = 1,
             npcs: Optional[int] = None,
             log_lambda_prop_min: float = -5,
             log_lambda_prop_max: float = -3,
             grid_length_prop: int = 10,
             log_lambda_out_min: float = -5,
             log_lambda_out_max: float = -3,
             grid_length_out: int = 10,
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
             plot_diagnostics: bool = False) -> ATEResult:
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
        Log-λ grid for **outcome** cross-validation ``Y ~ (A, W)`` (gaussian)
        and for the **undersmoothing** scan (same points, evaluated in
        decreasing λ order until ``|mean(EIF)| ≤ τ``).
    plot_diagnostics : bool, default False
        If True, open a matplotlib figure with (1) propensity CV curve
        (logistic deviance vs λ), (2) outcome CV curve (MSE vs λ), and (3)
        the undersmoothing path: ``|mean(EIF)|`` vs outcome λ with the
        threshold line and vertical markers for the CV and selected
        undersmoothed λ. Requires ``matplotlib`` (``pip install matplotlib``).

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
    5. Walk the **outcome** λ grid in **decreasing**
       order; pick the first (largest) λ for which
       ``|mean(EIF_diff)| ≤ τ`` — call it ``λ_u``.
    6. **Doubly robust** point estimate (same nuisances ``(π̂, μ̂₁, μ̂₀)``):
       ``ψ̂ = mean(A/π̂·(Y-μ̂₁)+μ̂₁ - (1-A)/(1-π̂)·(Y-μ̂₀) - μ̂₀)``.
       One-step influence function (centered at ``ψ̂``):
       ``φ_i = A_i/π̂_i·(Y_i-μ̂_{1i}) + μ̂_{1i} - (1-A_i)/(1-π̂_i)·(Y_i-μ̂_{0i})
       - μ̂_{0i} - ψ̂``.
       CI: ``ψ̂ ± z_{1-α/2} · std(φ) / √n``.

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

    # --- 4. Undersmoothing sweep: largest λ → smallest --------------------
    lam_und: Optional[float] = None
    mu1_und = mu0_und = None
    for lam in np.sort(lambdas_out)[::-1]:
        try:
            mu1, mu0 = _mu_pair(float(lam))
        except Exception:
            continue
        eif = _eif_plugin_centered(mu1, mu0)
        if abs(eif.mean()) <= threshold:
            lam_und = float(lam)
            mu1_und, mu0_und = mu1, mu0
            break

    if lam_und is None:
        # Threshold never met → fall back to the smallest λ in the grid.
        lam_und = float(lambdas_out.min())
        mu1_und, mu0_und = _mu_pair(lam_und)

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
    psi = _psi_dr(mu1_und, mu0_und)
    eif_dr = _eif_dr(mu1_und, mu0_und, psi)
    sigma_und = float(np.std(eif_dr, ddof=0))
    z = float(_normal.ppf(1.0 - alpha / 2.0))
    half = z * sigma_und / np.sqrt(n)

    return ATEResult(estimate=psi, lower=psi - half, upper=psi + half)


__all__ = ["ATEResult", "ate_hapc"]
