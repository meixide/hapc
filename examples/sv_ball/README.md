# Convex SV-ball solver — 27 October 2026

## Estimator and authorised change

`norm="sv"` now minimises the empirical loss over

\[
\{\alpha:\|V\alpha\|_1\le C\},\qquad C=\|V\alpha_0(\lambda)\|_1.
\]

Gaussian loss is mean squared error; binomial loss is mean logistic negative log likelihood with a profiled, unpenalised intercept. The existing initializer and outer lambda selection still determine C. A minimum inside the ball is allowed. The previous multiplicative equality-constrained algorithm remains explicitly callable as `hapc_core.pcghal_boundary_call` / `pcghal_boundary_classi_call` for comparisons.

This is the change authorised by Carlos after the first audit: the boundary algorithm could keep zero PC coefficients locked and could stop at a point that was not even a local constrained minimum. The original example `Z=V=I, alpha0=(1,0), y=(0,2)` now returns `(0,1)` with risk 0.5 instead of 2.5, and zero numerical gap. Another test verifies that a slack-radius unconstrained optimum remains interior.

The manuscript's estimator definition, algorithms, score equations and simulations must reflect this ball formulation before submission. Nothing here establishes a statistical rate, boundedness of a CV-selected radius as n grows, or calibrated confidence intervals.

## Numerical method

The split variable b satisfies `b=V alpha`; b is projected onto the L1 ball. ADMM alternates a Gaussian quadratic solve or a profiled-logistic safeguarded Newton solve, the exact L1 projection, and a dual update. The Gaussian factorisation is cached while rho stays fixed. Rho adapts by residual balancing.

Sparse HAL cases can approach the optimum slowly. After at most 500 ADMM iterations, an epigraph log-barrier refinement solves the **same** convex problem if the certificate is still insufficient. Its q slack variables are eliminated analytically, leaving a k-by-k Newton system. Active-face equations provide additional dual candidates without dividing by nearly zero barrier slacks. A guessed active set is never itself a convergence certificate: every candidate is checked by the same global bound. The user iteration budget includes ADMM iterations and accepted barrier Newton steps; inner logistic Newton solves are not separate outer iterations.

Primal incumbents are always feasible. Numerical objective ties are broken by a tighter dual gap; the best valid dual lower bound is kept across iterations. This prevents freezing coefficients when objective changes round to zero. The history is a feasible-incumbent history, not the raw (possibly infeasible) ADMM alpha trajectory.

The shared HAL decomposition now removes eigenvalues below `8 * machine_epsilon * n * largest_eigenvalue` when positive numerical rank exists. A repeated binary design requested with 20 PCs correctly retains one PC, rather than noise eigenvectors. Arbitrary rank-deficient penalty maps do not receive a finite optimality certificate. Fully constant/intercept-only designs and scalability of explicit HAL materialisation remain separate engineering cases, not a claimed validated domain.

Method reference: Boyd, Parikh, Chu, Peleato and Eckstein (2011), [Distributed Optimization and Statistical Learning via ADMM](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html). The code and bounds below were implemented/derived here; they are not copied package code.

## What the certificate means

For any observation-wise loss dual variable w and coefficient multiplier nu, stationarity is `Z' w + V' nu = 0`. Gaussian `w=2(Z alpha-y)/n` gives the conjugate term

\[
K(w)=-y^\top w-\frac n4\|w\|_2^2.
\]

For logistic loss choose probabilities p with the observed outcome total, let `w=(p-y01)/n`, and set

\[
K(p)=\frac1n\sum_i[-p_i\log p_i-(1-p_i)\log(1-p_i)].
\]

The dual lower bound in exact arithmetic is `K-C*||nu||infinity`. For full-column-rank V, a candidate multiplier is corrected by

\[
\nu_c=\nu-V(V^\top V)^{-1}(Z^\top w+V^\top\nu).
\]

The implementation explicitly subtracts residual-error bounds `A*||Z'w+V'nu_c||2 + B*|sum(w)|`, where `A=C/sigma_min(V)` and, for two-class logistic samples, `B=|logit(mean(y01))|+A*max_i ||Z_i||2`. These follow by bounding the feasible coefficient vector and the profiled intercept. Gaussian B is zero. One-class logistic samples use exact constant dual probabilities and a finite limiting intercept: their likelihood infimum exists, but there is no finite intercept MLE.

Thus feasible risk minus a retained dual lower bound bounds global optimisation error. `converged=True` requires

`gap <= tol * (1 + abs(risk))`,

plus finite diagnostics, dual-residual and roundoff checks. `max_iter` and `polish_stalled` do not count as convergence. This is a **numerical** certificate, with floating-point rank/residual checks, not an interval-arithmetic or machine-checked proof.

## Evidence at matched radii

All cases and failures are retained in [comparison.csv](results/2026-10-27-ball-v1/comparison.csv).

| HAL n | Boundary risk / status | Ball risk | Ball gap | Ball iterations |
|---:|---:|---:|---:|---:|
| 32 | 10.321804 / stalled | 10.067462 | 4.54e-9 | 561 |
| 64 | 22.036617 / stalled | 21.673710 | 3.72e-8 | 558 |
| 128 | 46.209679 / stalled | 45.690566 | 4.03e-7 | 543 |
| 256 | 95.793912 / 10000 iterations | 95.034508 | 8.98e-7 | 547 |

These are the distinct-knot univariate HAL designs from the first audit, with C=1 and the same ridge initializer for both algorithms. All four ball fits are certified at `tol=1e-8`. Their coefficient SV is at most 1 to numerical precision. Boundary fits have no dual certificate; their infinite gap field means unavailable, not an estimated infinite error.

A further 20 Gaussian and 20 logistic orthonormal-transform stress cases all meet the same relative criterion. All 44 ball risks are below their matched boundary risk in this run. Maximum scaled gap across the 44 is 9.981e-9; maximum positive budget error is 1.78e-15. Gaussian absolute gaps can be around 1.28e-4 when risk is large: the tolerance is scaled, not an absolute 1e-8 assertion.

Ten independent SLSQP epigraph fits (five per loss) agree in risk to at most **8.68e-9**. The reference optimises alpha, the intercept where applicable, and separate epigraph variables, independently of the ADMM implementation and stopping rule. All reference solves succeed; none is excluded. See [independent_reference.csv](results/2026-10-27-ball-v1/independent_reference.csv).

The experiment measures optimisation, not predictive superiority or a statistical convergence rate. The HAL stress signal depends on n and does not have a fixed SV budget. A smaller training risk at the same feasible radius establishes a better fit to this optimisation problem, not better out-of-sample performance.

## Interfaces and validation

- Eighteen Python unit tests pass: nine ball tests and nine legacy boundary regression tests. They cover independent Gaussian/logistic references, zeros, interior fits, initial-direction changes at equal radius, small budgets, rank-deficient losses, numerical-rank truncation, unfinished status, extreme offsets and one-class limits, actual histories/predictions, and CV diagnostics.
- R native/public Gaussian and logistic tests pass, including both Gaussian initializers, boundary activation counterexample, interior optimum, predictions, controls, and all-fold/refit diagnostics.
- The existing R binomial suite passes **77 checks, zero failures and zero warnings**. This is the relevant existing suite, not a claim that every historical Python/package test was run.
- R and Python compile and load in isolated libraries. No installed user package was replaced. No commit, push, release or manuscript edit was performed.

Python `optimizer_output` exposes `status`, `converged`, `constraint`, `sv_radius`, `optimality_gap`, `dual_bound`, `dual_residual`, `constraint_violation`, and the feasible history. CV results also expose all candidate-fold statuses, gaps and radii (lambda-major flattened order). `tangent_norm` is a legacy boundary field and NaN on ball fits.

R uses attributes prefixed `sv_`; these include `sv_risk` and `sv_iterations`. Gaussian CV refit diagnostics are on `res_opt`; binomial outputs also expose them at the top level. CV diagnostic matrices have folds as rows and lambdas as columns. Missing or unfinished fits must not be interpreted as certified.

`tol` now controls the scaled primal-dual gap. `crit="grad"` and `crit="risk"` both use that criterion. `step_factor` remains accepted/validated for call compatibility but does not tune the ball solver. Initializer/lambda/C selection is retained. Help sources and generated R help were updated.

## SV and the intercept

C bounds the coefficient variation `||beta||1`, not the full anchored norm by the same number. In a centred Gaussian fit, `a0=mean(y)-means(H)' beta`, so the full anchored variation is at most `|mean(y)|+2C` for zero-order indicators. For a profiled binary-logistic fit using uncentred indicators, the anchored intercept lies between `logit(prevalence)-max(H beta)` and `logit(prevalence)-min(H beta)`, giving full variation at most `|logit(prevalence)|+2C` when both classes occur. These are conditional algebraic bounds, not assurances about the selected radius or prevalence asymptotics. With redundant atoms, coefficient variation is a valid upper bound; the canonical representation is needed for equality.

## Reproduction and remaining work

After rebuilding the package from the saved sources with NumPy/SciPy (Python), Eigen/pybind11/CMake (native extension), or Rcpp/RcppEigen (R):

```bash
python -m unittest discover -s tests -p 'test_sv_*.py' -v
Rscript tests/r/test_sv_descent.R
python examples/sv_ball/compare.py --output /tmp/hapc-sv-ball-run
```

[Run manifest](results/2026-10-27-ball-v1/manifest.json), [source snapshot](results/2026-10-27-ball-v1/source/), [previous boundary build sources](results/2026-10-27-ball-v1/previous_boundary_source/) and build/test logs are retained. The experiment script is also saved next to the run. For that standalone saved copy, run it against the source snapshot's built package and put `source/tests` on PYTHONPATH so the independent reference helper can be imported.

Next work for a defensible paper: audit SV CV and the way its radius varies with training size; Gaussian SV currently uses a basis built from all covariates and non-orthogonal row subsets to initialise each fold. Binomial SV likewise retains its existing global covariate-basis construction. The numerical certificate applies to each supplied finite-dimensional problem; it does not validate that CV pipeline. Verify approximation of the truth at the retained PC count and chosen radius; align the manuscript definition and KKT/score theory with the ball solver; then design the larger simulation campaign and inference validation. Historical NPIV/ATE results are not re-labelled as results of this new estimator.

## Session closure

[Development closure](../../sessions/2026-10-27_sv_and_ate_close.md) integrates decisions, pending CV/radius work and the separate ATE baseline. The earlier no-commit/push statement describes the validation stage; the requested closure commits and pushes the reviewed development work. No package release or manuscript submission is implied.
