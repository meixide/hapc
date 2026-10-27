# HAPC point-treatment ATE TMLE

Initial implementation, 23 October 2026. R API: `hapc.ate.tmle(X, Y, A, ...)`.
Independent implementation of standard TMLE score equations; no source code
copied from tmle or tmle3 and no SuperLearner/tmle/tmle3 runtime dependency.

## Scope and use

```r
fit <- hapc.ate.tmle(W, Y, A, cf_folds = 5, norm = "1")
c(estimate = fit$estimate, se = fit$se,
  lower = fit$lower, upper = fit$upper)
fit$diagnostics
```

Binary point treatment, baseline numeric covariates, independent complete-data
observations. Binary outcomes use logistic fluctuation. Continuous outcomes use
an identity fluctuation unless fixed `outcome_bounds` are supplied, in which
case targeting uses a logistic fluctuation on the rescaled outcome. No outcome
bounds are inferred from sample extrema. ATT/ATC, outcome missingness, repeated
measures, survival and proximal models are outside this first implementation.

Both nuisance fits use `cv.hapc` inside the outer training folds. The default
norm is explicitly `"1"`; `"2"` is also supported. Both inner tuning and outer
partitions are recorded. A known randomized propensity can be supplied as
`g = 0.5`. External initial predictions can be supplied as `Q = cbind(Q0,Q1)`
and `g`, which permits comparisons that hold nuisances fixed; out-of-fold
provenance of externally supplied predictions is the caller's responsibility.

The pooled two-coordinate targeting step solves the two arm-specific residual
scores using covariates A/g and (1-A)/(1-g). These coordinates are orthogonal
on observed treatment arms. The returned estimate averages Q1-star minus
Q0-star. The EIF is

```
A/g * (Y-Q1-star) - (1-A)/(1-g) * (Y-Q0-star)
  + Q1-star - Q0-star - estimate
```

The standard error is sd(EIF)/sqrt(n); the Wald interval is not truncated.
Pooled targeting updates the cross-fitted initial predictions using outcomes;
only the initial nuisance fits are out-of-fold. There is no claim that
targeting, zero empirical scores or cross-fitting alone proves coverage.
`cf_folds=1` explicitly selects ordinary in-sample TMLE.

Propensity clipping, initial Q clipping, raw propensity range, fitted lambdas,
targeting coefficients and score residuals are returned. Non-finite inputs and
non-finite logistic fluctuation solutions fail explicitly. Estimator warnings
are not swallowed. The existing `ate.hapc` API and estimator are unchanged.

## Verification

- Package built and installed into an isolated temporary R library from the
  current source tree, including pre-existing local modifications. The global
  R installation was not replaced.
- 26 expectations in `tests/r/test_ate_tmle.R` pass: comparison against an
  independent offset GLM and least squares, EIF scores, standard errors,
  scale equivariance, bounded continuous outcomes, invalid inputs, clipping,
  RNG preservation, native HAPC fitting, and held-out outcome perturbation.
- Additional native binomial norm="2" smoke fit passed; documentation checked
  with tools::checkRd. Documentation generated with roxygen2.

Run tests against an installed development build:

```sh
Rscript -e 'testthat::test_file("tests/r/test_ate_tmle.R", stop_on_failure=TRUE)'
```

## First simulation: exploratory, not a coverage certification

Run: `examples/ate_tmle/results/2026-10-23-current-smoke-40/`.
Script: `examples/ate_tmle_simulation.R`. Reproduce in a **new** output directory:

```sh
Rscript examples/ate_tmle_simulation.R /tmp/hapc-tmle-new-run 40
```

Two independent uniform covariates, nonlinear baseline outcome, moderate
confounding and overlap. Continuous ATE = 0.8; binary ATE computed by nested
numerical integration. For each outcome and n=100/300: 40 seeds, outer 3-fold
cross-fitting, inner 3-fold CV, degree 2, norm="1", eight log-lambda values
from -8 to 2, rank floor(n_training^0.8), g bound 0.025. Plug-in, one-step
and TMLE share initial predictions; oracle nuisances provide a separate
targeting/inference check. Plug-in intervals are deliberately not reported
using a naive variance that ignores nuisance fitting.

| Outcome | n | One-step RMSE | TMLE RMSE | One-step coverage | TMLE coverage |
|---|---:|---:|---:|---:|---:|
| Binary | 100 | 0.0875 | 0.0862 | 97.5% | 97.5% |
| Binary | 300 | 0.0539 | 0.0538 | 100% | 100% |
| Continuous | 100 | 0.1622 | 0.1609 | 97.5% | 97.5% |
| Continuous | 300 | 0.0726 | 0.0721 | 97.5% | 97.5% |

All 160 HAPC and 160 oracle attempts returned estimates. 12 HAPC fits
produced glmnet partial-path convergence warnings; these are retained per
seed and require attention before a larger validation campaign. There were no silently excluded attempts.
The Monte Carlo standard error at 97.5% coverage is about 2.5 percentage
points. Wilson intervals for Monte Carlo coverage are saved, including for
40/40 coverage (which does not establish perfect coverage). This small
experiment shows no clear superiority over one-step; intervals are generally
conservative. Small RMSE differences do not establish an improvement.

Saved evidence: per-seed CSV, summary CSV with coverage uncertainty, config,
sessionInfo, test log and code hashes. The oracle comparison uses the same
generative mechanism and observations but true Q and g. These are entirely
synthetic data.

## References and next experiments

- tmle 2.1.1 source/API: https://github.com/cran/tmle
- tmle3 framework: https://tlverse.org/tmle3/articles/framework.html
- Bounded outcome fluctuations: Gruber and van der Laan (2010),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3126669/

Next: investigate partial-path convergence warnings, increase replication
counts, test weak overlap, rare outcomes and model misspecification, and
compare in-sample vs cross-fitted TMLE with matched initial fits. Test an
upstream tmle installation with externally supplied identical nuisances
before claiming numerical package equivalence. The current independent GLM
comparison checks the fluctuation equations, not equivalence of all defaults
or functionality of upstream packages. Extend to other estimands only after
this ATE baseline has adequate evidence.

The final validation was rebuilt against the source after concurrent engine
edits; all 26 expectations passed again and the 40-seed campaign was repeated.
Exact R/native source snapshots are archived with both runs. The earlier
`2026-10-23-smoke-40` directory is retained as historical evidence.

## Session closure — 27 October

The 26 test expectations also pass against the final isolated R build after the SV changes. This does not rerun the 160-fit campaign or resolve its 12 partial-path warnings. [Development closure](../../sessions/2026-10-27_sv_and_ate_close.md) distinguishes this ATE baseline from SV-ball optimisation and records the next work. Commit/push at closure does not constitute a release.
