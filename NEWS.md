# Development version

- **Default changes (may change results).** In `cv.hapc()` / `cv_hapc()`:
  - `inductive = TRUE` where it is implemented: gaussian `norm = "1"/"2"` and binomial `norm = "sv"/"2"`.
  - `lambda_grid = "scaled"` for gaussian `norm = "1"/"2"`, with 40 points.
  - When these arguments are not given:
    - gaussian `norm = "sv"`, `approx = TRUE` and `family = "logit-hazard"` keep the previous CV and the fixed grid;
    - calls that give `lambdas`, `log_lambda_min` or `log_lambda_max` keep that grid but are inductive where implemented.
  - The internal CVs of `ate.hapc()`, `hazard.hapc()` and `hapc.ate.tmle()` give their log-lambda bounds, so they keep their grids and become inductive where implemented.
  - An explicit `inductive = TRUE` or `lambda_grid = "scaled"` where it is not implemented raises an error.
  - `selection = "refit"` and `clip = FALSE` are unchanged.
  - Gaussian `norm = "sv"` stays transductive by decision: it is kept as a theoretical artefact and is not developed further.
- **`max_degree` now defaults to 2** (main effects and two-way interactions) in every R and Python entry point: `hapc`, `cv.hapc`, `ate.hapc`, `hazard.hapc`, `kernel.hapc`, `cross_kernel.hapc`, `design.hapc` and their Python counterparts. `cv.hapc.treat()` and `hapc.ate.tmle()` already used 2. Values above `ncol(X)` act as `ncol(X)`.

- Inductive cross-validation for `cv.hapc()` / `cv_hapc()`, gaussian `norm = "1"` (PC-HAL) and `"2"` (PC-HAR). New arguments, identical in R and Python and backed by one C++ routine (`src/cv_inductive.cpp`). The defaults listed here were the initial ones; see the default changes above:
  - `inductive = FALSE`. With `TRUE`, the HAL kernel, its centring and its eigendecomposition are built from the training rows of each fold, and held-out rows are scored through `kernel_cross_call` with the training knots and centring. The previous CV builds one basis from all of `X`, so held-out covariates enter the knots, the centring and the PCs; and because the training rows of that basis are not orthonormal, its closed-form fold fits (`ridge_call`, `fast_pchal_call`) are not the training-fold minimisers. With `TRUE` they are exact (tests agree with direct solves to 1e-14).
  - `lambdas = NULL`: a user-supplied grid, also for the other families/norms.
  - `lambda_grid = "fixed"`. With `"scaled"`: 40 log-spaced values (`grid_length`), `(s1^2/n) * 10^[-9, 1]` for PC-HAR and `[1e-5, 1] * lambda_max` with `lambda_max = max_j s_j |u_j' y_c| / n` for PC-HAL, from the basis that is fitted. The fixed grid `exp([-5, -3])` does not scale with `n`.
  - `selection = "refit"` (full-sample refit at the selected lambda). `"fold_average"` averages the fold fits without refit; `"holdout"` uses one split (`train_frac = 0.8`) and returns the training-part fit. Only `"holdout"` with `inductive = TRUE` is covered by the hold-out selection theory for PC-HAR/PC-HAL; the full-sample refit is not, because its norm radius is not the validated one.
  - `clip = FALSE`. With `TRUE`, held-out and returned predictions are clipped to the training-outcome range of the fit that produced them.
  - The new path also returns `best_index`, `fold_mses`, `cv_predictions`, `foldid`, `fold_alphas`, `fold_intercepts`, `fold_ranks`, `intercept`, `rank` and `lambda_grid`.
  - `approx = TRUE` is available only with the previous CV.
- `inductive` is also available for binomial `norm = "sv"` and `"2"` (per-fold design via `pchal_des` on the training rows; the R entry point `pchal_cv_classi_call` takes it as a 15th argument). Binomial `norm = "1"` was already inductive. Gaussian `norm = "sv"` still uses the full-sample basis; `inductive = TRUE` raises there.
- `max_degree` is documented: 1 is main effects only (an additive model when `ncol(X) >= 2`); the full HAL dictionary is `max_degree = ncol(X)`.
- Tests: `tests/test_inductive_cv.py`, `tests/r/test_inductive_cv.R` (run from pytest by `tests/test_r_inductive_cv.py`): fold fits that do not depend on held-out covariates, exact fold minimisers against the explicit training-only dictionary, bit-identical reproduction of the previous CV with `inductive = FALSE`, grid, hold-out, fold averaging, clipping, binomial ridge against a training-only design.

- New exported `cv.hapc.treat()` (R only): binomial HAPC with treatment columns outside the HAL kernel and unpenalised (`penalty_factor = 0`, with automatic retries under separation), optional unpenalised standardised linear covariates, the `cv.hapc` folds and edge-extended lambda grid up to `lambda_max`. It is the engine of `learner_hapc_trat` in the `tmle2t` package, which previously reached into hapc internals; on the same inputs it returns bit-identical predictions. Tests: `tests/r/test_cv_hapc_treat.R` (without A it reproduces `cv.hapc(norm = "1")`). R package version 2.7.1.9000.

- `norm="sv"` now fits the convex coefficient-SV ball `||V alpha||1 <= C`, with C still determined by the initial fit at lambda. Interior optima and activation of zero PC coefficients are allowed. This replaces the multiplicative equality-constrained fit and may change results.
- A numerical primal-dual gap controls convergence. Single and CV outputs expose optimisation diagnostics; iteration-limited outputs remain explicitly uncertified. `tol` is a scaled gap tolerance. `crit` and `step_factor` remain accepted for call compatibility; both crit values use the gap and step_factor is ignored by the ball solver.
- The previous boundary algorithm remains available through explicit native Python comparator functions. Its empirical results are kept separately.
- HAL PCs associated only with numerical eigenvalue roundoff are discarded when positive numerical rank exists.
- R/Python validation, matched-radius stress results, source snapshots and remaining CV/statistical assumptions are documented in `examples/sv_ball/README.md`. No release has been made.

- An independent R `hapc.ate.tmle` implementation and exploratory complete-data binary-treatment ATE examples are included; see `examples/ate_tmle/README.md`. No coverage or superiority claim is implied.
- Development decisions, verification limits and the next CV/radius audit are consolidated in `sessions/2026-10-27_sv_and_ate_close.md`.
