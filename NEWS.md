# Development version

- `norm="sv"` now fits the convex coefficient-SV ball `||V alpha||1 <= C`, with C still determined by the initial fit at lambda. Interior optima and activation of zero PC coefficients are allowed. This replaces the multiplicative equality-constrained fit and may change results.
- A numerical primal-dual gap controls convergence. Single and CV outputs expose optimisation diagnostics; iteration-limited outputs remain explicitly uncertified. `tol` is a scaled gap tolerance. `crit` and `step_factor` remain accepted for call compatibility; both crit values use the gap and step_factor is ignored by the ball solver.
- The previous boundary algorithm remains available through explicit native Python comparator functions. Its empirical results are kept separately.
- HAL PCs associated only with numerical eigenvalue roundoff are discarded when positive numerical rank exists.
- R/Python validation, matched-radius stress results, source snapshots and remaining CV/statistical assumptions are documented in `examples/sv_ball/README.md`. No release has been made.

- An independent R `hapc.ate.tmle` implementation and exploratory complete-data binary-treatment ATE examples are included; see `examples/ate_tmle/README.md`. No coverage or superiority claim is implied.
- Development decisions, verification limits and the next CV/radius audit are consolidated in `sessions/2026-10-27_sv_and_ate_close.md`.
