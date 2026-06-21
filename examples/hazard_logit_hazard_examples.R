# =============================================================================
# family = "logit-hazard": discrete-time logistic hazard via HAPC
# =============================================================================
# Five worked examples (the unified pipeline in hazard.R shipped two setups;
# these are three more, plus two that echo the original flavours). Each example:
#
#   1. simulates baseline covariates X, a latent event time T^event from a known
#      discrete hazard, and an independent censoring time C;
#   2. forms the *observed* data the user actually has:
#         T      = min(T^event, C)        (observed time)
#         Delta  = 1(T^event <= C)        (event indicator)
#   3. fits  hazard.hapc(X, T, Delta, ...)  -- the family = "logit-hazard" wrapper;
#   4. draws TWO diagnostics:
#         (left)  CV risk (logistic deviance) vs lambda, with the selected
#                 lambda marked -- we check it is an *interior* grid point;
#         (right) true hazard vs estimated hazard on the person-period rows.
#
# Run:  Rscript examples/hazard_logit_hazard_examples.R
# Output: examples/hazard_logit_hazard_examples.png  + a console summary table.
# =============================================================================

suppressMessages(library(hapc))
set.seed(2024)

# ---- helper: simulate one right-censored discrete-survival data set ----------
# true_haz(t, x) returns P(event in interval t | survived, covariates x).
simulate_survival <- function(n, grid, true_haz, gen_X, cens_grid = grid) {
  X <- gen_X(n)
  Tev <- integer(n)
  for (i in seq_len(n)) {
    Tev[i] <- max(grid)                       # administrative censoring at end
    for (t in grid) {
      if (rbinom(1L, 1L, true_haz(t, X[i, ])) == 1L) { Tev[i] <- t; break }
    }
  }
  Cen   <- sample(cens_grid, n, replace = TRUE)
  Tobs  <- pmin(Tev, Cen)
  Delta <- as.integer(Tev <= Cen)
  list(X = X, T = Tobs, Delta = Delta, grid = grid, true_haz = true_haz)
}

# ---- helper: fit + collect diagnostics --------------------------------------
# Fits hazard.hapc and then *adaptively widens* the log-lambda grid: if the CV
# optimum lands on a grid endpoint, we extend that side and refit. This makes
# the "interior optimum" check hold robustly regardless of the random data /
# language RNG, rather than relying on hand-tuned ranges.
run_example <- function(name, sim, max_degree, norm = "1",
                        log_lambda_min = -7, log_lambda_max = 3,
                        grid_length = 18, max_expand = 4L) {
  lo <- log_lambda_min; hi <- log_lambda_max
  for (attempt in 0:max_expand) {
    fit <- hazard.hapc(
      X = sim$X, T = sim$T, Delta = sim$Delta,
      norm = norm, max_degree = max_degree, time_grid = sim$grid,
      log_lambda_min = lo, log_lambda_max = hi,
      grid_length = grid_length, nfolds = 5
    )
    if (isTRUE(fit$interior)) break
    if (which.min(fit$risk) == 1L) lo <- lo - 2 else hi <- hi + 2
  }
  truth <- mapply(function(t, i) sim$true_haz(t, sim$X[i, ]),
                  fit$data$time, fit$data$id)
  fit$truth   <- truth
  fit$name    <- name
  fit$rcorr   <- cor(truth, fit$hazard)
  fit$slope   <- coef(lm(fit$hazard ~ truth))[2]
  fit$log_lam <- c(lo, hi)
  fit
}

# ---- helper: two-panel plot for one example ---------------------------------
plot_example <- function(fit) {
  # (left) CV risk vs lambda
  plot(fit$lambdas, fit$risk, log = "x", pch = 16, col = "darkgreen",
       xlab = expression(lambda), ylab = "CV logistic deviance",
       main = sprintf("%s\nCV risk vs lambda  (interior: %s)",
                      fit$name, fit$interior))
  abline(v = fit$best_lambda, col = "red", lty = 2)
  legend("topleft", bty = "n", cex = 0.8,
         legend = sprintf("best lambda = %.3g", fit$best_lambda))

  # (right) true vs estimated hazard
  plot(fit$truth, fit$hazard, pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.35),
       xlab = "True hazard", ylab = "Estimated hazard",
       main = sprintf("%s\nr = %.3f, slope = %.3f",
                      fit$name, fit$rcorr, fit$slope))
  abline(0, 1, col = "red", lwd = 1.5, lty = 2)
}

# =============================================================================
# The five data-generating processes
# =============================================================================
examples <- list()

# (1) Linear-in-time, additive covariates (echoes hazard.R setup_1) ------------
examples$linear_additive <- list(
  sim = simulate_survival(
    n = 300, grid = 1:6,
    true_haz = function(t, x) plogis(-2.6 + 0.28 * t + 1.2 * x[1] - 0.9 * x[2]),
    gen_X = function(n) cbind(x1 = runif(n), x2 = runif(n))),
  max_degree = 1, norm = "2")

# (2) Non-linear time effect with a bump at t = 3 (echoes setup_2) -------------
examples$bump_time <- list(
  sim = simulate_survival(
    n = 300, grid = 1:5,
    true_haz = function(t, x) {
      te <- if (t == 3) 1.5 else 0.2 * t
      plogis(-2.5 + te + 2.0 * x[1])
    },
    gen_X = function(n) cbind(age = runif(n, 0.1, 0.9))),
  max_degree = 2, norm = "1")

# (3) NEW: time x covariate interaction ----------------------------------------
examples$time_interaction <- list(
  sim = simulate_survival(
    n = 300, grid = 1:7,
    true_haz = function(t, x) plogis(-3.0 + 0.15 * t + 1.4 * x[1] +
                                       0.45 * t * x[1] - 0.6 * x[2]),
    gen_X = function(n) cbind(x1 = runif(n), x2 = runif(n))),
  max_degree = 2, norm = "2")

# (4) NEW: U-shaped (non-monotone) hazard in time ------------------------------
examples$ushaped_time <- list(
  sim = simulate_survival(
    n = 300, grid = 1:6,
    true_haz = function(t, x) plogis(-2.9 + 0.55 * abs(t - 3.5) + 1.1 * x[1]),
    gen_X = function(n) cbind(x1 = runif(n))),
  max_degree = 2, norm = "1")

# (5) NEW: three covariates with a nonlinear (threshold) effect ----------------
examples$three_cov_nonlinear <- list(
  sim = simulate_survival(
    n = 300, grid = 1:8,
    true_haz = function(t, x) plogis(-2.7 + 0.18 * t + 1.3 * x[1] - 1.0 * x[2] +
                                       0.9 * as.numeric(x[3] > 0.5)),
    gen_X = function(n) cbind(x1 = runif(n), x2 = runif(n), x3 = runif(n))),
  max_degree = 2, norm = "2")

# =============================================================================
# Fit all, plot, summarise
# =============================================================================
fits <- lapply(names(examples), function(nm) {
  e <- examples[[nm]]
  run_example(nm, e$sim, max_degree = e$max_degree, norm = e$norm)
})
names(fits) <- names(examples)

out_png <- file.path("examples", "hazard_logit_hazard_examples.png")
png(out_png, width = 1100, height = 2000, res = 130)
par(mfrow = c(5, 2), mar = c(4.2, 4.2, 3.6, 1))
invisible(lapply(fits, plot_example))
dev.off()
cat("Saved figure to", out_png, "\n\n")

summary_tbl <- do.call(rbind, lapply(fits, function(f) data.frame(
  example     = f$name,
  pp_rows     = nrow(f$data),
  events      = sum(f$data$Y),
  best_lambda = signif(f$best_lambda, 4),
  interior    = f$interior,
  r           = round(f$rcorr, 3),
  slope       = round(f$slope, 3),
  row.names   = NULL
)))
cat("Summary (interior = best lambda strictly inside the CV grid):\n")
print(summary_tbl, row.names = FALSE)

if (all(summary_tbl$interior)) {
  cat("\nAll five CV optima are interior grid points.\n")
} else {
  cat("\nNOTE: widen log_lambda_min/max for examples with interior = FALSE.\n")
}
