# testthat checks for the R binomial + norm = "1" path (logistic LASSO).
# Run with testthat::test_file(); tests/test_r_binomial_lasso.py drives it
# from pytest. Simulated data only.

library(testthat)
suppressPackageStartupMessages(library(hapc))
# testthat loads waldo lazily for its global-state check; with some library
# versions waldo only loads if glue is already loaded, so load both first.
for (ns in c("glue", "waldo")) invisible(loadNamespace(ns))

sim_binary <- function(seed, n = 90, p = 3, shift = 0) {
  set.seed(seed)
  X <- matrix(runif(n * p), n)
  pr <- plogis(2 * sin(pi * X[, 1]) + 1.5 * X[, 2] * X[, 3] - 1.5 - shift)
  list(X = X, pr = pr, Y = rbinom(n, 1, pr), Xnew = matrix(runif(30 * p), 30))
}

# Reference: the PC basis from design.hapc() (materialises the HAL design, fine
# at this size) and a separate glmnet fit with an unpenalised intercept
# (intercept = FALSE gives the <= 2.6.0 estimator, kept for comparisons).
reference_fit <- function(X, q, max_degree, npcs, lambda, Xnew,
                          intercept = TRUE) {
  des <- design.hapc(X, max_degree = max_degree, npcs = npcs, center = TRUE)
  k <- length(des$d)
  Xt <- des$U[, seq_len(k), drop = FALSE] %*% diag(des$d, k, k)
  fit <- glmnet::glmnet(Xt, cbind(1 - q, q), family = "binomial", alpha = 1,
                        lambda = lambda, intercept = intercept,
                        standardize = FALSE)
  a <- as.numeric(fit$beta)
  b0 <- if (intercept) as.numeric(fit$a0) else {
    eta <- as.numeric(Xt %*% a); b <- 0
    for (it in 1:50) {
      p <- plogis(eta + b); g <- sum(p - q); h <- sum(p * (1 - p))
      if (abs(g) < 1e-10) break
      b <- b - g / h
    }
    b
  }
  Kn <- cross_kernel.hapc(X, Xnew, max_degree = max_degree, center = TRUE)
  list(abs_alpha = abs(a),
       log_odds = as.numeric(Kn %*% (des$U %*% (a / des$d)) + b0))
}

skip_if_not_installed("glmnet")

test_that("kernel-based fit reproduces the design.hapc-based fit", {
  for (s in 1:3) {
    D <- sim_binary(s)
    for (lam in c(0.005, 0.03)) {
      new <- hapc(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
                  npcs = 40, lambda = lam, predict = D$Xnew)
      ref <- reference_fit(D$X, D$Y, 2, 40, lam, D$Xnew)
      expect_equal(new$log_odds, ref$log_odds, tolerance = 1e-6)
      expect_equal(new$predictions, plogis(ref$log_odds), tolerance = 1e-6)
      expect_equal(abs(new$alpha), ref$abs_alpha, tolerance = 1e-6)
    }
  }
})

test_that("{-1,+1} labels give the same fit as {0,1}", {
  D <- sim_binary(4)
  a <- hapc(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
            npcs = 40, lambda = 0.01, predict = D$Xnew)
  b <- hapc(D$X, 2 * D$Y - 1, family = "binomial", norm = "1",
            max_degree = 2, npcs = 40, lambda = 0.01, predict = D$Xnew)
  expect_equal(a$predictions, b$predictions)
})

test_that("soft labels in [0,1] fit the cross-entropy target", {
  D <- sim_binary(5)
  q <- D$pr  # fractional labels, e.g. ltmle pseudo-outcomes
  expect_warning(
    fit <- hapc(D$X, q, family = "binomial", norm = "1", max_degree = 2,
                npcs = 40, lambda = 0.01, predict = D$Xnew),
    "soft labels")
  ref <- reference_fit(D$X, q, 2, 40, 0.01, D$Xnew)
  expect_equal(fit$log_odds, ref$log_odds, tolerance = 1e-6)
  expect_true(all(fit$predictions > 0 & fit$predictions < 1))

  # Unpenalised intercept: fitted probabilities average to mean(q).
  fit1 <- suppressWarnings(
    hapc(D$X, q, family = "binomial", norm = "1", max_degree = 2,
         npcs = 40, lambda = 0.01, predict = D$X))
  expect_equal(mean(fit1$predictions), mean(q), tolerance = 1e-4)

  expect_warning(
    cv <- cv.hapc(D$X, q, family = "binomial", norm = "1", max_degree = 2,
                  npcs = 40, log_lambda_min = -7, log_lambda_max = -2,
                  grid_length = 8, predict = D$Xnew),
    "soft labels")
  expect_true(all(is.finite(cv$deviances)))
  expect_true(all(cv$predictions > 0 & cv$predictions < 1))

  # Labels exactly in {0,1} are not treated as soft.
  expect_silent(hapc(D$X, D$Y, family = "binomial", norm = "1",
                     max_degree = 2, npcs = 40, lambda = 0.01))
})

test_that("the fit matches glmnet with an unpenalised intercept (rare outcome)", {
  D <- sim_binary(6, n = 120, shift = 2)  # rare outcome
  fit <- hapc(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
              npcs = 50, lambda = 0.01, predict = D$Xnew)
  ref <- reference_fit(D$X, D$Y, 2, 50, 0.01, D$Xnew)
  expect_equal(fit$log_odds, ref$log_odds, tolerance = 1e-5)
  # and differs from the <= 2.6.0 estimator (intercept fixed at 0 in the fit)
  old <- reference_fit(D$X, D$Y, 2, 50, 0.01, D$Xnew, intercept = FALSE)
  expect_gt(max(abs(fit$log_odds - old$log_odds)), 1e-3)
  expect_length(fit$intercept, 1L)
  expect_true(is.finite(fit$intercept))
})

test_that("CV fits the lambda path once per fold and agrees with per-lambda fits", {
  D <- sim_binary(7, n = 100)
  cv <- cv.hapc(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
                npcs = 50, log_lambda_min = -6, log_lambda_max = -2,
                grid_length = 6, nfolds = 4, predict = D$Xnew)
  # Per-lambda refits on the same folds (the previous loop).
  n <- nrow(D$X); fs <- n %/% 4
  folds <- pmin(((seq_len(n) - 1L) %/% fs) + 1L, 4L)
  set.seed(12345L); folds <- sample(folds)
  dev1 <- sapply(cv$lambdas, function(l) mean(sapply(1:4, function(k) {
    te <- folds == k
    f <- hapc(D$X[!te, ], D$Y[!te], family = "binomial", norm = "1",
              max_degree = 2, npcs = 50, lambda = l, predict = D$X[te, ])
    p <- pmin(pmax(f$predictions, 1e-15), 1 - 1e-15)
    mean(-(D$Y[te] * log(p) + (1 - D$Y[te]) * log(1 - p)))
  })))
  best <- which.min(dev1)
  expect_equal(cv$best_lambda, cv$lambdas[best])
  expect_equal(cv$deviances[best], dev1[best], tolerance = 1e-4)
  expect_equal(cv$best_index, best)
  expect_true(cv$lambda_position %in%
                c("interior", "lower_edge", "upper_edge", "null_model"))
  expect_length(cv$predictions, nrow(D$Xnew))
})

test_that("extend_grid moves the grid down when the optimum is on its lower edge", {
  D <- sim_binary(8, n = 150)
  args <- list(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
               npcs = 60, log_lambda_min = -2, log_lambda_max = -1,
               grid_length = 4)
  cv0 <- do.call(cv.hapc, args)
  expect_equal(cv0$lambda_position, "lower_edge")
  expect_equal(cv0$n_extend, 0L)
  cv2 <- suppressWarnings(do.call(cv.hapc, c(args, extend_grid = 3L)))
  expect_gt(cv2$n_extend, 0L)
  expect_length(cv2$lambdas, 4L * (1L + cv2$n_extend))
  expect_false(is.unsorted(cv2$lambdas))
  expect_lt(cv2$best_lambda, min(cv0$lambdas))
  expect_equal(tail(cv2$deviances, 4L), cv0$deviances, tolerance = 1e-8)
})

test_that("extend_grid moves the grid up, but not past lambda_max", {
  D <- sim_binary(8, n = 150)
  args <- list(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
               npcs = 60, log_lambda_min = -12, log_lambda_max = -10,
               grid_length = 4)
  cv0 <- do.call(cv.hapc, args)
  expect_equal(cv0$lambda_position, "upper_edge")
  cv2 <- suppressWarnings(do.call(cv.hapc, c(args, extend_grid = 6L)))
  expect_gt(cv2$n_extend, 0L)
  expect_false(is.unsorted(cv2$lambdas))
  expect_gt(cv2$best_lambda, max(cv0$lambdas))
  expect_lte(max(cv2$lambdas), cv2$lambda_max * (1 + 1e-12))
  expect_true(cv2$lambda_position %in% c("interior", "null_model"))
})

test_that("an intercept-only truth is reported as null_model", {
  set.seed(3); n <- 150
  X <- matrix(runif(n * 3), n); Y <- rbinom(n, 1, 0.1)
  cv <- suppressWarnings(cv.hapc(X, Y, family = "binomial", norm = "1",
                                 max_degree = 2, npcs = 40,
                                 log_lambda_min = -6, log_lambda_max = -3,
                                 grid_length = 5, extend_grid = 8L,
                                 predict = X))
  expect_true(cv$lambda_position %in% c("interior", "null_model"))
  # unpenalised intercept: mean fitted probability equals the prevalence
  expect_equal(mean(cv$predictions), mean(Y), tolerance = 0.02)
})

test_that("extend_grid also extends the sv and ridge binomial grids", {
  D <- sim_binary(9, n = 120, shift = 2)   # rare outcome
  for (nm in c("2", "sv")) {
    args <- list(D$X, D$Y, family = "binomial", norm = nm, max_degree = 2,
                 npcs = 30, log_lambda_min = -14, log_lambda_max = -12,
                 grid_length = 3, nfolds = 3)
    cv0 <- do.call(cv.hapc, args)
    expect_null(cv0$lambda_position)       # default: unchanged output
    cv2 <- suppressWarnings(do.call(cv.hapc, c(args, extend_grid = 6L)))
    expect_true(cv2$n_extend >= 0L)
    expect_false(is.unsorted(cv2$lambdas))
    expect_true(cv2$lambda_position %in% c("interior", "lower_edge", "upper_edge"))
    expect_equal(cv2$best_lambda, cv2$lambdas[cv2$best_index])
  }
})

test_that("sv and ridge binomial fits are calibrated (unpenalised intercept)", {
  D <- sim_binary(10, n = 200, shift = 2.5)
  for (nm in c("2", "sv")) {
    f <- hapc(D$X, D$Y, family = "binomial", norm = nm, max_degree = 2,
              npcs = 40, lambda = 0.05, predict = D$X)
    expect_equal(mean(f$predictions), mean(D$Y), tolerance = 1e-3)
    expect_equal(f$predictions, plogis(f$log_odds), tolerance = 1e-8)
  }
})


test_that("impossible intercept configurations are errors", {
  D <- sim_binary(11)
  for (nm in c("1", "2", "sv")) {
    expect_error(hapc(D$X, D$Y, family = "binomial", norm = nm, max_degree = 2,
                      npcs = 20, lambda = 0.01, intercept = FALSE),
                 "intercept = FALSE is not supported")
    expect_error(cv.hapc(D$X, D$Y, family = "binomial", norm = nm,
                         max_degree = 2, npcs = 20, grid_length = 3,
                         intercept = FALSE),
                 "intercept = FALSE is not supported")
  }
  expect_error(hapc(D$X, D$Y, family = "binomial", norm = "1", npcs = 20,
                    lambda = 0.01, intercept = NA), "must be NULL, TRUE or FALSE")
  y <- D$X[, 1] + rnorm(nrow(D$X), sd = 0.1)
  expect_error(hapc(D$X, y, norm = "1", npcs = 20, lambda = 0.01,
                    intercept = FALSE), "incompatible with center = TRUE")
  expect_error(hapc(D$X, y, norm = "1", npcs = 20, lambda = 0.01,
                    center = FALSE, intercept = TRUE), "requires center = TRUE")
  # consistent explicit values are accepted
  expect_silent(hapc(D$X, D$Y, family = "binomial", norm = "1", max_degree = 2,
                     npcs = 20, lambda = 0.01, intercept = TRUE))
  expect_no_error(hapc(D$X, y, norm = "1", npcs = 20, lambda = 0.01,
                       center = FALSE, intercept = FALSE))
})

test_that("binomial ridge (norm = '2') matches glmnet(alpha = 0) on the same basis", {
  D <- sim_binary(12, n = 120, shift = 1.5)
  npcs <- 30; lam <- 0.02
  fit <- hapc(D$X, D$Y, family = "binomial", norm = "2", max_degree = 2,
              npcs = npcs, lambda = lam, predict = D$Xnew)
  des <- design.hapc(D$X, max_degree = 2, npcs = npcs, center = TRUE)
  Xt <- des$U %*% diag(des$d)
  g <- glmnet::glmnet(Xt, D$Y, family = "binomial", alpha = 0, lambda = lam,
                      standardize = FALSE, intercept = TRUE, thresh = 1e-14)
  a <- as.numeric(g$beta)
  Kn <- cross_kernel.hapc(D$X, D$Xnew, max_degree = 2, center = TRUE)
  ref <- as.numeric(Kn %*% (des$U %*% (a / des$d)) + g$a0)
  expect_equal(fit$log_odds, ref, tolerance = 1e-5)
})
