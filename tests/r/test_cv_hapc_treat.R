# testthat checks for cv.hapc.treat() (binomial HAPC with unpenalised
# treatment columns). Run with testthat::test_file(). Simulated data only.

library(testthat)
suppressPackageStartupMessages(library(hapc))
for (ns in c("glue", "waldo")) invisible(loadNamespace(ns))

sim_rare <- function(seed, n = 400, p = 3, prev = 0.05, effect = 1.5) {
  set.seed(seed)
  X <- matrix(runif(n * p), n)
  A <- matrix(rbinom(n, 1, prev), n)
  Y <- rbinom(n, 1, plogis(sin(pi * X[, 1]) + X[, 2] - 1.2 + effect * A[, 1]))
  list(X = X, A = A, Y = Y)
}

args <- list(max_degree = 2L, log_lambda_min = -8, log_lambda_max = -1,
             grid_length = 15L, extend_grid = 3L)

test_that("without A or linear terms it is the cv.hapc logistic LASSO", {
  d <- sim_rare(1)
  npcs <- floor(nrow(d$X)^0.7)
  a <- do.call(cv.hapc.treat, c(list(d$X, d$Y, npcs = npcs, predict = d$X), args))
  b <- cv.hapc(d$X, d$Y, family = "binomial", norm = "1", max_degree = 2L,
               npcs = npcs, nfolds = 5L, log_lambda_min = -8, log_lambda_max = -1,
               grid_length = 15L, extend_grid = 3L, predict = d$X)
  expect_equal(a$best_lambda, b$best_lambda, tolerance = 1e-6)
  expect_equal(a$predictions, as.numeric(b$predictions), tolerance = 1e-4)
})

test_that("a rare unpenalised treatment keeps its coefficient", {
  d <- sim_rare(2)
  npcs <- floor(nrow(d$X)^0.7)
  r0 <- do.call(cv.hapc.treat, c(list(d$X, d$Y, A = d$A, npcs = npcs,
                                      predict = d$X, predict_A = d$A), args))
  expect_identical(r0$penalty_factor, 0)
  co <- as.matrix(stats::coef(r0$fit, s = r0$best_lambda))
  k <- nrow(co)                       # intercept, PCs, then A
  expect_gt(co[k, 1], 0.5)
  expect_true(r0$lambda_position %in% c("interior", "lower_edge", "upper_edge", "null_model"))
  expect_length(r0$predictions, nrow(d$X))
  expect_true(all(r0$predictions > 0 & r0$predictions < 1))
  # predictions under A = 1 vs A = 0 differ in the direction of the effect
  p1 <- do.call(cv.hapc.treat, c(list(d$X, d$Y, A = d$A, npcs = npcs, predict = d$X,
                                      predict_A = matrix(1, nrow(d$X))), args))$predictions
  p0 <- do.call(cv.hapc.treat, c(list(d$X, d$Y, A = d$A, npcs = npcs, predict = d$X,
                                      predict_A = matrix(0, nrow(d$X))), args))$predictions
  expect_gt(mean(p1 - p0), 0.05)
})

test_that("linear terms enter unpenalised and inputs are validated", {
  d <- sim_rare(3)
  npcs <- floor(nrow(d$X)^0.7)
  r <- do.call(cv.hapc.treat, c(list(d$X, d$Y, A = d$A, linear = 2L, npcs = npcs,
                                     predict = d$X[1:5, ], predict_A = d$A[1:5, , drop = FALSE]), args))
  co <- as.matrix(stats::coef(r$fit, s = r$best_lambda))
  expect_true(co[nrow(co), 1] != 0)   # the linear term (last column)
  expect_length(r$predictions, 5L)
  expect_error(cv.hapc.treat(d$X, d$Y, A = d$A, predict = d$X), "predict_A")
  expect_error(cv.hapc.treat(d$X, d$Y * 2), "\\[0, 1\\]")
})
