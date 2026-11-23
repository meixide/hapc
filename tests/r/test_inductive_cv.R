# testthat checks for the inductive gaussian CV of cv.hapc(norm = "1"/"2")
# (inductive, lambdas, lambda_grid, selection, clip). Run with
# testthat::test_file(); tests/test_r_inductive_cv.py drives it from pytest.
# The references are built from the explicit HAL dictionary of the training
# rows only. Simulated data only.

library(testthat)
suppressPackageStartupMessages(library(hapc))
for (ns in c("glue", "waldo")) invisible(loadNamespace(ns))

dictionary <- function(Xk, Xe, max_degree) {
  p <- ncol(Xk)
  blocks <- list()
  for (k in seq_len(max_degree)) {
    for (s in utils::combn(p, k, simplify = FALSE)) {
      B <- matrix(1, nrow(Xe), nrow(Xk))
      for (j in s) B <- B * outer(Xe[, j], Xk[, j], ">=")
      blocks[[length(blocks) + 1L]] <- B
    }
  }
  do.call(cbind, blocks)
}

soft <- function(z, t) sign(z) * pmax(abs(z) - t, 0)

# Prediction at Xe of the exact minimiser of the training-fold problem.
# r = NULL with ridge: kernel-free normal equations on the dictionary.
direct_fold_prediction <- function(X, Y, tr, Xe, lam, l1, r, max_degree) {
  H <- dictionary(X[tr, , drop = FALSE], X[tr, , drop = FALSE], max_degree)
  hbar <- colMeans(H)
  Hc <- sweep(H, 2, hbar)
  ybar <- mean(Y[tr]); yc <- Y[tr] - ybar; nt <- length(tr)
  He <- sweep(dictionary(X[tr, , drop = FALSE], Xe, max_degree), 2, hbar)
  if (is.null(r) && !l1) {
    beta <- t(Hc) %*% solve(Hc %*% t(Hc) + nt * lam * diag(nt), yc)
    return(drop(ybar + He %*% beta))
  }
  v <- svd(Hc, nu = 0, nv = r)$v
  Z <- Hc %*% v; Ze <- He %*% v
  if (!l1) {
    a <- solve(crossprod(Z) + nt * lam * diag(r), crossprod(Z, yc))
  } else {
    a <- numeric(r); cs <- colSums(Z^2) / nt
    for (it in 1:1000) {
      a_old <- a
      for (j in seq_len(r)) {
        rj <- yc - Z %*% a + Z[, j] * a[j]
        a[j] <- soft(sum(Z[, j] * rj) / nt, lam) / cs[j]
      }
      if (max(abs(a - a_old)) < 1e-15) break
    }
  }
  drop(ybar + Ze %*% a)
}

sim <- function(n, d, seed = 0) {
  set.seed(seed)
  X <- matrix(runif(n * d), n)
  Y <- sin(2 * pi * X[, 1]) + (X[, d] > 0.5) + rnorm(n, sd = 0.3)
  list(X = X, Y = Y)
}

LAMS <- exp(seq(log(1e-5), log(1e-1), length.out = 6))

test_that("a fold fit does not depend on its held-out covariates", {
  s <- sim(60, 2)
  for (norm in c("1", "2")) {
    fit <- function(X, ind) cv.hapc(X, s$Y, norm = norm, max_degree = 2, npcs = 60,
                                    lambdas = LAMS, inductive = ind,
                                    selection = "fold_average")
    a <- fit(s$X, TRUE)
    X2 <- s$X
    te <- a$foldid == 1
    X2[te, ] <- runif(2 * sum(te))
    b <- fit(X2, TRUE)
    expect_identical(a$foldid, b$foldid)
    expect_equal(b$fold_alphas[[1]], a$fold_alphas[[1]], tolerance = 1e-12)
    expect_identical(b$fold_intercepts[1], a$fold_intercepts[1])
    a0 <- fit(s$X, FALSE); b0 <- fit(X2, FALSE)
    expect_gt(max(abs(b0$fold_alphas[[1]] - a0$fold_alphas[[1]])), 1e-6)
  }
})

test_that("inductive fold fits are the exact training-fold minimisers", {
  cases <- list(list(d = 1, md = 1, npcs = NULL), list(d = 1, md = 1, npcs = 10),
                list(d = 2, md = 2, npcs = NULL), list(d = 2, md = 2, npcs = 25))
  for (cs in cases) for (l1 in c(FALSE, TRUE)) {
    npcs <- if (l1 && is.null(cs$npcs)) 30L else cs$npcs
    n <- if (cs$d == 1) 60 else 50
    s <- sim(n, cs$d, seed = cs$d)
    res <- cv.hapc(s$X, s$Y, norm = if (l1) "1" else "2", max_degree = cs$md,
                   npcs = if (is.null(npcs)) n else npcs, lambdas = LAMS,
                   inductive = TRUE)
    err <- 0
    for (k in 1:5) {
      tr <- which(res$foldid != k); te <- which(res$foldid == k)
      expect_equal(res$fold_ranks[k], if (is.null(npcs)) length(tr) - 1L else npcs)
      for (j in seq_along(LAMS)) {
        ref <- direct_fold_prediction(s$X, s$Y, tr, s$X[te, , drop = FALSE], LAMS[j],
                                      l1, npcs, cs$md)
        err <- max(err, abs(res$cv_predictions[te, j] - ref))
      }
    }
    expect_lt(err, 1e-8)
  }
})

test_that("the original fold fits are not exact (documents the defect)", {
  s <- sim(60, 1, seed = 1)
  res <- cv.hapc(s$X, s$Y, norm = "2", npcs = 10, lambdas = LAMS,
                 selection = "fold_average")
  tr <- which(res$foldid != 1); te <- which(res$foldid == 1)
  ref <- direct_fold_prediction(s$X, s$Y, tr, s$X[te, , drop = FALSE], LAMS[1],
                                FALSE, 10, 1)
  expect_gt(max(abs(res$cv_predictions[te, 1] - ref)), 1e-3)
})

test_that("inductive = FALSE reproduces the original CV", {
  s <- sim(70, 2)
  Xp <- matrix(runif(18), 9)
  for (norm in c("1", "2")) {
    old <- cv.hapc(s$X, s$Y, norm = norm, max_degree = 2, npcs = 70, lambdas = LAMS,
                   predict = Xp)
    expect_null(old$fold_mses)
    new <- .Call("fasthal_cv_general_call", s$X, s$Y, 70L, LAMS, 0L, 5L, Xp, 2L,
                 TRUE, norm == "1", FALSE, "refit", 0.8, FALSE, PACKAGE = "hapc")
    expect_equal(new$mses, old$mses, tolerance = 1e-12)
    expect_equal(new$res_opt, old$res_opt, tolerance = 1e-12)
    expect_equal(drop(new$predictions), drop(old$predictions), tolerance = 1e-12)
    # default fixed grid unchanged
    d <- cv.hapc(s$X, s$Y, norm = norm, max_degree = 2, npcs = 70)
    expect_equal(d$lambdas, exp(seq(-5, -3, length.out = 10)))
  }
})

test_that("scaled grid", {
  s <- sim(80, 2)
  ridge <- cv.hapc(s$X, s$Y, norm = "2", max_degree = 2, inductive = TRUE,
                   lambda_grid = "scaled")
  lasso <- cv.hapc(s$X, s$Y, norm = "1", max_degree = 2, inductive = TRUE,
                   lambda_grid = "scaled")
  for (r in list(ridge, lasso)) {
    expect_length(r$lambdas, 40L)
    expect_identical(r$lambda_grid, "scaled")
    expect_true(all(diff(log(r$lambdas)) > 0))
  }
  expect_equal(ridge$lambdas[40] / ridge$lambdas[1], 1e10, tolerance = 1e-10)
  expect_equal(lasso$lambdas[40] / lasso$lambdas[1], 1e5, tolerance = 1e-10)
  s1 <- sapply(1:5, function(k) {
    tr <- ridge$foldid != k
    max(eigen(kernel.hapc(s$X[tr, ], max_degree = 2), only.values = TRUE)$values) / sum(tr)
  })
  expect_equal(ridge$lambdas[40], 10 * max(s1), tolerance = 1e-10)
  top <- sapply(lasso$fold_alphas, function(a) max(abs(a[, 40])))
  nxt <- sapply(lasso$fold_alphas, function(a) max(abs(a[, 39])))
  expect_true(all(top == 0) && any(nxt > 0))
  expect_length(cv.hapc(s$X, s$Y, norm = "2", lambda_grid = "scaled",
                        grid_length = 7)$lambdas, 7L)
})

test_that("hold-out selection returns the clipped training-part fit", {
  s <- sim(100, 2)
  Xp <- matrix(runif(100, -0.5, 1.5), 50)
  r <- cv.hapc(s$X, s$Y, norm = "2", max_degree = 2, inductive = TRUE,
               lambda_grid = "scaled", selection = "holdout", clip = TRUE, predict = Xp)
  tr <- r$foldid == 0
  expect_equal(sum(tr), 80L)
  expect_equal(dim(r$fold_mses), c(1L, 40L))
  expect_true(all(is.na(r$cv_predictions[tr, ])))
  expect_equal(r$res_opt, r$fold_alphas[[1]][, r$best_index])
  expect_equal(r$intercept, mean(s$Y[tr]))
  rg <- range(s$Y[tr])
  expect_true(all(r$predictions >= rg[1] & r$predictions <= rg[2]))
  ref <- direct_fold_prediction(s$X, s$Y, which(tr), Xp, r$best_lambda, FALSE, NULL, 2)
  expect_equal(drop(r$predictions), pmin(pmax(ref, rg[1]), rg[2]), tolerance = 1e-8)
})

test_that("fold averaging predicts with the mean of the fold fits", {
  s <- sim(60, 1)
  Xp <- matrix(runif(12), 12)
  r <- cv.hapc(s$X, s$Y, norm = "2", inductive = TRUE, selection = "fold_average",
               lambdas = LAMS, predict = Xp)
  expect_null(r$res_opt)
  ref <- rowMeans(sapply(1:5, function(k)
    direct_fold_prediction(s$X, s$Y, which(r$foldid != k), Xp, r$best_lambda, FALSE,
                           NULL, 1)))
  expect_equal(drop(r$predictions), ref, tolerance = 1e-8)
})

test_that("user grid and argument checks", {
  s <- sim(50, 1)
  for (ind in c(FALSE, TRUE))
    expect_equal(cv.hapc(s$X, s$Y, norm = "1", lambdas = c(0.3, 0.01, 0.001),
                         inductive = ind)$lambdas, c(0.3, 0.01, 0.001))
  expect_error(cv.hapc(s$X, s$Y, norm = "1", lambdas = 0.1, lambda_grid = "scaled"))
  expect_error(cv.hapc(s$X, s$Y, norm = "sv", inductive = TRUE))
  expect_error(cv.hapc(s$X, as.numeric(s$Y > 0), family = "binomial", norm = "2",
                       clip = TRUE))
  expect_error(cv.hapc(s$X, s$Y, norm = "2", inductive = TRUE, approx = TRUE))
})

test_that("binomial inductive CV runs and differs from the full-sample design", {
  s <- sim(80, 2, seed = 3)
  Y <- rbinom(80, 1, plogis(3 * s$X[, 1] - 1.5))
  a <- cv.hapc(s$X, Y, family = "binomial", norm = "2", max_degree = 2, npcs = 15,
               lambdas = c(1e-3, 1e-2), inductive = TRUE)
  b <- cv.hapc(s$X, Y, family = "binomial", norm = "2", max_degree = 2, npcs = 15,
               lambdas = c(1e-3, 1e-2))
  expect_true(all(is.finite(a$deviances)))
  expect_gt(max(abs(a$deviances - b$deviances)), 1e-6)
  sv <- cv.hapc(s$X, Y, family = "binomial", norm = "sv", npcs = 15,
                lambdas = c(1e-2, 1e-1), inductive = TRUE, max_iter = 500L)
  expect_true(all(is.finite(sv$deviances)))
})
