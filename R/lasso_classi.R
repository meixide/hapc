#' Logistic LASSO HAPC helpers (binomial + norm = "1")
#'
#' Internal R helpers used by \code{\link{hapc}} and \code{\link{cv.hapc}} when
#' \code{family = "binomial"} and \code{norm = "1"}. These wrap
#' \code{glmnet::glmnet} on the rotated design \eqn{\tilde{X} = U \cdot
#' \mathrm{diag}(d)} so the L1 problem we solve is logistic LASSO in the
#' principal-component basis. The Python counterpart uses
#' \code{sklearn.linear_model.LogisticRegression(penalty="l1",
#' solver="liblinear", fit_intercept=FALSE)} on the same \eqn{\tilde{X}}.
#'
#' @keywords internal
#' @noRd
NULL

# --- internal: glmnet single-λ logistic LASSO on Xtilde = U %*% diag(d) -----
.hapc_binomial_lasso <- function(X, Y, max_degree, npcs, lambda,
                                 predict = NULL, center = TRUE) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required for family='binomial', norm='1'. ",
         "Install with install.packages('glmnet').")
  }
  if (!(lambda > 0)) stop("lambda must be > 0 for logistic LASSO.")

  des <- design.hapc(X, max_degree = max_degree, npcs = npcs, center = center)
  k <- length(des$d)
  Xtilde <- des$U[, seq_len(k), drop = FALSE] %*% diag(des$d[seq_len(k)],
                                                       nrow = k, ncol = k)

  Y_01 <- as.numeric(Y > 0)  # accepts {0,1} or {-1,+1}
  fit <- glmnet::glmnet(
    x = Xtilde, y = Y_01,
    family = "binomial",
    alpha = 1,
    lambda = lambda,
    intercept = FALSE,
    standardize = FALSE
  )
  alpha <- as.numeric(fit$beta)

  Y_pm1 <- ifelse(Y_01 == 1, 1, -1)
  eta <- as.numeric(Xtilde %*% alpha)
  ymu <- Y_pm1 * eta
  risk <- mean(ifelse(ymu > 0,
                      log1p(exp(-ymu)),
                      -ymu + log1p(exp(ymu))))

  predictions <- NULL
  probabilities <- NULL
  predicted_classes <- NULL
  if (!is.null(predict)) {
    Ktest <- cross_kernel.hapc(X, predict, max_degree = max_degree,
                               center = center)
    v <- des$U[, seq_len(k), drop = FALSE] %*%
         ((1 / (des$d[seq_len(k)] + 1e-12)) * alpha)
    log_odds <- as.numeric(Ktest %*% v)
    predictions <- log_odds
    probabilities <- 1 / (1 + exp(-log_odds))
    predicted_classes <- ifelse(probabilities > 0.5, 1, -1)
  }

  list(
    alpha = alpha,
    predictions = predictions,
    probabilities = probabilities,
    predicted_classes = predicted_classes,
    lambda = lambda,
    risk = risk,
    iter = 0L
  )
}

# --- internal: K-fold logistic-LASSO CV (same shape as binomial+sv CV) ------
.cv_hapc_binomial_lasso <- function(X, Y, max_degree, npcs, lambdas,
                                    nfolds, predict = NULL,
                                    center = TRUE, fold_seed = 12345L) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required for family='binomial', norm='1'. ",
         "Install with install.packages('glmnet').")
  }
  if (any(lambdas <= 0)) stop("All lambdas must be > 0 for logistic LASSO.")

  n <- nrow(X)
  if (length(Y) != n) stop("length(Y) must equal nrow(X).")
  L <- length(lambdas)

  fold_size <- n %/% nfolds
  folds <- ((seq_len(n) - 1L) %/% fold_size) + 1L
  folds[folds > nfolds] <- nfolds
  if (fold_size * nfolds < n) folds[(fold_size * nfolds + 1L):n] <- nfolds
  set.seed(fold_seed)
  folds <- sample(folds)

  Y_01 <- as.numeric(Y > 0)
  fold_dev <- matrix(NA_real_, nrow = nfolds, ncol = L)

  for (k in seq_len(nfolds)) {
    te <- which(folds == k)
    tr <- which(folds != k)
    if (length(te) == 0L || length(tr) == 0L) next
    Xtr <- X[tr, , drop = FALSE]; Ytr <- Y[tr]
    Xte <- X[te, , drop = FALSE]; Yte01 <- Y_01[te]

    for (j in seq_len(L)) {
      fit <- .hapc_binomial_lasso(
        Xtr, Ytr, max_degree = max_degree, npcs = npcs,
        lambda = lambdas[j], predict = Xte, center = center
      )
      probs <- pmin(pmax(fit$probabilities, 1e-15), 1 - 1e-15)
      dev <- ifelse(Yte01 == 1, -log(probs), -log(1 - probs))
      fold_dev[k, j] <- mean(dev)
    }
  }

  deviances <- colMeans(fold_dev, na.rm = TRUE)
  best_idx <- which.min(deviances)
  best_lambda <- lambdas[best_idx]

  full <- .hapc_binomial_lasso(
    X, Y, max_degree = max_degree, npcs = npcs,
    lambda = best_lambda, predict = predict, center = center
  )

  out <- list(
    deviances = deviances,
    lambdas = lambdas,
    best_lambda = best_lambda,
    res_opt = list(alpha = full$alpha)
  )
  if (!is.null(predict)) out$predictions <- full$probabilities
  out
}
