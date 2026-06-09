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

# --- internal: classify a binomial response vector --------------------------
# Returns one of:
#   "01"   hard labels in {0,1}
#   "pm1"  hard labels in {-1,+1}
#   "soft" fractional labels in [0,1] (e.g. EM-HAL E-step posteriors)
# Errors if any value falls outside [0,1] and the set is not exactly {-1,+1}.
.hapc_binary_label_kind <- function(Y) {
  Y <- as.numeric(Y)
  u <- unique(Y[!is.na(Y)])
  if (all(u %in% c(0, 1))) return("01")
  if (setequal(u, c(-1, 1))) return("pm1")
  if (all(u >= 0 & u <= 1)) return("soft")
  stop("family='binomial' requires Y in {0,1}, {-1,+1}, or soft labels in ",
       "[0,1]; found values outside [0,1].", call. = FALSE)
}

# --- internal: validate labels + enforce the soft-label norm restriction ----
# Soft labels are supported only for norm in {"1","2"}; norm="sv" is rejected.
# A warning is emitted whenever soft (non-binary) labels are detected.
.hapc_check_binomial_labels <- function(Y, norm) {
  kind <- .hapc_binary_label_kind(Y)
  if (kind == "soft") {
    if (identical(norm, "sv")) {
      stop("Soft labels (Y in (0,1)) are not implemented for norm='sv'; ",
           "use norm='1' or norm='2'.", call. = FALSE)
    }
    warning("Non-binary labels detected in Y: treating them as soft labels ",
            "in [0,1] (cross-entropy target). Supported only for norm='1' ",
            "and norm='2'.", call. = FALSE)
  }
  invisible(kind)
}

# --- internal: glmnet single-λ logistic LASSO on Xtilde = U %*% diag(d) -----
.hapc_binomial_lasso <- function(X, Y, max_degree, npcs, lambda,
                                 predict = NULL, center = TRUE) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required for family='binomial', norm='1'. ",
         "Install with install.packages('glmnet').")
  }
  if (!(lambda > 0)) stop("lambda must be > 0 for logistic LASSO.")

  .calibrate_intercept <- function(y01, eta) {
    b0 <- 0
    for (it in seq_len(50)) {
      z <- eta + b0
      p <- 1 / (1 + exp(-z))
      g <- sum(p - y01)
      h <- sum(p * (1 - p))
      if (abs(g) < 1e-10 || h < 1e-12) break
      b0 <- b0 - g / h
    }
    b0
  }

  des <- design.hapc(X, max_degree = max_degree, npcs = npcs, center = center)
  k <- length(des$d)
  Xtilde <- des$U[, seq_len(k), drop = FALSE] %*% diag(des$d[seq_len(k)],
                                                       nrow = k, ncol = k)

  # Map the response to a soft target in [0,1]: {0,1} -> as is, {-1,+1} ->
  # (Y+1)/2, fractional soft labels pass through unchanged.
  kind <- .hapc_binary_label_kind(Y)
  Y_soft <- if (kind == "pm1") (as.numeric(Y) + 1) / 2 else as.numeric(Y)

  # glmnet's two-column response form (failures, successes) accepts fractional
  # "counts", so cbind(1 - q, q) encodes the soft cross-entropy target. On
  # hard {0,1} labels this reduces to the ordinary Bernoulli fit.
  fit <- glmnet::glmnet(
    x = Xtilde, y = cbind(1 - Y_soft, Y_soft),
    family = "binomial",
    alpha = 1,
    lambda = lambda,
    intercept = FALSE,
    standardize = FALSE
  )
  alpha <- as.numeric(fit$beta)
  b0 <- .calibrate_intercept(Y_soft, as.numeric(Xtilde %*% alpha))

  eta <- as.numeric(Xtilde %*% alpha + b0)
  p <- pmin(pmax(1 / (1 + exp(-eta)), 1e-15), 1 - 1e-15)
  risk <- mean(-(Y_soft * log(p) + (1 - Y_soft) * log(1 - p)))

  predictions <- NULL
  probabilities <- NULL
  predicted_classes <- NULL
  if (!is.null(predict)) {
    Ktest <- cross_kernel.hapc(X, predict, max_degree = max_degree,
                               center = center)
    v <- des$U[, seq_len(k), drop = FALSE] %*%
         ((1 / (des$d[seq_len(k)] + 1e-12)) * alpha)
    log_odds <- as.numeric(Ktest %*% v + b0)
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

  # Soft target in [0,1] used for the held-out cross-entropy deviance.
  kind <- .hapc_binary_label_kind(Y)
  Y_soft <- if (kind == "pm1") (as.numeric(Y) + 1) / 2 else as.numeric(Y)
  fold_dev <- matrix(NA_real_, nrow = nfolds, ncol = L)

  for (k in seq_len(nfolds)) {
    te <- which(folds == k)
    tr <- which(folds != k)
    if (length(te) == 0L || length(tr) == 0L) next
    Xtr <- X[tr, , drop = FALSE]; Ytr <- Y[tr]
    Xte <- X[te, , drop = FALSE]; Yte <- Y_soft[te]

    for (j in seq_len(L)) {
      fit <- .hapc_binomial_lasso(
        Xtr, Ytr, max_degree = max_degree, npcs = npcs,
        lambda = lambdas[j], predict = Xte, center = center
      )
      probs <- pmin(pmax(fit$probabilities, 1e-15), 1 - 1e-15)
      dev <- -(Yte * log(probs) + (1 - Yte) * log(1 - probs))
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
