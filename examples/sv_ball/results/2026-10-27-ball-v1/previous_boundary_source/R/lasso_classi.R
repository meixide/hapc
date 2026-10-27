#' Logistic LASSO HAPC helpers (binomial + norm = "1")
#'
#' Internal R helpers used by \code{\link{hapc}} and \code{\link{cv.hapc}} when
#' \code{family = "binomial"} and \code{norm = "1"}. These wrap
#' \code{glmnet::glmnet} on the rotated design \eqn{\tilde{X} = U \cdot
#' \mathrm{diag}(d)} so the L1 problem we solve is logistic LASSO in the
#' principal-component basis. U and d come from the eigendecomposition of the
#' n x n HAL kernel (\code{\link{kernel.hapc}}), not from
#' \code{\link{design.hapc}}, so memory is O(n^2) rather than
#' O(n^2 \eqn{\sum_k \binom{p}{k}}). The Python counterpart uses
#' \code{sklearn.linear_model.LogisticRegression} (L1, \code{solver = "saga"},
#' unpenalised intercept) on the same \eqn{\tilde{X}}.
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

# --- internal: response as a soft target in [0,1] ----------------------------
# {0,1} -> as is, {-1,+1} -> (Y+1)/2, fractional soft labels pass through.
.hapc_soft01 <- function(Y) {
  kind <- .hapc_binary_label_kind(Y)
  if (kind == "pm1") (as.numeric(Y) + 1) / 2 else as.numeric(Y)
}

# --- internal: PC basis of the HAL design from the kernel --------------------
# U and d are the left singular vectors / singular values of the (centred) HAL
# design H, obtained from the eigendecomposition of K = H H^T. This is what
# design.hapc() computes, but design.hapc() materialises H itself
# (n x n * sum_k choose(p, k), ~11 GB for n = 1764, p = 29, max_degree = 2)
# while K is n x n. Same npcs clamp as design.hapc(): n - 1 if centred, else n.
#
# Eigenvector signs are arbitrary and solver dependent (LAPACK here, Eigen in
# design.hapc()), so each column is flipped to make its largest-|.| entry
# positive (the convention of sklearn's svd_flip); the Python binomial
# norm = "1" path applies the same rule, so alpha agrees across languages.
# Directions with d <= 1e-7 * d[1] are numerically null: they are excluded
# from the fit (their alpha is 0) and from the 1/d prediction map.
#
# The uncentred kernel's column means are kept so that predictions can centre
# cross_kernel.hapc(..., center = FALSE) directly: cross_kernel.hapc() with
# center = TRUE would recompute the whole n x n training kernel just for them.
.hapc_kernel_basis <- function(X, max_degree, npcs, center) {
  n <- nrow(X)
  K <- kernel.hapc(X, max_degree = max_degree, center = FALSE)
  colmean <- colMeans(K)
  grand <- mean(colmean)
  # Same double centring as kernel.hapc(center = TRUE); K is symmetric.
  if (center) K <- K - outer(colmean, colmean, "+") + grand
  e <- eigen(K, symmetric = TRUE)
  rm(K)
  k <- max(1L, min(as.integer(npcs), if (center) n - 1L else n))
  U <- e$vectors[, seq_len(k), drop = FALSE]
  d <- sqrt(pmax(e$values[seq_len(k)], 0))
  rm(e)
  s <- sign(U[cbind(apply(abs(U), 2L, which.max), seq_len(k))])
  s[s == 0] <- 1
  U <- U * rep(s, each = n)
  list(X = X, U = U, d = d, keep = d > 1e-7 * d[1L],
       max_degree = max_degree, center = center,
       colmean = colmean, grand = grand)
}

# Rotated design Xtilde = U diag(d) restricted to the kept directions.
.hapc_basis_xtilde <- function(base) {
  base$U[, base$keep, drop = FALSE] * rep(base$d[base$keep], each = nrow(base$U))
}

# Map from PC coefficients (kept directions) to log-odds at new points:
# Ktest %*% U diag(1/d), so that eta_new = P %*% alpha + b0.
.hapc_basis_predictor <- function(base, Xnew) {
  Kn <- cross_kernel.hapc(base$X, Xnew, max_degree = base$max_degree,
                          center = FALSE)
  # Centring of cross_kernel.hapc(center = TRUE), with the stored means.
  if (base$center) {
    Kn <- Kn - rowMeans(Kn) - rep(base$colmean, each = nrow(Kn)) + base$grand
  }
  Kn %*% (base$U[, base$keep, drop = FALSE] *
            rep(1 / base$d[base$keep], each = nrow(base$U)))
}

# --- internal: validate the `intercept` argument ----------------------------
# Every binomial fit has an unpenalised intercept estimated jointly with the
# PC coefficients; a gaussian fit has an intercept exactly when center = TRUE
# (the response is centred). `intercept = NULL` (default) means "whatever the
# model fits"; an explicit value that the model cannot honour is an error
# rather than being silently ignored.
.hapc_check_intercept <- function(intercept, family, center) {
  if (is.null(intercept)) return(invisible(TRUE))
  if (!is.logical(intercept) || length(intercept) != 1L || is.na(intercept)) {
    stop("'intercept' must be NULL, TRUE or FALSE.", call. = FALSE)
  }
  if (family %in% c("binomial", "logit-hazard") && !intercept) {
    stop("intercept = FALSE is not supported for family = '", family, "': ",
         "the intercept is always estimated jointly and left unpenalised ",
         "(fitting with the intercept fixed at 0 centres the logistic fit at ",
         "p = 1/2 instead of the prevalence). Leave 'intercept' unset.",
         call. = FALSE)
  }
  if (family == "gaussian" && !identical(intercept, center)) {
    stop(if (intercept)
           "intercept = TRUE requires center = TRUE for family = 'gaussian': without centring no intercept is fitted."
         else
           "intercept = FALSE is incompatible with center = TRUE for family = 'gaussian': centring fits the intercept (the mean of Y). Use center = FALSE for a fit without intercept.",
         call. = FALSE)
  }
  invisible(TRUE)
}

# --- internal: glmnet logistic LASSO over a lambda path ----------------------
# Fits glmnet on Xt along `path` and returns coefficients at each value of `s`
# (columns of B) plus intercepts b0. coef(fit, s = s) is used rather than
# fit$beta because glmnet may stop the path early on convergence and return
# fewer lambdas than requested; coef() then reuses the last fitted solution.
#
# glmnet's two-column response form (failures, successes) accepts fractional
# "counts", so cbind(1 - q, q) encodes the soft cross-entropy target. On
# hard {0,1} labels this reduces to the ordinary Bernoulli fit.
#
# The intercept is fitted jointly and never penalised (glmnet does not
# penalise it). Centring makes the columns of Xt orthogonal to 1, which removes
# the intercept from least squares but not from logistic regression: fitting
# with b0 = 0 (hapc <= 2.6.0) centres the fit at p = 1/2 instead of the
# prevalence.
.hapc_glmnet_path <- function(Xt, q, path, s) {
  k <- ncol(Xt)
  # glmnet needs at least two columns; a zero column never enters the model.
  if (k < 2L) Xt <- cbind(Xt, matrix(0, nrow(Xt), 2L - k))
  fit <- glmnet::glmnet(
    x = Xt, y = cbind(1 - q, q),
    family = "binomial",
    alpha = 1,
    lambda = sort(unique(path), decreasing = TRUE),
    intercept = TRUE,
    standardize = FALSE
  )
  cf <- as.matrix(stats::coef(fit, s = s))
  B <- unname(cf[1L + seq_len(k), , drop = FALSE])
  b0 <- unname(cf[1L, ])
  list(B = B, b0 = b0)
}

# --- internal: glmnet single-λ logistic LASSO on Xtilde = U %*% diag(d) -----
# `path` is the lambda sequence glmnet walks down (warm starts); the returned
# fit is the one at `lambda`. hapc() fits lambda alone, as before.
.hapc_binomial_lasso <- function(X, Y, max_degree, npcs, lambda,
                                 predict = NULL, center = TRUE,
                                 path = lambda) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required for family='binomial', norm='1'. ",
         "Install with install.packages('glmnet').")
  }
  if (!(lambda > 0)) stop("lambda must be > 0 for logistic LASSO.")

  base <- .hapc_kernel_basis(X, max_degree, npcs, center)
  Xtilde <- .hapc_basis_xtilde(base)
  Y_soft <- .hapc_soft01(Y)

  cf <- .hapc_glmnet_path(Xtilde, Y_soft, path = c(path, lambda), s = lambda)
  alpha <- numeric(length(base$d))
  alpha[base$keep] <- cf$B[, 1L]
  b0 <- cf$b0[1L]

  eta <- as.numeric(Xtilde %*% alpha[base$keep] + b0)
  p <- pmin(pmax(1 / (1 + exp(-eta)), 1e-15), 1 - 1e-15)
  risk <- mean(-(Y_soft * log(p) + (1 - Y_soft) * log(1 - p)))

  # Binomial output convention (all norms): predictions = probabilities,
  # log_odds = linear predictor.
  predictions <- NULL
  log_odds <- NULL
  if (!is.null(predict)) {
    log_odds <- as.numeric(.hapc_basis_predictor(base, predict) %*%
                             alpha[base$keep] + b0)
    predictions <- 1 / (1 + exp(-log_odds))
  }

  list(
    alpha = alpha,
    intercept = b0,
    predictions = predictions,
    log_odds = log_odds,
    lambda = lambda,
    risk = risk,
    iter = 0L
  )
}

# --- internal: K-fold logistic-LASSO CV (same shape as binomial+sv CV) ------
# The kernel basis is computed once per fold and glmnet fits the whole lambda
# grid on it. If extend_grid > 0 and the CV optimum sits on an edge of the
# grid, the grid is extended on that side (same log spacing, another
# length(lambdas) points) up to extend_grid times: downwards from the smallest
# lambda, upwards from the largest one but never past lambda_max, the
# smallest lambda at which every PC coefficient is 0 (intercept-only model).
.cv_hapc_binomial_lasso <- function(X, Y, max_degree, npcs, lambdas,
                                    nfolds, predict = NULL,
                                    center = TRUE, fold_seed = 12345L,
                                    extend_grid = 0L) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required for family='binomial', norm='1'. ",
         "Install with install.packages('glmnet').")
  }
  if (any(lambdas <= 0)) stop("All lambdas must be > 0 for logistic LASSO.")

  n <- nrow(X)
  if (length(Y) != n) stop("length(Y) must equal nrow(X).")
  lambdas <- sort(lambdas)
  L0 <- length(lambdas)

  fold_size <- n %/% nfolds
  folds <- ((seq_len(n) - 1L) %/% fold_size) + 1L
  folds[folds > nfolds] <- nfolds
  if (fold_size * nfolds < n) folds[(fold_size * nfolds + 1L):n] <- nfolds
  set.seed(fold_seed)
  folds <- sample(folds)

  # Soft target in [0,1] used for the fit and the held-out cross-entropy.
  Y_soft <- .hapc_soft01(Y)

  # Per fold: rotated training design and held-out predictor map. Only these
  # (n_train x k and n_test x k) are kept, not the kernels.
  fold_data <- lapply(seq_len(nfolds), function(k) {
    te <- which(folds == k)
    tr <- which(folds != k)
    if (length(te) == 0L || length(tr) == 0L) return(NULL)
    base <- .hapc_kernel_basis(X[tr, , drop = FALSE], max_degree, npcs, center)
    list(Xt = .hapc_basis_xtilde(base), q = Y_soft[tr],
         P = .hapc_basis_predictor(base, X[te, , drop = FALSE]),
         q_te = Y_soft[te])
  })

  # Mean held-out deviance at each lambda in lams (nfolds x length(lams)).
  fold_deviance <- function(lams) {
    t(vapply(fold_data, function(f) {
      if (is.null(f)) return(rep(NA_real_, length(lams)))
      cf <- .hapc_glmnet_path(f$Xt, f$q, path = lams, s = lams)
      eta <- f$P %*% cf$B + rep(cf$b0, each = nrow(f$P))
      probs <- pmin(pmax(1 / (1 + exp(-eta)), 1e-15), 1 - 1e-15)
      colMeans(-(f$q_te * log(probs) + (1 - f$q_te) * log(1 - probs)))
    }, numeric(length(lams))))
  }

  fold_dev <- matrix(fold_deviance(lambdas), nrow = nfolds)
  deviances <- colMeans(fold_dev, na.rm = TRUE)

  # lambda_max: at the intercept-only fit the gradient of the mean loss
  # w.r.t. alpha is Xt'(q - mean(q)) / n; the largest over folds bounds every
  # fold's path.
  lambda_max <- max(vapply(fold_data, function(f) {
    if (is.null(f)) return(0)
    max(abs(crossprod(f$Xt, f$q - mean(f$q)))) / length(f$q)
  }, numeric(1)))
  cv_dev <- function(lams) colMeans(matrix(fold_deviance(lams), nrow = nfolds),
                                    na.rm = TRUE)

  step <- if (L0 > 1L) log(lambdas[2L]) - log(lambdas[1L]) else log(10)
  n_extend <- 0L
  while (n_extend < extend_grid) {
    b <- which.min(deviances)
    L <- length(lambdas)
    if (b == 1L) {
      extra <- exp(log(lambdas[1L]) - step * rev(seq_len(L0)))
      deviances <- c(cv_dev(extra), deviances)
      lambdas <- c(extra, lambdas)
    } else if (b == L && L > 1L && lambdas[L] < lambda_max) {
      extra <- exp(log(lambdas[L]) + step * seq_len(L0))
      extra <- c(extra[extra < lambda_max], lambda_max)
      deviances <- c(deviances, cv_dev(extra))
      lambdas <- c(lambdas, extra)
    } else {
      break
    }
    n_extend <- n_extend + 1L
  }

  best_idx <- which.min(deviances)
  best_lambda <- lambdas[best_idx]
  L <- length(lambdas)
  lambda_position <- if (L == 1L) NA_character_ else
    if (best_lambda >= lambda_max * (1 - 1e-8)) "null_model" else
    if (best_idx == 1L) "lower_edge" else
    if (best_idx == L) "upper_edge" else "interior"
  if (extend_grid > 0L && lambda_position %in% c("lower_edge", "upper_edge")) {
    warning("cv.hapc(family='binomial', norm='1'): the CV-optimal lambda is ",
            "still on the ", sub("_edge", "", lambda_position),
            " edge of the grid after ", n_extend, " extension(s).",
            call. = FALSE)
  }

  # Full-data refit walks the grid down from its largest value to best_lambda.
  full <- .hapc_binomial_lasso(
    X, Y, max_degree = max_degree, npcs = npcs,
    lambda = best_lambda, predict = predict, center = center,
    path = lambdas[lambdas >= best_lambda]
  )

  out <- list(
    deviances = deviances,
    lambdas = lambdas,
    best_lambda = best_lambda,
    best_index = best_idx,
    lambda_position = lambda_position,
    n_extend = n_extend,
    lambda_max = lambda_max,
    res_opt = list(alpha = full$alpha, intercept = full$intercept)
  )
  if (!is.null(predict)) out$predictions <- full$predictions
  out
}
