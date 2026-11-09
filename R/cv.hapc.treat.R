#' Cross-validated binomial HAPC with unpenalised treatment columns
#'
#' Logistic LASSO on the HAPC basis of the covariates \code{X}, plus a block of
#' columns \code{A} (typically binary treatments) that enter \emph{outside} the
#' HAL kernel as ordinary linear terms with a reduced (by default zero)
#' penalty. Optionally, some covariates (\code{linear}) also enter as
#' unpenalised standardised linear terms while staying inside the kernel, so
#' that the lasso shrinks the HAPC part towards a linear logistic model instead
#' of towards the intercept-only model.
#'
#' Why: in the HAL kernel a rare binary column lives in low-variance principal
#' components, which the lasso on \eqn{U \mathrm{diag}(d)} penalises heavily.
#' With a rare treatment the outcome regression then ignores the treatment and
#' a propensity model under-predicts continuation. Taking the treatment out of
#' the kernel, unpenalised, keeps its coefficient. The treatment effect is
#' additive on the logit scale (as in a main-terms logistic regression); the
#' covariate part is HAPC.
#'
#' The penalty factor really has to be 0: with a rare treatment the gradient of
#' its coefficient is of the order of prevalence times effect, so even a factor
#' of 0.1 keeps it at 0. Under (quasi-)separation glmnet may fail to converge
#' with factor 0 and return a truncated path; the fit is then retried with the
#' next value of \code{retry_factors} and the factor used is returned.
#'
#' Folds are those of \code{\link{cv.hapc}} (block partition, shuffled with
#' \code{set.seed(fold_seed)}). The lambda grid is log-spaced and, as in
#' \code{cv.hapc(extend_grid = )}, extended on the edge where the optimum
#' falls: downwards by another \code{grid_length} points, upwards never past
#' \code{lambda_max}, the smallest lambda at which every PC coefficient is 0
#' (intercept, \code{A} and \code{linear} terms only).
#'
#' \code{Y} may contain soft labels in \[0, 1\] (fit by glmnet with the
#' two-column response \code{cbind(1 - Y, Y)}); labels are not recoded.
#'
#' Note: \code{set.seed(fold_seed)} changes the global random number stream,
#' as \code{cv.hapc} does.
#'
#' R only; there is no Python counterpart yet.
#'
#' @param X Numeric matrix of covariates for the HAL kernel (n x p, p >= 1,
#'   no constant columns).
#' @param Y Numeric vector in \[0, 1\] of length n.
#' @param A Numeric matrix (n x q) of columns entering outside the kernel with
#'   penalty factor \code{penalty_factor}, or \code{NULL}.
#' @param linear Columns of \code{X} (logical, integer indices or names) that
#'   also enter as unpenalised linear terms, standardised with their
#'   full-sample mean and sd, or \code{NULL}.
#' @param penalty_factor Relative penalty of the \code{A} columns (0 =
#'   unpenalised).
#' @param max_degree,npcs HAL kernel degree and number of principal
#'   components (clamped to n - 1).
#' @param log_lambda_min,log_lambda_max,grid_length Log-spaced lambda grid.
#' @param nfolds Number of CV folds.
#' @param extend_grid Maximum number of grid extensions when the optimum lies
#'   on an edge.
#' @param predict Numeric matrix of new \code{X} rows (same columns), or
#'   \code{NULL}.
#' @param predict_A Numeric matrix of new \code{A} rows; required when both
#'   \code{A} and \code{predict} are given.
#' @param fold_seed Seed for the fold shuffle.
#' @param retry_factors Penalty factors tried, in order, when glmnet returns a
#'   truncated path with \code{penalty_factor}.
#'
#' @return A list with \code{best_lambda}, \code{best_index},
#'   \code{lambda_position} (\code{"interior"}, \code{"lower_edge"},
#'   \code{"upper_edge"} or \code{"null_model"}), \code{n_extend},
#'   \code{lambdas}, \code{deviances} (CV cross-entropy per lambda),
#'   \code{lambda_max}, \code{penalty_factor} (the factor actually used in the
#'   final fit), \code{fit} (the final \code{glmnet} object) and, if
#'   \code{predict} is given, \code{predictions} (probabilities).
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 400
#' X <- matrix(runif(n * 3), n)
#' A <- matrix(rbinom(n, 1, 0.05), n, dimnames = list(NULL, "A1"))
#' Y <- rbinom(n, 1, plogis(sin(pi * X[, 1]) - 1 + 1.2 * A[, 1]))
#' cv <- cv.hapc.treat(X, Y, A = A, max_degree = 2, npcs = floor(n^0.7),
#'                     log_lambda_min = -8, log_lambda_max = -1,
#'                     grid_length = 15, extend_grid = 3,
#'                     predict = X, predict_A = A)
#' cv$lambda_position
#' }
#' @export
cv.hapc.treat <- function(X, Y, A = NULL, linear = NULL, penalty_factor = 0,
                          max_degree = 2L, npcs = floor(nrow(X)^0.7),
                          log_lambda_min = -8, log_lambda_max = -1,
                          grid_length = 15L, nfolds = 5L, extend_grid = 3L,
                          predict = NULL, predict_A = NULL,
                          fold_seed = 12345L,
                          retry_factors = c(0.01, 0.03, 0.1, 0.3, 1)) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required for cv.hapc.treat(). ",
         "Install with install.packages('glmnet').")
  }
  X <- as.matrix(X); storage.mode(X) <- "double"; dimnames(X) <- NULL
  n <- nrow(X)
  if (ncol(X) < 1L) stop("X needs at least one column.")
  if (length(Y) != n) stop("length(Y) must equal nrow(X).")
  if (any(Y < 0 | Y > 1)) stop("Y must lie in [0, 1].")
  if (is.null(A)) A <- matrix(0, n, 0L)
  A <- as.matrix(A); storage.mode(A) <- "double"; dimnames(A) <- NULL
  if (nrow(A) != n) stop("nrow(A) must equal nrow(X).")
  es_l <- rep(FALSE, ncol(X))
  if (!is.null(linear)) {
    if (is.logical(linear)) es_l <- linear
    else if (is.character(linear)) stop("Use logical or integer `linear` once X has no names; ",
                                        "convert names with colnames(X) %in% linear before calling.")
    else es_l[linear] <- TRUE
  }
  if (length(es_l) != ncol(X)) stop("`linear` must index the columns of X.")
  if (!is.null(predict)) {
    predict <- as.matrix(predict); storage.mode(predict) <- "double"; dimnames(predict) <- NULL
    if (ncol(predict) != ncol(X)) stop("predict must have the columns of X.")
    if (ncol(A) > 0L) {
      if (is.null(predict_A)) stop("predict_A is required when A is given.")
      predict_A <- as.matrix(predict_A); storage.mode(predict_A) <- "double"; dimnames(predict_A) <- NULL
      if (nrow(predict_A) != nrow(predict) || ncol(predict_A) != ncol(A))
        stop("predict_A must have nrow(predict) rows and the columns of A.")
    } else predict_A <- matrix(0, nrow(predict), 0L)
  }
  q_t <- ncol(A); q_l <- sum(es_l)
  npcs <- max(1L, as.integer(npcs))
  lambdas <- exp(seq(log_lambda_min, log_lambda_max, length.out = grid_length))

  ## unpenalised block: A (factor penalty_factor) and the standardised linear
  ## covariates (factor 0), which also stay inside the kernel
  mu_l <- colMeans(X[, es_l, drop = FALSE]); sd_l <- apply(X[, es_l, drop = FALSE], 2, stats::sd)
  free <- function(Xm, Am) cbind(Am, scale(Xm[, es_l, drop = FALSE], mu_l, sd_l))
  design <- function(Xc, Ff) {
    b <- .hapc_kernel_basis(Xc, max_degree, npcs, TRUE)
    list(base = b, Z = unname(cbind(.hapc_basis_xtilde(b), Ff)))
  }
  lasso <- function(Z, q, lams, pf = penalty_factor) {
    f <- suppressWarnings(glmnet::glmnet(
      Z, cbind(1 - q, q), family = "binomial", alpha = 1, lambda = sort(lams, decreasing = TRUE),
      penalty.factor = c(rep(1, ncol(Z) - q_t - q_l), rep(pf, q_t), rep(0, q_l)),
      intercept = TRUE, standardize = FALSE))
    if (length(f$lambda) < length(lams) && pf < 1 && q_t > 0L) {
      nxt <- retry_factors[retry_factors > pf]
      if (length(nxt)) return(lasso(Z, q, lams, nxt[1]))
    }
    attr(f, "pf") <- pf
    f
  }

  ## CV folds as in cv.hapc; each fold keeps its design so that extending
  ## the grid only refits glmnet
  tam <- n %/% nfolds
  folds <- pmin(((seq_len(n) - 1L) %/% tam) + 1L, nfolds)
  set.seed(fold_seed); folds <- sample(folds)
  fd <- lapply(seq_len(nfolds), function(k) {
    tr <- folds != k; te <- !tr
    D <- design(X[tr, , drop = FALSE], free(X[tr, , drop = FALSE], A[tr, , drop = FALSE]))
    list(Z = D$Z, q = Y[tr], q_te = Y[te],
         Zte = unname(cbind(.hapc_basis_predictor(D$base, X[te, , drop = FALSE]),
                            free(X[te, , drop = FALSE], A[te, , drop = FALSE]))))
  })
  ## matrix(): with a single lambda (upward extension next to lambda_max)
  ## vapply returns a vector
  cv_dev <- function(lams) rowMeans(matrix(vapply(fd, function(f) {
    m <- lasso(f$Z, f$q, lams)
    p <- pmin(pmax(stats::predict(m, f$Zte, s = lams, type = "response"), 1e-15), 1 - 1e-15)
    colMeans(-(f$q_te * log(p) + (1 - f$q_te) * log(1 - p)))
  }, numeric(length(lams))), nrow = length(lams)))
  ## lambda_max: gradient of the mean loss at the fit with intercept and free
  ## columns only, with glmnet's rescaling of penalty.factor (factors sum to
  ## the number of columns)
  k_pc <- ncol(fd[[1]]$Z) - q_t - q_l
  lambda_max <- max(vapply(fd, function(f) {
    Tt <- f$Z[, -seq_len(k_pc), drop = FALSE]
    r <- f$q - suppressWarnings(stats::glm.fit(cbind(1, Tt), f$q, family = stats::quasibinomial()))$fitted.values
    max(abs(crossprod(f$Z[, seq_len(k_pc), drop = FALSE], r))) / length(r) / (ncol(f$Z) / k_pc)
  }, numeric(1)))
  dev <- cv_dev(lambdas)
  L0 <- length(lambdas); step <- log(lambdas[2]) - log(lambdas[1]); n_ext <- 0L
  while (n_ext < extend_grid) {
    b <- which.min(dev); L <- length(lambdas)
    if (b == 1L) {
      extra <- exp(log(lambdas[1]) - step * rev(seq_len(L0)))
      dev <- c(cv_dev(extra), dev); lambdas <- c(extra, lambdas)
    } else if (b == L && lambdas[L] < lambda_max) {
      extra <- exp(log(lambdas[L]) + step * seq_len(L0))
      extra <- c(extra[extra < lambda_max], lambda_max)
      dev <- c(dev, cv_dev(extra)); lambdas <- c(lambdas, extra)
    } else break
    n_ext <- n_ext + 1L
  }
  b <- which.min(dev); L <- length(lambdas)
  D <- design(X, free(X, A))
  fit <- lasso(D$Z, Y, lambdas[lambdas >= lambdas[b]])
  out <- list(
    best_lambda = lambdas[b], best_index = b,
    lambda_position = if (lambdas[b] >= lambda_max * (1 - 1e-8)) "null_model" else
      if (b == 1L) "lower_edge" else if (b == L) "upper_edge" else "interior",
    n_extend = n_ext, lambdas = lambdas, deviances = dev, lambda_max = lambda_max,
    penalty_factor = attr(fit, "pf"), fit = fit)
  if (!is.null(predict)) {
    out$predictions <- as.numeric(stats::predict(
      fit, unname(cbind(.hapc_basis_predictor(D$base, predict), free(predict, predict_A))),
      s = lambdas[b], type = "response"))
  }
  out
}
