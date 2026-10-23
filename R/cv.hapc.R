#' Cross-validated HAPC fit
#'
#' k-fold cross-validation over a log-spaced grid of \code{lambda} values for
#' the Highly Adaptive Principal Components (HAPC) model, supporting all three
#' norm constraints (\code{"sv"}, \code{"1"}, \code{"2"}) and both
#' \code{"gaussian"} and \code{"binomial"} families.
#'
#' Mirror of the Python function \code{hapc.cv_hapc()}: identical argument
#' names, defaults, fold partitioning (\code{std::mt19937(12345)} + shuffle),
#' and per-fold optimisation pipeline.
#'
#' @inheritParams hapc
#' @param family Character: \code{"gaussian"}, \code{"binomial"}, or
#'   \code{"logit-hazard"}.  The last is a discrete-time \emph{logistic hazard}
#'   model: the second argument \code{Y} is then the observed time
#'   \eqn{T = \min(T^{event}, C)} and \code{Delta} (event indicator) must be
#'   supplied; the call is forwarded to \code{\link{hazard.hapc}}.
#' @param Delta Event indicator \eqn{\Delta \in \{0,1\}} of length
#'   \code{nrow(X)}, required only when \code{family = "logit-hazard"}.
#'   Ignored otherwise.
#' @param time_grid Optional discrete time grid for
#'   \code{family = "logit-hazard"} (see \code{\link{hazard.hapc}}). Ignored
#'   otherwise.
#' @param log_lambda_min Numeric; lower bound of the log-lambda grid. Default
#'   \code{-5}.
#' @param log_lambda_max Numeric; upper bound of the log-lambda grid. Default
#'   \code{-3}.
#' @param grid_length Integer; number of grid points. Default \code{10L}.
#' @param nfolds Integer; number of CV folds. Default \code{5L}.
#' @param extend_grid Integer; only used when \code{family = "binomial"}. If
#'   the CV-optimal lambda is on an edge of the grid, the grid is extended on
#'   that side (same log spacing, another \code{grid_length} points) up to
#'   \code{extend_grid} times. For \code{norm = "1"} the upper extension stops
#'   at \code{lambda_max} (the smallest lambda giving the intercept-only fit);
#'   for \code{norm \%in\% c("sv", "2")} the CV is rerun on the extended grid.
#'   Default \code{0L} (no extension). A warning is issued if the optimum is
#'   still on an edge after the extensions. R-only option.
#'
#' @return A list with elements \code{mses} / \code{deviances} (binomial),
#'   \code{lambdas}, \code{best_lambda}, \code{res_opt} (refit on full data),
#'   and \code{predictions} when \code{predict} is non-NULL. For
#'   \code{family = "binomial"}, \code{norm = "1"} also \code{best_index},
#'   \code{lambda_position} (\code{"interior"}, \code{"lower_edge"},
#'   \code{"upper_edge"} of the final grid, or \code{"null_model"} when the
#'   optimum is \code{lambda_max}), \code{n_extend} (number of grid
#'   extensions), \code{lambda_max} and \code{res_opt$intercept}. For
#'   \code{family = "binomial"}, \code{norm \%in\% c("sv", "2")} and
#'   \code{extend_grid > 0}, also \code{best_index}, \code{lambda_position}
#'   and \code{n_extend}.
#'
#' @details
#' Fold construction is implemented in C++: block partition (remainder in the
#' last fold), then a shuffle with fixed seed \code{12345}
#' (\code{std::mt19937}), identical to Python \code{cv_hapc()}.
#'
#' Loss-by-family contract (binary responses are \emph{always} fit and scored
#' with logistic loss):
#' \describe{
#'   \item{\code{family = "gaussian"}}{\code{norm = "sv"} → ridge/LASSO init +
#'     PGD on squared error per fold; \code{norm \%in\% c("1","2")} → closed
#'     form LASSO/ridge in PC basis, MSE per fold.}
#'   \item{\code{family = "binomial"}, \code{norm = "sv"}}{Logistic ridge init +
#'     projected gradient descent on logistic loss per fold, with an
#'     unpenalised intercept profiled out at every step; logistic deviance
#'     scoring.}
#'   \item{\code{family = "binomial"}, \code{norm = "2"}}{Logistic ridge only
#'     (no PGD) per fold, with an unpenalised intercept; logistic deviance
#'     scoring.}
#'   \item{\code{family = "binomial"}, \code{norm = "1"}}{Logistic LASSO via
#'     \code{glmnet} (\code{family = "binomial"}, \code{alpha = 1},
#'     unpenalised intercept) on
#'     \eqn{\tilde{X} = U \cdot \mathrm{diag}(d)} per fold; logistic
#'     deviance scoring. U and d are computed once per fold from the HAL
#'     kernel and \code{glmnet} fits the whole lambda grid on them. Requires
#'     the \code{glmnet} package (in \code{Suggests}).}
#' }
#'
#' @examples
#' \dontrun{
#' n <- 200; d <- 3
#' X <- matrix(runif(n * d, 0, 1), ncol = d)
#' Y <- sin(pi * X[, 1]) + X[, 2] + rnorm(n, sd = 0.1)
#' cv <- cv.hapc(X, Y, max_degree = 2, npcs = n - 1, nfolds = 5,
#'               grid_length = 10, norm = "sv")
#' }
#'
#' @export
cv.hapc <- function(X, Y,
                    family = c("gaussian", "binomial", "logit-hazard"),
                    max_degree = 1L,
                    npcs = nrow(X),
                    log_lambda_min = -5,
                    log_lambda_max = -3,
                    grid_length = 10L,
                    nfolds = 5L,
                    norm = c("sv", "1", "2"),
                    predict = NULL,
                    max_iter = 5000L,
                    tol = 1e-3,
                    step_factor = 0.8,
                    verbose = FALSE,
                    crit = c("grad", "risk"),
                    center = TRUE,
                    approx = FALSE,
                    ini = c("1", "2"),
                    Delta = NULL,
                    time_grid = NULL,
                    intercept = NULL,
                    extend_grid = 0L) {

  family <- match.arg(family)
  .hapc_check_intercept(intercept, family, as.logical(center))
  norm   <- match.arg(norm)
  crit   <- match.arg(crit)
  ini    <- match.arg(ini)

  # family = "logit-hazard": discrete-time logistic hazard. The second positional
  # argument Y carries the observed times T = min(T^event, C); the event
  # indicator Delta must be supplied. norm = "sv" is not implemented (flagged
  # inside hazard.hapc). See ?hazard.hapc.
  if (identical(family, "logit-hazard")) {
    if (is.null(Delta)) {
      stop("family='logit-hazard' requires 'Delta' (event indicator); pass the ",
           "observed times T as the second argument and Delta separately. ",
           "See ?hazard.hapc.", call. = FALSE)
    }
    return(hazard.hapc(
      X = X, T = Y, Delta = Delta, norm = norm,
      max_degree = max_degree, npcs = if (missing(npcs)) NULL else npcs,
      time_grid = time_grid,
      log_lambda_min = log_lambda_min, log_lambda_max = log_lambda_max,
      grid_length = grid_length, nfolds = nfolds,
      predict = predict, center = center, verbose = verbose,
      max_iter = max_iter, tol = tol, step_factor = step_factor
    ))
  }

  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  Y <- as.numeric(Y)
  p <- ncol(X)

  max_degree     <- as.integer(max_degree)
  npcs           <- as.integer(npcs)
  log_lambda_min <- as.numeric(log_lambda_min)
  log_lambda_max <- as.numeric(log_lambda_max)
  grid_length    <- as.integer(grid_length)
  nfolds         <- as.integer(nfolds)
  max_iter       <- as.integer(max_iter)
  tol            <- as.numeric(tol)
  step_factor    <- as.numeric(step_factor)
  verbose        <- as.logical(verbose)
  center         <- as.logical(center)
  approx         <- as.logical(approx)

  if (!is.null(predict)) predict <- matrix(predict, ncol = p)

  lambdas <- exp(seq(log_lambda_min, log_lambda_max, length.out = grid_length))

  if (family == "binomial") {
    # Validate labels; allow soft labels in [0,1] only for norm in {"1","2"}.
    kind <- .hapc_check_binomial_labels(Y, norm)
    # C++ sv/2 paths expect Y in [0,1]; map {-1,+1} -> {0,1} if needed.
    Y_01 <- if (kind == "pm1") (as.numeric(Y) + 1) / 2 else as.numeric(Y)
    if (norm %in% c("sv", "2")) {
      run <- function(lams) .Call(
        "pchal_cv_classi_call",
        X, Y_01, max_degree, npcs,
        lams, nfolds,
        max_iter, tol, step_factor,
        verbose, as.character(crit),
        if (is.null(predict)) NULL else predict, center,
        norm == "sv",  # with_pgd: PGD for "sv", logistic ridge only for "2"
        PACKAGE = "hapc"
      )
      res <- run(lambdas)
      if (extend_grid <= 0L) return(res)
      # Grid extension on the edge where the CV optimum lies (same log
      # spacing, grid_length more points); the whole CV is rerun on the
      # extended grid. Ridge and PGD have no finite lambda_max, so the
      # upper side is extended too.
      step <- if (length(lambdas) > 1L) log(lambdas[2L]) - log(lambdas[1L]) else log(10)
      n_extend <- 0L
      while (n_extend < extend_grid) {
        b <- which.min(res$deviances)
        L <- length(res$lambdas)
        if (L > 1L && b == 1L) {
          lambdas <- c(exp(log(lambdas[1L]) - step * rev(seq_len(grid_length))), lambdas)
        } else if (L > 1L && b == L) {
          lambdas <- c(lambdas, exp(log(lambdas[L]) + step * seq_len(grid_length)))
        } else {
          break
        }
        res <- run(lambdas)
        n_extend <- n_extend + 1L
      }
      b <- which.min(res$deviances)
      L <- length(res$lambdas)
      res$best_index <- b
      res$lambda_position <- if (L == 1L) NA_character_ else
        if (b == 1L) "lower_edge" else if (b == L) "upper_edge" else "interior"
      res$n_extend <- n_extend
      if (res$lambda_position %in% c("lower_edge", "upper_edge")) {
        warning("cv.hapc(family='binomial', norm='", norm, "'): the CV-optimal ",
                "lambda is still on the ", sub("_edge", "", res$lambda_position),
                " edge of the grid after ", n_extend, " extension(s).",
                call. = FALSE)
      }
      return(res)
    }
    if (norm == "1") {
      return(.cv_hapc_binomial_lasso(
        X, Y, max_degree = max_degree, npcs = npcs,
        lambdas = lambdas, nfolds = nfolds,
        predict = predict, center = center,
        extend_grid = as.integer(extend_grid)
      ))
    }
    stop("family='binomial' supports norm in c('sv','1','2'); got '", norm, "'.")
  }

  if (norm == "sv") {
    .Call(
      "pchal_cv_call",
      X, Y, max_degree, npcs,
      lambdas, nfolds,
      max_iter, tol, step_factor,
      verbose, as.character(crit),
      if (is.null(predict)) NULL else predict, center,
      as.character(ini),
      PACKAGE = "hapc"
    )
  } else if (norm == "1") {
    .Call(
      "fasthal_cv_call",
      X, Y, npcs, lambdas, nfolds,
      if (is.null(predict)) NULL else predict, max_degree,
      center, approx, as.logical(1),
      PACKAGE = "hapc"
    )
  } else {  # norm == "2"
    .Call(
      "fasthal_cv_call",
      X, Y, npcs, lambdas, nfolds,
      if (is.null(predict)) NULL else predict, max_degree,
      center, approx, as.logical(0),
      PACKAGE = "hapc"
    )
  }
}
