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
#' @param log_lambda_min Numeric; lower bound of the log-lambda grid. Default
#'   \code{-5}.
#' @param log_lambda_max Numeric; upper bound of the log-lambda grid. Default
#'   \code{-3}.
#' @param grid_length Integer; number of grid points. Default \code{10L}.
#' @param nfolds Integer; number of CV folds. Default \code{5L}.
#'
#' @return A list with elements \code{mses} / \code{deviances} (binomial),
#'   \code{lambdas}, \code{best_lambda}, \code{res_opt} (refit on full data),
#'   and \code{predictions} when \code{predict} is non-NULL.
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
#'     projected gradient descent on logistic loss per fold; logistic deviance
#'     scoring.}
#'   \item{\code{family = "binomial"}, \code{norm = "2"}}{Logistic ridge only
#'     (no PGD) per fold; logistic deviance scoring.}
#'   \item{\code{family = "binomial"}, \code{norm = "1"}}{Logistic LASSO via
#'     \code{glmnet} (\code{family = "binomial"}, \code{alpha = 1},
#'     \code{intercept = FALSE}) on \eqn{\tilde{X} = U \cdot
#'     \mathrm{diag}(d)} per fold; logistic deviance scoring. Requires the
#'     \code{glmnet} package (in \code{Suggests}).}
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
                    family = c("gaussian", "binomial"),
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
                    ini = c("1", "2")) {

  family <- match.arg(family)
  norm   <- match.arg(norm)
  crit   <- match.arg(crit)
  ini    <- match.arg(ini)

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
    if (norm == "sv") {
      return(.Call(
        "pchal_cv_classi_call",
        X, Y_01, max_degree, npcs,
        lambdas, nfolds,
        max_iter, tol, step_factor,
        verbose, as.character(crit),
        if (is.null(predict)) NULL else predict, center,
        TRUE,  # with_pgd
        PACKAGE = "hapc"
      ))
    }
    if (norm == "2") {
      return(.Call(
        "pchal_cv_classi_call",
        X, Y_01, max_degree, npcs,
        lambdas, nfolds,
        max_iter, tol, step_factor,
        verbose, as.character(crit),
        if (is.null(predict)) NULL else predict, center,
        FALSE,  # with_pgd: logistic ridge only, deviance CV
        PACKAGE = "hapc"
      ))
    }
    if (norm == "1") {
      return(.cv_hapc_binomial_lasso(
        X, Y, max_degree = max_degree, npcs = npcs,
        lambdas = lambdas, nfolds = nfolds,
        predict = predict, center = center
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
