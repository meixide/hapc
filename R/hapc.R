#' Single-lambda HAPC fit
#'
#' Fits the Highly Adaptive Principal Components (HAPC) model at a single
#' regularisation parameter \code{lambda}, using one of three norm
#' constraints: sectional variation (\code{"sv"}), L1 (\code{"1"}) or L2
#' (\code{"2"}).  Supports \code{family = "gaussian"} and
#' \code{family = "binomial"}.
#'
#' This is the R counterpart of the Python function \code{hapc.hapc()}.  All
#' argument names, defaults, and behaviour are identical between R and Python
#' (the only language-level difference is that Python uses \code{lambda_}
#' since \code{lambda} is a reserved keyword there).
#'
#' @param X Numeric matrix of features (rows = observations, cols = features).
#' @param Y Numeric response vector of length \code{nrow(X)}. For
#'   \code{family = "binomial"}: hard labels in \code{{0,1}} or \code{{-1,+1}},
#'   or \emph{soft} labels in \code{[0,1]} (e.g. EM-HAL E-step posteriors).
#'   Soft labels are supported only for \code{norm = "1"} or \code{"2"}
#'   (cross-entropy target); \code{norm = "sv"} requires hard labels.
#' @param family Character: \code{"gaussian"} (squared error, default) or
#'   \code{"binomial"} (logistic loss).
#' @param max_degree Integer; maximum interaction order for the HAL basis.
#'   Default \code{1L} (additive HAL).
#' @param npcs Integer; number of principal components to keep. Default
#'   \code{nrow(X)}.
#' @param lambda Numeric scalar; regularisation parameter. Default
#'   \code{0.01}.
#' @param norm Character: \code{"sv"}, \code{"1"} or \code{"2"}. Default
#'   \code{"sv"}. For \code{family = "binomial"} all three are supported and
#'   the loss is always logistic: \code{"sv"} = PGD on logistic loss with
#'   sectional-variation projection, \code{"2"} = logistic ridge (Newton-
#'   Raphson IRLS, no PGD), \code{"1"} = logistic LASSO via
#'   \code{glmnet::glmnet(..., family = "binomial", alpha = 1,
#'   intercept = FALSE)} on \eqn{\tilde{X} = U \cdot \mathrm{diag}(d)}
#'   (requires the \code{glmnet} package).
#' @param predict Optional numeric matrix of new observations (same number of
#'   columns as \code{X}). If supplied, predictions are returned in the result.
#' @param max_iter Integer; maximum projected-gradient iterations (only used
#'   when \code{norm = "sv"}). Default \code{5000L}.
#' @param tol Numeric; convergence tolerance (only used when
#'   \code{norm = "sv"}). Default \code{1e-3}.
#' @param step_factor Numeric; PGD line-search factor (only used when
#'   \code{norm = "sv"}). Default \code{0.8}.
#' @param verbose Logical; print progress (only used when \code{norm = "sv"}).
#'   Default \code{FALSE}.
#' @param crit Character; PGD stopping criterion (only used when
#'   \code{norm = "sv"}): \code{"grad"} (gradient infinity-norm, default) or
#'   \code{"risk"} (relative risk decrease).
#' @param center Logical; whether to centre the kernel matrix and response.
#'   Default \code{TRUE}.
#' @param approx Logical; if \code{TRUE} use approximate eigendecomposition
#'   (power iteration) for \code{norm \%in\% c("1","2")}. Ignored for
#'   \code{norm = "sv"}. Default \code{FALSE}.
#' @param ini Character; initialiser for the projected-gradient solver when
#'   \code{norm = "sv"}: \code{"1"} = LASSO via \code{fast_pchal_call}
#'   (default, matches Python), \code{"2"} = ridge via \code{ridge_call}.
#'
#' @return A named list whose structure depends on \code{family} and
#'   \code{norm}:
#' \describe{
#'   \item{gaussian, \code{norm = "sv"}}{Optimiser output (e.g.\ \code{res_opt}
#'     with \code{alpha}, risk, iterations) plus optional \code{predictions}.}
#'   \item{gaussian, \code{norm \%in\% c("1","2")}}{\code{alpha} on the PC
#'     basis, optional \code{predictions}, \code{lambda}.}
#'   \item{binomial, \code{norm = "sv"}}{Same layout as the one-lambda binomial
#'     CV C++ wrapper (\code{pchal_cv_classi_call} with \code{nfolds = 1}):
#'     deviances grid, \code{best_lambda}, \code{best_alpha}, optional
#'     probabilities in \code{predictions}.}
#'   \item{binomial, \code{norm = "2"}}{Logistic ridge only: \code{alpha},
#'     \code{risk}, \code{iter = 0}, optional \code{predictions} / class
#'     outputs.}
#'   \item{binomial, \code{norm = "1"}}{Logistic LASSO (\code{glmnet}):
#'     \code{alpha}, \code{lambda}, \code{risk}, \code{iter = 0L}, and
#'     optional \code{predictions} (log-odds), \code{probabilities},
#'     \code{predicted_classes}.}
#' }
#'
#' @examples
#' \dontrun{
#' n <- 100; d <- 3
#' X <- matrix(runif(n * d, 0, 1), ncol = d)
#' Y <- sin(pi * X[, 1]) + X[, 2] + rnorm(n, sd = 0.1)
#' fit <- hapc(X, Y, max_degree = 2, npcs = n - 1, lambda = 0.05, norm = "sv")
#' }
#'
#' @export
hapc <- function(X, Y,
                 family = c("gaussian", "binomial"),
                 max_degree = 1L,
                 npcs = nrow(X),
                 lambda = 0.01,
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

  max_degree  <- as.integer(max_degree)
  npcs        <- as.integer(npcs)
  lambda      <- as.numeric(lambda)
  max_iter    <- as.integer(max_iter)
  tol         <- as.numeric(tol)
  step_factor <- as.numeric(step_factor)
  verbose     <- as.logical(verbose)
  center      <- as.logical(center)
  approx      <- as.logical(approx)

  if (!is.null(predict)) predict <- matrix(predict, ncol = p)

  if (family == "binomial") {
    # Validate labels; allow soft labels in [0,1] only for norm in {"1","2"}.
    kind <- .hapc_check_binomial_labels(Y, norm)
    # C++ sv/2 paths expect Y in [0,1]; map {-1,+1} -> {0,1} if needed.
    Y_01 <- if (kind == "pm1") (as.numeric(Y) + 1) / 2 else as.numeric(Y)
    if (norm == "sv") {
      res <- .Call(
        "pchal_cv_classi_call",
        X, Y_01,
        max_degree, npcs,
        as.numeric(lambda), 1L,
        max_iter, tol,
        step_factor, verbose, as.character(crit),
        if (is.null(predict)) NULL else predict, center,
        TRUE,  # with_pgd: logistic ridge + PGD on logistic loss
        PACKAGE = "hapc"
      )
      return(res)
    }
    if (norm == "2") {
      res <- .Call(
        "single_pcghal_classi_ridge_call",
        X, Y_01,
        max_degree, npcs,
        lambda,
        if (is.null(predict)) NULL else predict, center,
        PACKAGE = "hapc"
      )
      return(res)
    }
    if (norm == "1") {
      return(.hapc_binomial_lasso(X, Y, max_degree, npcs, lambda,
                                   predict = predict, center = center))
    }
    stop("family='binomial' supports norm in c('sv','1','2'); got '", norm, "'.")
  }

  if (norm == "sv") {
    res <- .Call(
      "single_pcghal_call",
      X, Y,
      max_degree, npcs,
      lambda,
      max_iter, tol,
      step_factor, verbose, as.character(crit),
      if (is.null(predict)) NULL else predict, center,
      as.character(ini),
      PACKAGE = "hapc"
    )
  } else if (norm == "2") {
    res <- .Call(
      "single_lambda_pchar",
      X, Y, npcs,
      lambda,
      if (is.null(predict)) NULL else predict, max_degree,
      center, approx, as.logical(0),
      PACKAGE = "hapc"
    )
  } else {  # norm == "1"
    res <- .Call(
      "single_lambda_pchar",
      X, Y, npcs,
      lambda,
      if (is.null(predict)) NULL else predict, max_degree,
      center, approx, as.logical(1),
      PACKAGE = "hapc"
    )
  }

  res
}
