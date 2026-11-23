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
#' @param max_degree Integer; maximum interaction order of the HAL dictionary.
#'   The default \code{1L} (same in Python \code{cv_hapc()}) keeps main effects
#'   only, i.e. an additive model when \code{ncol(X) >= 2}. The full HAL
#'   dictionary analysed by the PC-HAR/PC-HAL theory is
#'   \code{max_degree = ncol(X)}.
#' @param log_lambda_min Numeric; lower bound of the log-lambda grid. Default
#'   \code{-5}. Used only with \code{lambda_grid = "fixed"} and
#'   \code{lambdas = NULL}.
#' @param log_lambda_max Numeric; upper bound of the log-lambda grid. Default
#'   \code{-3}. Used as \code{log_lambda_min}.
#' @param grid_length Integer; number of grid points. Default \code{10L} for
#'   \code{lambda_grid = "fixed"} and \code{40L} for
#'   \code{lambda_grid = "scaled"}.
#' @param nfolds Integer; number of CV folds. Default \code{5L}. Ignored with
#'   \code{selection = "holdout"}.
#' @param extend_grid Integer; only used when \code{family = "binomial"}. If
#'   the CV-optimal lambda is on an edge of the grid, the grid is extended on
#'   that side (same log spacing, another \code{grid_length} points) up to
#'   \code{extend_grid} times. For \code{norm = "1"} the upper extension stops
#'   at \code{lambda_max} (the smallest lambda giving the intercept-only fit);
#'   for \code{norm \%in\% c("sv", "2")} the CV is rerun on the extended grid.
#'   Default \code{0L} (no extension). A warning is issued if the optimum is
#'   still on an edge after the extensions. R-only option.
#' @param inductive Logical. If \code{TRUE}, the HAL kernel, its centring
#'   and its PCs are built from the training rows of each fold only, and the
#'   held-out rows are scored through the cross-kernel with the training knots
#'   and centring. The fold fits are then the exact penalised least-squares
#'   minimisers on the training fold. If \code{FALSE} (default, original
#'   behaviour) one basis is built from all rows of \code{X} and the fold fits
#'   use its training rows; held-out covariates then enter the basis and the
#'   closed-form fold fits are not the exact training-fold minimisers.
#'   Implemented for \code{family = "gaussian"}, \code{norm \%in\% c("1","2")}
#'   and \code{family = "binomial"}, \code{norm \%in\% c("sv","2")}; the
#'   binomial \code{norm = "1"} path is always inductive. Not yet available
#'   for gaussian \code{norm = "sv"}.
#' @param lambdas Optional numeric vector: the lambda grid to search. Takes
#'   precedence over \code{log_lambda_min}, \code{log_lambda_max} and
#'   \code{grid_length}; cannot be combined with \code{lambda_grid = "scaled"}.
#' @param lambda_grid \code{"fixed"} (default): \code{exp(seq(log_lambda_min,
#'   log_lambda_max, length.out = grid_length))}, the same for every \eqn{n}.
#'   \code{"scaled"} (gaussian, \code{norm \%in\% c("1","2")}): a grid scaled
#'   to the data, \code{grid_length} log-spaced values (default 40). For
#'   \code{norm = "2"} (PC-HAR) it is \eqn{(s_1^2/n)\,10^{[-9,1]}}; for
#'   \code{norm = "1"} (PC-HAL) it runs from \eqn{10^{-5}\lambda_{\max}} to
#'   \eqn{\lambda_{\max} = \max_j s_j|u_j^\top y_c|/n} (the smallest lambda
#'   that sets every coefficient to zero). Here \eqn{s_j}, \eqn{u_j} and
#'   \eqn{n} belong to the basis the folds are fitted on: the largest value
#'   over the training folds when \code{inductive = TRUE}, the full sample
#'   otherwise.
#' @param selection Gaussian, \code{norm \%in\% c("1","2")}. What is returned
#'   at the selected lambda: \code{"refit"} (default, original behaviour) the
#'   fit on all data; \code{"fold_average"} no refit, predictions are the
#'   average of the \code{nfolds} fold fits (each in its own basis);
#'   \code{"holdout"} a single split (\code{train_frac} of the rows for
#'   training, the rest for validation), lambda chosen on the validation part,
#'   and the training-part fit is returned. Only \code{"holdout"} with
#'   \code{inductive = TRUE} (and \code{clip = TRUE}) is the procedure covered
#'   by the hold-out selection theory for PC-HAR/PC-HAL; the full-sample refit
#'   is not covered, because the refit's norm radius is not the one that was
#'   validated.
#' @param train_frac Training fraction for \code{selection = "holdout"}.
#'   Default \code{0.8}. The split is a shuffle with \code{std::mt19937(12345)}.
#' @param clip Logical. Gaussian, \code{norm \%in\% c("1","2")}: clip every
#'   prediction (held-out scoring and returned predictions) to the range of the
#'   training outcomes of the fit that produced it. Default \code{FALSE}.
#'
#' @return With the original gaussian \code{norm \%in\% c("1","2")} CV
#'   (\code{inductive = FALSE}, \code{lambda_grid = "fixed"},
#'   \code{selection = "refit"}, \code{clip = FALSE}) and for the other
#'   families and norms: a list with elements \code{mses} / \code{deviances} (binomial),
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
#'   Otherwise (gaussian, \code{norm \%in\% c("1","2")}) the list also has
#'   \code{best_index}; \code{fold_mses} (folds x lambdas; one row for
#'   hold-out); \code{cv_predictions} (n x lambdas held-out predictions,
#'   \code{NA} for training rows of a hold-out split); \code{foldid} (fold of
#'   each row; for hold-out 0 = training, 1 = validation); \code{fold_alphas}
#'   (per fold, PCs x lambdas, in that fold's basis); \code{fold_intercepts}
#'   and \code{fold_ranks}; \code{intercept} and \code{rank} of the returned
#'   fit; \code{lambda_grid} (\code{"user"} or \code{"scaled"});
#'   \code{selection}, \code{inductive} and \code{clip}. \code{res_opt} is the
#'   coefficient vector of the full-sample refit (\code{"refit"}) or of the
#'   training-part fit (\code{"holdout"}), and \code{NULL} for
#'   \code{"fold_average"}.
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
#'     SV-ball optimisation on squared error per fold; \code{norm \%in\% c("1","2")} → closed
#'     form LASSO/ridge in PC basis, MSE per fold.}
#'   \item{\code{family = "binomial"}, \code{norm = "sv"}}{Logistic ridge init +
#'     SV-ball optimisation on logistic loss per fold, with an
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
#' For \code{norm = "sv"}, every candidate fit uses its initialiser's radius;
#' the outer CV rule and basis construction are unchanged. Fold diagnostics
#' are in attributes \code{sv_fold_status}, \code{sv_fold_gap}, and
#' \code{sv_fold_radius} (rows = folds, columns = lambdas). Refit diagnostics
#' are on \code{res_opt}; binomial outputs also expose them at the top level.
#' Numerical optimisation certificates do not validate the CV tuning rule.
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
                    extend_grid = 0L,
                    inductive = FALSE,
                    lambdas = NULL,
                    lambda_grid = c("fixed", "scaled"),
                    selection = c("refit", "fold_average", "holdout"),
                    train_frac = 0.8,
                    clip = FALSE) {

  family <- match.arg(family)
  lambda_grid <- match.arg(lambda_grid)
  selection <- match.arg(selection)
  for (nm in c("inductive", "clip")) {
    v <- get(nm)
    if (!is.logical(v) || length(v) != 1L || is.na(v))
      stop("'", nm, "' must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(lambdas) && lambda_grid == "scaled")
    stop("give either 'lambdas' or lambda_grid = \"scaled\", not both.", call. = FALSE)
  if (lambda_grid == "scaled" && missing(grid_length)) grid_length <- 40L
  new_cv <- inductive || lambda_grid == "scaled" || selection != "refit" || clip
  .hapc_check_intercept(intercept, family, as.logical(center))
  norm   <- match.arg(norm)
  crit   <- match.arg(crit)
  ini    <- match.arg(ini)

  # family = "logit-hazard": discrete-time logistic hazard. The second positional
  # argument Y carries the observed times T = min(T^event, C); the event
  # indicator Delta must be supplied. norm = "sv" is not implemented (flagged
  # inside hazard.hapc). See ?hazard.hapc.
  if (identical(family, "logit-hazard")) {
    if (new_cv || !is.null(lambdas))
      stop("inductive, lambdas, lambda_grid, selection and clip are not ",
           "available for family = 'logit-hazard'.", call. = FALSE)
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

  if (norm == "sv") .hapc_check_sv_inputs(X, Y, max_iter, tol, step_factor)

  if (!is.null(predict)) predict <- matrix(predict, ncol = p)

  if (!is.null(lambdas)) {
    lambdas <- as.numeric(lambdas)
    if (!length(lambdas) || anyNA(lambdas) || any(lambdas < 0))
      stop("'lambdas' must be non-negative numbers.", call. = FALSE)
  } else if (lambda_grid == "fixed") {
    lambdas <- exp(seq(log_lambda_min, log_lambda_max, length.out = grid_length))
  }

  euclid <- family == "gaussian" && norm %in% c("1", "2")
  if (!euclid && (lambda_grid == "scaled" || selection != "refit" || clip))
    stop("lambda_grid = \"scaled\", selection and clip are implemented for ",
         "family = 'gaussian' with norm = '1' or '2' only.", call. = FALSE)
  if (family == "gaussian" && norm == "sv" && inductive)
    stop("inductive = TRUE is not yet implemented for family = 'gaussian', ",
         "norm = 'sv'.", call. = FALSE)

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
        inductive,     # per-fold design from the training rows only
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

  if (euclid && new_cv) {
    if (approx)
      stop("approx = TRUE is only available with the original CV ",
           "(inductive = FALSE, lambda_grid = \"fixed\", selection = \"refit\", ",
           "clip = FALSE).", call. = FALSE)
    res <- .Call(
      "fasthal_cv_general_call",
      X, Y, npcs, if (lambda_grid == "scaled") NULL else lambdas,
      as.integer(grid_length), nfolds,
      if (is.null(predict)) NULL else predict, max_degree,
      center, norm == "1", inductive, selection, as.numeric(train_frac), clip,
      PACKAGE = "hapc"
    )
    res$selection <- selection
    res$inductive <- inductive
    res$clip <- clip
    return(res)
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
