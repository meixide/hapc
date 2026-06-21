#' Discrete-time logistic hazard via HAPC (\code{family = "logit-hazard"})
#'
#' \code{hazard.hapc} fits a discrete-time \emph{logistic hazard} model with
#' HAPC.  It is a convenience wrapper around \code{\link{cv.hapc}} with
#' \code{family = "binomial"}: the right-censored survival data
#' \eqn{(X_i, T_i, \Delta_i)} are expanded into a person-period (one row per
#' subject-per-interval-at-risk) table whose binary response is the discrete
#' hazard indicator, the visit time is added as the first HAL covariate, and the
#' regularisation parameter \eqn{\lambda} is chosen by cross-validated logistic
#' deviance.
#'
#' This is the \code{family = "logit-hazard"} mode referenced by
#' \code{\link{cv.hapc}} / \code{\link{hapc}}: \emph{hard} 0/1 hazard labels are
#' fit with logistic loss.  It is the R counterpart of the Python function
#' \code{hapc.hazard_hapc()}; argument names, defaults and behaviour match.
#'
#' @section Data contract:
#' For subject \eqn{i} let \eqn{T_i^{event}} be the (latent) event time and
#' \eqn{C_i} the censoring time.  The user supplies only the \emph{observed}
#' quantities
#' \deqn{T_i = \min(T_i^{event}, C_i), \qquad
#'       \Delta_i = \mathbf{1}(T_i^{event} \le C_i),}
#' i.e. \code{T} (observed time) and \code{Delta} (event indicator), together
#' with baseline covariates \code{X}.  Times are assumed \emph{discrete}
#' (interval / visit indices).  Each subject contributes one person-period row
#' for every grid time \eqn{g \le T_i}; the hazard label is \eqn{Y = 1} only at
#' the event interval (\eqn{g = T_i} and \eqn{\Delta_i = 1}) and \eqn{0}
#' otherwise (including the last interval of censored subjects).
#'
#' @details
#' \strong{Model.} Let the discrete hazard be the conditional event probability
#' in interval \eqn{t} given survival up to \eqn{t},
#' \deqn{\lambda(t \mid x) = P(T^{event} = t \mid T^{event} \ge t, X = x),}
#' modelled on the logit scale by a Highly Adaptive Principal Components fit
#' \eqn{f} of the augmented covariate \eqn{(t, x)},
#' \deqn{\mathrm{logit}\,\lambda(t \mid x) = f(t, x).}
#' The HAL basis spans indicator (and, for \code{max_degree > 1}, interaction)
#' tensor products in \eqn{(t, x)}, so the time effect, the covariate effects
#' and their interactions are all estimated nonparametrically; the
#' \eqn{\ell_1}/\eqn{\ell_2} penalty (\code{norm}) controls smoothness and is
#' tuned by cross-validation.
#'
#' \strong{Person-period likelihood.} Under independent right-censoring the
#' observed-data likelihood factorises over the at-risk intervals,
#' \deqn{\prod_{i=1}^{n} \prod_{t \le T_i}
#'       \lambda(t \mid x_i)^{Y_{it}}\,(1 - \lambda(t \mid x_i))^{1 - Y_{it}},
#'       \qquad Y_{it} = \mathbf{1}(T^{event}_i = t),}
#' which is exactly the Bernoulli (logistic) likelihood of the expanded
#' person-period table.  Fitting a binomial HAPC model to \eqn{Y_{it}} against
#' \eqn{(t, x_i)} therefore estimates the discrete hazard
#' (Cox 1972; Brown 1975; Allison 1982).
#'
#' \strong{Survival.} The conditional survival function follows from the
#' estimated hazard by the product-limit relation
#' \deqn{S(t \mid x) = P(T^{event} > t \mid x) =
#'       \prod_{s \le t} \bigl(1 - \lambda(s \mid x)\bigr),}
#' returned for new subjects when \code{predict} is supplied.
#'
#' @param X Numeric matrix of baseline covariates, one row per subject
#'   (\code{nrow(X)} subjects).
#' @param T Numeric vector of observed times
#'   \eqn{T_i = \min(T_i^{event}, C_i)}, length \code{nrow(X)}.  Assumed
#'   discrete (interval indices).
#' @param Delta Numeric/integer vector of event indicators
#'   \eqn{\Delta_i \in \{0,1\}}, length \code{nrow(X)} (\code{1} = event,
#'   \code{0} = right-censored).
#' @param norm Character: \code{"1"} (logistic LASSO, default, requires
#'   \code{glmnet}) or \code{"2"} (logistic ridge).  \code{norm = "sv"} is
#'   \strong{not implemented} for this family and raises an error.
#' @param max_degree Integer; HAL interaction order over \code{[time, X]}.
#'   Default \code{1L}.
#' @param npcs Integer; number of principal components.  Default \code{NULL} =
#'   the number of person-period rows (capped internally as in
#'   \code{\link{cv.hapc}}).
#' @param time_grid Optional numeric vector of the discrete time points (the
#'   risk-set grid).  Default \code{NULL} infers it from \code{T}: the integer
#'   sequence \code{min(T):max(T)} when \code{T} is integer-valued, otherwise
#'   \code{sort(unique(T))}.  Subjects are assumed at risk from
#'   \code{min(time_grid)} onward.
#' @param log_lambda_min,log_lambda_max,grid_length Log-\eqn{\lambda} grid for
#'   cross-validation (defaults \code{-4}, \code{-1}, \code{15L}).
#' @param nfolds Integer; number of CV folds.  Default \code{5L}.
#' @param predict Optional numeric matrix of \emph{baseline covariates} for new
#'   subjects (same number of columns as \code{X}).  When supplied the fitted
#'   model (refit at the CV-selected \eqn{\lambda}) is evaluated on the full
#'   time grid for each new subject, returning a hazard surface and the implied
#'   survival curves \eqn{S(t\mid x) = \prod_{g \le t}(1 - \lambda(g\mid x))}.
#' @param center,verbose,max_iter,tol,step_factor Passed through to
#'   \code{\link{cv.hapc}} / \code{\link{hapc}}.
#'
#' @return An object of class \code{"hapc_hazard"} (a list) with:
#'   \item{hazard}{Estimated hazard for each person-period row (aligned with
#'     \code{data}); these are the cross-validated predictions at the winning
#'     \eqn{\lambda}.}
#'   \item{data}{The person-period \code{data.frame}: \code{id}, \code{time},
#'     the baseline covariate columns, the hazard label \code{Y}, and the
#'     estimated \code{hazard}.}
#'   \item{times}{The discrete time grid used.}
#'   \item{lambdas, risk, best_lambda}{CV grid, mean logistic deviance per
#'     \eqn{\lambda}, and the deviance-minimising \eqn{\lambda}.}
#'   \item{interior}{Logical; \code{TRUE} when \code{best_lambda} is strictly
#'     inside the grid (not at either endpoint) -- a basic sanity check that the
#'     grid brackets the optimum.}
#'   \item{cv}{The full underlying \code{\link{cv.hapc}} result.}
#'   \item{predict_hazard, predict_survival, predict_times}{Present only when
#'     \code{predict} is supplied: matrices (new subjects \eqn{\times} grid
#'     times) of estimated hazards and survival probabilities.}
#'
#' @references
#' Cox, D. R. (1972). Regression models and life-tables.
#' \emph{Journal of the Royal Statistical Society B}, 34(2), 187--220.
#'
#' Brown, C. C. (1975). On the use of indicator variables for studying the
#' time-dependence of parameters in a response-time model.
#' \emph{Biometrics}, 31(4), 863--872.
#'
#' Allison, P. D. (1982). Discrete-time methods for the analysis of event
#' histories. \emph{Sociological Methodology}, 13, 61--98.
#'
#' Singer, J. D. and Willett, J. B. (2003).
#' \emph{Applied Longitudinal Data Analysis}. Oxford University Press.
#'
#' Benkeser, D. and van der Laan, M. (2016). The Highly Adaptive Lasso
#' estimator. \emph{IEEE International Conference on Data Science and Advanced
#' Analytics (DSAA)}, 689--696.
#'
#' @seealso \code{\link{cv.hapc}} for the underlying cross-validated binomial
#'   fit and the \code{family = "logit-hazard"} dispatch; \code{\link{hapc}} for
#'   a single-\eqn{\lambda} fit.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 300
#' X <- cbind(age = runif(n), trt = rbinom(n, 1, 0.5))
#' grid <- 1:6
#'
#' ## true discrete hazard used to simulate event times
#' haz <- function(t, x) plogis(-2.5 + 0.25 * t + 1.2 * x[1] - 0.8 * x[2])
#' Tev <- integer(n); Cen <- sample(grid, n, replace = TRUE)
#' for (i in seq_len(n)) {
#'   Tev[i] <- max(grid)
#'   for (t in grid) if (rbinom(1, 1, haz(t, X[i, ])) == 1) { Tev[i] <- t; break }
#' }
#' Tobs  <- pmin(Tev, Cen)             # observed time  T = min(T^event, C)
#' Delta <- as.integer(Tev <= Cen)     # event indicator
#'
#' ## fit the discrete-time logistic hazard
#' fit <- hazard.hapc(X, Tobs, Delta, norm = "1", max_degree = 2,
#'                    time_grid = grid)
#' fit                                  # print method: rows, events, lambda
#' fit$best_lambda; fit$interior        # CV optimum and interior-grid check
#'
#' ## identical call through the family dispatcher
#' fit2 <- cv.hapc(X, Tobs, family = "logit-hazard", Delta = Delta,
#'                 norm = "1", max_degree = 2, time_grid = grid)
#'
#' ## hazard surface + survival curves S(t | x) for two new subjects
#' newX <- rbind(c(age = 0.3, trt = 0), c(age = 0.8, trt = 1))
#' fit3 <- hazard.hapc(X, Tobs, Delta, norm = "1", time_grid = grid,
#'                     predict = newX)
#' fit3$predict_hazard      # 2 x length(grid) hazards
#' fit3$predict_survival    # 2 x length(grid) survival probabilities
#' }
#'
#' @export
hazard.hapc <- function(X, T, Delta,
                        norm = "1",
                        max_degree = 1L,
                        npcs = NULL,
                        time_grid = NULL,
                        log_lambda_min = -4,
                        log_lambda_max = -1,
                        grid_length = 15L,
                        nfolds = 5L,
                        predict = NULL,
                        center = TRUE,
                        verbose = FALSE,
                        max_iter = 5000L,
                        tol = 1e-3,
                        step_factor = 0.8) {

  norm <- as.character(norm)[1L]
  if (identical(norm, "sv")) {
    stop("family='logit-hazard' (discrete-time logistic hazard) is not ",
         "implemented for norm='sv'; use norm='1' or norm='2'.", call. = FALSE)
  }
  if (!norm %in% c("1", "2")) {
    stop("family='logit-hazard' supports norm in c('1','2'); got '", norm, "'.",
         call. = FALSE)
  }

  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  Tobs  <- as.numeric(T)
  Delta <- as.numeric(Delta)
  n <- nrow(X)
  if (length(Tobs) != n)  stop("length(T) must equal nrow(X).", call. = FALSE)
  if (length(Delta) != n) stop("length(Delta) must equal nrow(X).", call. = FALSE)
  if (!all(Delta %in% c(0, 1)))
    stop("Delta must be a 0/1 event indicator.", call. = FALSE)

  if (is.null(time_grid)) time_grid <- .hapc_infer_time_grid(Tobs)
  time_grid <- sort(unique(as.numeric(time_grid)))
  if (length(time_grid) < 2L)
    stop("Need at least two distinct time points; got ", length(time_grid),
         ".", call. = FALSE)
  if (max(Tobs) > max(time_grid))
    stop("Some observed times exceed max(time_grid); supply a wider ",
         "'time_grid'.", call. = FALSE)

  cn <- colnames(X)
  if (is.null(cn)) cn <- paste0("X", seq_len(ncol(X)))

  pp <- .hapc_personperiod(X, Tobs, Delta, time_grid)
  Xpp <- pp$Xpp
  colnames(Xpp) <- c("time", cn)

  # Default npcs = numerical rank of the person-period HAL design. The
  # expansion creates many duplicate rows (discrete time x repeated baseline
  # covariates), so the kernel is rank-deficient; keeping the null-space
  # directions makes the 1/d prediction reconstruction blow up (notably for
  # norm = "2"). Dropping the near-zero singular values is both numerically
  # safe and a sensible amount of regularisation.
  if (is.null(npcs)) {
    des0 <- design.hapc(Xpp, max_degree = max_degree, npcs = nrow(Xpp),
                        center = center)
    npcs <- max(1L, sum(des0$d > 1e-7 * des0$d[1]))
  }

  cv <- cv.hapc(
    X = Xpp, Y = pp$Y, family = "binomial", norm = norm,
    max_degree = max_degree, npcs = npcs,
    log_lambda_min = log_lambda_min, log_lambda_max = log_lambda_max,
    grid_length = grid_length, nfolds = nfolds,
    predict = Xpp, center = center, verbose = verbose,
    max_iter = max_iter, tol = tol, step_factor = step_factor
  )

  hazard_train <- as.numeric(cv$predictions)
  lambdas <- as.numeric(cv$lambdas)
  risk <- if (!is.null(cv$deviances)) as.numeric(cv$deviances)
          else as.numeric(cv$mses)
  best_lambda <- as.numeric(cv$best_lambda)
  best_idx <- which.min(risk)
  interior <- best_idx > 1L && best_idx < length(lambdas)

  data <- data.frame(id = pp$id, time = pp$time)
  data[cn] <- Xpp[, -1, drop = FALSE]
  data$Y <- pp$Y
  data$hazard <- hazard_train

  out <- list(
    hazard      = hazard_train,
    data        = data,
    times       = time_grid,
    lambdas     = lambdas,
    risk        = risk,
    best_lambda = best_lambda,
    interior    = interior,
    cv          = cv
  )

  if (!is.null(predict)) {
    Xnew <- as.matrix(predict)
    storage.mode(Xnew) <- "double"
    if (ncol(Xnew) != ncol(X))
      stop("predict must have the same number of columns as X.", call. = FALSE)
    m <- nrow(Xnew); K <- length(time_grid)
    # Full-grid expansion, row order = (subject 1: all times), (subject 2: ...)
    newXpp <- cbind(
      time = rep(time_grid, times = m),
      Xnew[rep(seq_len(m), each = K), , drop = FALSE]
    )
    colnames(newXpp) <- c("time", cn)
    fit <- hapc(
      X = Xpp, Y = pp$Y, family = "binomial", norm = norm,
      max_degree = max_degree, npcs = npcs, lambda = best_lambda,
      predict = newXpp, center = center, verbose = verbose,
      max_iter = max_iter, tol = tol, step_factor = step_factor
    )
    haz_vec <- as.numeric(fit$probabilities)
    haz_mat <- matrix(haz_vec, nrow = m, ncol = K, byrow = TRUE)
    surv_mat <- t(apply(1 - haz_mat, 1L, cumprod))
    out$predict_hazard   <- haz_mat
    out$predict_survival <- surv_mat
    out$predict_times    <- time_grid
  }

  class(out) <- "hapc_hazard"
  out
}

# --- internal: default discrete time grid from observed times ---------------
.hapc_infer_time_grid <- function(Tobs) {
  Tobs <- as.numeric(Tobs)
  if (all(abs(Tobs - round(Tobs)) < 1e-9)) {
    return(seq(min(round(Tobs)), max(round(Tobs))))
  }
  sort(unique(Tobs))
}

# --- internal: person-period expansion --------------------------------------
# Returns id / time vectors, the [time, X] design matrix `Xpp`, and the binary
# hazard response `Y`. Subject i contributes one row per grid time g <= T_i;
# Y = 1 only at the event interval (g == T_i & Delta_i == 1).
.hapc_personperiod <- function(X, Tobs, Delta, time_grid) {
  n <- nrow(X); p <- ncol(X)
  ids   <- vector("list", n)
  times <- vector("list", n)
  Xrows <- vector("list", n)
  Yvals <- vector("list", n)
  for (i in seq_len(n)) {
    g_i <- time_grid[time_grid <= Tobs[i]]
    if (length(g_i) == 0L) g_i <- time_grid[1L]
    k <- length(g_i)
    ids[[i]]   <- rep.int(i, k)
    times[[i]] <- g_i
    Xrows[[i]] <- matrix(rep(X[i, ], each = k), nrow = k, ncol = p)
    Yvals[[i]] <- as.integer(g_i == Tobs[i] & Delta[i] == 1)
  }
  time_col <- unlist(times, use.names = FALSE)
  Xpp <- cbind(time_col, do.call(rbind, Xrows))
  list(
    id   = unlist(ids, use.names = FALSE),
    time = time_col,
    Xpp  = Xpp,
    Y    = unlist(Yvals, use.names = FALSE)
  )
}

#' Print a discrete-time logistic hazard fit
#'
#' Compact summary of a \code{"hapc_hazard"} object: the number of person-period
#' rows, the number of events, the discrete time grid, and the
#' cross-validation-selected \eqn{\lambda} together with whether it is an
#' interior grid point.
#'
#' @param x An object of class \code{"hapc_hazard"} returned by
#'   \code{\link{hazard.hapc}}.
#' @param ... Ignored; present for S3 method consistency.
#'
#' @return \code{x}, invisibly.
#'
#' @seealso \code{\link{hazard.hapc}}
#' @export
print.hapc_hazard <- function(x, ...) {
  cat("<hapc discrete-time logistic hazard fit>\n")
  cat("  person-period rows:", nrow(x$data),
      " | events:", sum(x$data$Y), "\n")
  cat("  time grid:", paste(x$times, collapse = ", "), "\n")
  cat(sprintf("  best lambda: %.4g  (interior grid point: %s)\n",
              x$best_lambda, x$interior))
  invisible(x)
}
