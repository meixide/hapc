#' Point-treatment ATE with HAPC nuisance fits and TMLE targeting
#'
#' Estimate E[Y(1)-Y(0)] with treatment-stratified cross-fitting of HAPC
#' initial outcome and propensity fits, followed by a pooled two-coordinate
#' fluctuation of the out-of-fold predictions. No SuperLearner dependency.
#'
#' @param X Finite numeric baseline-covariate matrix, one row per independent unit.
#' @param Y Finite numeric outcome vector. Missing outcomes are not yet supported.
#' @param A Binary numeric or logical treatment vector coded 0/1; both arms required.
#' @param family Outcome family: auto detects binary 0/1 outcomes, otherwise gaussian.
#' @param outcome_bounds Optional fixed, scientifically known lower/upper outcome
#'   bounds. Gaussian outcomes use an identity fluctuation unless bounds are
#'   supplied, in which case targeting is logistic on the rescaled [0,1] outcome.
#'   Binomial outcomes always use bounds c(0,1). Do not use sample extrema as bounds.
#' @param alpha Significance level for the untruncated Wald interval.
#' @param cf_folds Outer folds (default 5); 1 selects ordinary in-sample TMLE.
#' @param seed Seed for outer folds, without changing the caller's RNG state.
#' @param g Optional known propensity (scalar or length n) or externally fitted
#'   out-of-fold propensity predictions. Supplying g skips propensity fitting.
#' @param Q Optional n by 2 matrix of initial response-scale predictions, columns
#'   Q0 and Q1 in that order. Supply out-of-fold predictions for cross-fitted use.
#' @param g_bound Propensities are clipped to [g_bound,1-g_bound]; default 0.025.
#' @param q_bound Numerical probability bound before logistic targeting.
#' @param max_degree,npcs,nfolds,norm HAPC complexity and inner CV controls.
#'   norm defaults explicitly to "1". npcs=NULL uses each training sample's size.
#' @param log_lambda_min,log_lambda_max,grid_length Inner-CV log-lambda grid.
#' @param target_tol Absolute tolerance on the rescaled targeting score.
#' @param ... Additional named controls passed to cv.hapc (e.g. max_iter).
#' @return A list with estimate, se, lower, upper, eif, epsilon, initial plug-in
#'   and one-step comparisons, initial/targeted predictions, folds, and diagnostics.
#' @details The update solves the two arm-specific scores with covariates
#'   A/g and (1-A)/(1-g), holding g fixed. The estimate is the mean difference
#'   of the targeted predictions, not an AIPW correction. Wald inference requires
#'   causal identification, positivity, suitable nuisance rates and moments;
#'   cross-fitting and a zero empirical score alone do not prove coverage.
#'   ATT, ATC, missingness, clustered data and proximal estimands are outside
#'   this initial API. Existing ate.hapc is unchanged.
#' @examples
#' \dontrun{
#' set.seed(1)
#' W <- matrix(runif(400), 200, 2)
#' A <- rbinom(200, 1, 0.5)
#' Y <- A + W[, 1] + rnorm(200)
#' fit <- hapc.ate.tmle(W, Y, A, g = 0.5, cf_folds = 2)
#' c(fit$estimate, fit$lower, fit$upper)
#' }
#' @export
hapc.ate.tmle <- function(X, Y, A, family = c("auto", "binomial", "gaussian"),
                          outcome_bounds = NULL, alpha = 0.05,
                          cf_folds = 5L, seed = 1L, g = NULL, Q = NULL,
                          g_bound = 0.025, q_bound = 1e-6,
                          max_degree = 2L, npcs = NULL, nfolds = 5L, norm = "1",
                          log_lambda_min = -8, log_lambda_max = 2,
                          grid_length = 16L, target_tol = 1e-8, ...) {
  family <- match.arg(family)
  norm <- match.arg(norm, c("1", "2"))
  X <- as.matrix(X)
  if (!is.numeric(X) || any(!is.finite(X)) || ncol(X) < 1L)
    stop("X must be a finite numeric matrix with at least one column.")
  n <- nrow(X)
  if (n < 4L || !is.numeric(Y) || length(Y) != n || any(!is.finite(Y)))
    stop("Y must be a finite numeric vector with nrow(X) >= 4 observations.")
  if (!(is.numeric(A) || is.logical(A)) || length(A) != n ||
      anyNA(A) || !all(A %in% c(0, 1)) || length(unique(A)) != 2L)
    stop("A must contain both treatment arms, coded 0/1.")
  A <- as.numeric(A)
  scalar <- function(x, lo, hi) is.numeric(x) && length(x) == 1L &&
    is.finite(x) && x > lo && x < hi
  if (!scalar(alpha, 0, 1) || !scalar(g_bound, 0, .5) ||
      !scalar(q_bound, 0, .5) || !scalar(target_tol, 0, 1))
    stop("Invalid alpha, g_bound, q_bound or target_tol.")
  integer_control <- function(x, lower) is.numeric(x) && length(x) == 1L &&
    is.finite(x) && x == as.integer(x) && x >= lower
  if (!integer_control(cf_folds, 1) || cf_folds > min(table(A)) ||
      !integer_control(nfolds, 2) || !integer_control(max_degree, 1) ||
      !integer_control(grid_length, 2) ||
      (!is.null(npcs) && !integer_control(npcs, 1)))
    stop("Invalid fold, degree, rank or grid size; cf_folds cannot exceed either arm count.")
  if (!scalar(seed, -1, .Machine$integer.max) || seed != as.integer(seed))
    stop("seed must be a nonnegative integer.")
  if (!is.numeric(log_lambda_min) || !is.numeric(log_lambda_max) ||
      length(log_lambda_min) != 1L || length(log_lambda_max) != 1L ||
      !is.finite(log_lambda_min) || !is.finite(log_lambda_max) ||
      log_lambda_min >= log_lambda_max) stop("Invalid log-lambda interval.")
  dots <- list(...)
  reserved <- c("X", "Y", "family", "max_degree", "npcs", "nfolds", "norm",
                "predict", "log_lambda_min", "log_lambda_max", "grid_length")
  if (length(dots) && (is.null(names(dots)) || any(!nzchar(names(dots))) ||
                       anyDuplicated(names(dots)) || any(names(dots) %in% reserved)))
    stop("Additional cv.hapc controls must have distinct, non-reserved names.")
  if (family == "auto") family <- if (all(Y %in% c(0, 1))) "binomial" else "gaussian"
  if (family == "binomial" && !all(Y %in% c(0, 1)))
    stop("Binomial Y must be coded 0/1.")
  if (family == "binomial") {
    if (!is.null(outcome_bounds) && !identical(as.numeric(outcome_bounds), c(0, 1)))
      stop("Binomial outcome_bounds must be c(0,1).")
    outcome_bounds <- c(0, 1)
  }
  if (!is.null(outcome_bounds) &&
      (!is.numeric(outcome_bounds) || length(outcome_bounds) != 2L ||
       any(!is.finite(outcome_bounds)) || diff(outcome_bounds) <= 0 ||
       any(Y < outcome_bounds[1] | Y > outcome_bounds[2])))
    stop("outcome_bounds must be increasing finite bounds containing every Y.")
  check_pred <- function(x, size, label) {
    if (!is.numeric(x) || length(x) != size || any(!is.finite(x)))
      stop(label, " predictions have invalid size or non-finite values.")
    as.numeric(x)
  }
  if (!is.null(g)) {
    if (length(g) == 1L) g <- rep(g, n)
    g <- check_pred(g, n, "g")
    if (any(g <= 0 | g >= 1)) stop("Supplied g must be strictly between 0 and 1.")
  }
  if (!is.null(Q)) {
    Q <- as.matrix(Q)
    if (!identical(dim(Q), c(n, 2L))) stop("Q must be n by 2, ordered Q0,Q1.")
    check_pred(Q, 2L*n, "Q")
  }
  folds <- if (cf_folds == 1L) rep(0L, n) else .ate_make_folds(n, cf_folds, A, seed)
  g_supplied <- !is.null(g); q_supplied <- !is.null(Q)
  if (!g_supplied) g <- numeric(n)
  if (!q_supplied) Q <- matrix(NA_real_, n, 2L)
  fits <- vector("list", cf_folds)
  # All model construction/tuning is inside the outer training sample.
  fit_predict <- function(x, y, newx, fam) {
    if (length(unique(y)) == 1L) return(list(pred = rep(y[1], nrow(newx)), lambda = NA_real_))
    args <- c(list(X = x, Y = y, family = fam, max_degree = max_degree,
                   npcs = min(if (is.null(npcs)) nrow(x) else npcs, nrow(x)),
                   nfolds = nfolds, norm = norm, predict = newx,
                   log_lambda_min = log_lambda_min, log_lambda_max = log_lambda_max,
                   grid_length = grid_length), dots)
    cv <- do.call(cv.hapc, args)
    # cv.hapc returns response-scale probabilities for BOTH supported norms;
    # unlike single-lambda hapc(norm='1'), these are not log-odds.
    pred <- check_pred(cv$predictions, nrow(newx), fam)
    if (fam == "binomial" && any(pred < 0 | pred > 1))
      stop("Binomial CV did not return probabilities.")
    list(pred = pred, lambda = cv$best_lambda)
  }
  for (k in seq_len(cf_folds)) {
    te <- which(folds == k-1L)
    tr <- if (cf_folds == 1L) seq_len(n) else which(folds != k-1L)
    if ((!g_supplied || !q_supplied) && length(tr) < nfolds)
      stop("nfolds exceeds an outer training sample size.")
    if (!g_supplied) {
      gf <- fit_predict(X[tr,,drop=FALSE], A[tr], X[te,,drop=FALSE], "binomial")
      g[te] <- gf$pred; fits[[k]]$lambda_g <- gf$lambda
    }
    if (!q_supplied) {
      xt <- cbind(A[tr], X[tr,,drop=FALSE])
      xe <- rbind(cbind(0, X[te,,drop=FALSE]), cbind(1, X[te,,drop=FALSE]))
      qf <- fit_predict(xt, Y[tr], xe, family)
      Q[te,] <- matrix(qf$pred, nrow=length(te), ncol=2L)
      fits[[k]]$lambda_Q <- qf$lambda
    }
  }
  g_raw <- g
  g <- pmax(g_bound, pmin(1-g_bound, g))
  initial_Q <- Q
  bounded <- !is.null(outcome_bounds)
  clipped_Q <- 0
  if (bounded) {
    shift <- outcome_bounds[1]; scale <- diff(outcome_bounds)
    ys <- (Y-shift)/scale
    qs <- (Q-shift)/scale
    clipped_Q <- mean(qs < q_bound | qs > 1-q_bound)
    qs <- pmax(pmin(qs, 1-q_bound), q_bound)
    initial_Q <- shift + scale*qs
  } else {
    shift <- 0; scale <- 1; ys <- Y; qs <- Q
  }
  eps <- numeric(2L); score <- numeric(2L); targeted <- qs
  for (a in 0:1) {
    j <- a+1L; obs <- which(A == a)
    H <- if (a == 1) 1/g else 1/(1-g)
    if (bounded) {
      offset <- stats::qlogis(qs[,j])
      fun <- function(e) mean(H[obs]*(ys[obs]-stats::plogis(offset[obs]+e*H[obs])))
      # A constant outcome within an arm gives an infinite logistic MLE.
      # Fail explicitly rather than reporting a misleading Wald interval.
      if (all(ys[obs] == 0) || all(ys[obs] == 1))
        stop("Targeting has no finite logistic solution in arm ", a, ".")
      bracket <- 1
      while (fun(-bracket)*fun(bracket) > 0 && bracket < 1e6) bracket <- bracket*2
      if (fun(-bracket)*fun(bracket) > 0) stop("Unable to bracket targeting score.")
      eps[j] <- stats::uniroot(fun, c(-bracket, bracket), tol=1e-12)$root
      targeted[,j] <- stats::plogis(offset + eps[j]*H)
    } else {
      eps[j] <- sum(H[obs]*(ys[obs]-qs[obs,j]))/sum(H[obs]^2)
      targeted[,j] <- qs[,j] + eps[j]*H
    }
    score[j] <- mean((A == a)*H*(ys-targeted[,j]))
  }
  if (max(abs(score)) > target_tol) stop("Targeting did not satisfy the score tolerance.")
  Qstar <- shift + scale*targeted
  observed <- function(q) q[cbind(seq_len(n), as.integer(A)+1L)]
  H <- A/g-(1-A)/(1-g)
  plug <- mean(initial_Q[,2]-initial_Q[,1])
  one_step <- plug + mean(H*(Y-observed(initial_Q)))
  estimate <- mean(Qstar[,2]-Qstar[,1])
  eif <- H*(Y-observed(Qstar)) + Qstar[,2]-Qstar[,1]-estimate
  se <- stats::sd(eif)/sqrt(n)
  z <- stats::qnorm(1-alpha/2)
  names(eps) <- c("Q0", "Q1")
  list(estimate=estimate, se=se, lower=estimate-z*se, upper=estimate+z*se,
       alpha=alpha, eif=eif, epsilon=eps, plugin=plug, one_step=one_step,
       one_step_se=stats::sd(H*(Y-observed(initial_Q)) + initial_Q[,2]-initial_Q[,1])/sqrt(n),
       Q_initial=initial_Q, Q_star=Qstar, g=g, g_raw=g_raw, folds=folds,
       diagnostics=list(converged=TRUE, score=score, eif_mean=mean(eif),
                        g_clipped_fraction=mean(g != g_raw), Q_clipped_fraction=clipped_Q,
                        g_range=range(g_raw), fluctuation=if (bounded) "logistic" else "gaussian",
                        family=family, outcome_bounds=outcome_bounds, norm=norm,
                        crossfit=cf_folds > 1L, supplied_g=g_supplied, supplied_Q=q_supplied,
                        nuisance_fits=fits), call=match.call())
}
