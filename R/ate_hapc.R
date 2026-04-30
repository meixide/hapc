#' Average Treatment Effect via HAPC + outcome undersmoothing
#'
#' \code{ate.hapc} estimates the Average Treatment Effect (ATE)
#' \eqn{\psi = E[Y(1) - Y(0)]} using HAPC for both nuisances, with the
#' outcome regularisation parameter \emph{undersmoothed} so that the
#' empirical mean of the efficient influence function (EIF) falls below
#' \eqn{\sigma / (\sqrt{n} \log n)}.
#'
#' Mirror of the Python function \code{hapc.ate_hapc()}.  Argument names,
#' defaults and behaviour are identical to \code{cv.hapc()} except that
#' \code{family} is removed (binomial for propensity, gaussian for outcome), separate
#' log-\eqn{\lambda} grids are used for propensity vs outcome/undersmoothing,
#' and \code{A} (binary treatment) plus \code{alpha} are added.
#'
#' @param max_degree,npcs,nfolds,norm,predict,max_iter,tol,step_factor,verbose,crit,center,approx,ini
#'   Same meaning as in \code{\link{cv.hapc}} (\code{family} is fixed internally).
#'   \code{predict} is accepted for signature parity and ignored (EIF on training sample only).
#' @param X Numeric matrix of covariates \eqn{W} (rows = observations).
#'   \emph{Do not} include the treatment column; it is added internally for the
#'   outcome model.
#' @param Y Numeric continuous outcome of length \code{nrow(X)}.
#' @param A Binary treatment in \eqn{\{0,1\}} or \eqn{\{-1,+1\}} of length
#'   \code{nrow(X)}.
#' @param alpha Numeric in \eqn{(0,1)}.  The returned interval has confidence
#'   \eqn{1 - \alpha}.  Default \code{0.05}.
#' @param log_lambda_prop_min,log_lambda_prop_max,grid_length_prop
#'   Log-\eqn{\lambda} grid for propensity cross-validation only (\code{A ~ W},
#'   binomial), same construction as \code{cv.hapc}.
#' @param log_lambda_out_min,log_lambda_out_max,grid_length_out
#'   Log-\eqn{\lambda} grid for outcome cross-validation (\code{Y ~ (A,W)},
#'   gaussian) and for the undersmoothing scan over outcome \eqn{\lambda}.
#' @param plot_diagnostics Logical; if \code{TRUE}, draw three base-\code{graphics}
#'   panels before returning: (1) propensity CV curve (logistic deviance vs \eqn{\lambda}),
#'   (2) outcome CV curve (MSE vs \eqn{\lambda}), (3) undersmoothing trajectory
#'   \eqn{|\bar{\varphi}(\lambda)|} vs outcome \eqn{\lambda} with the threshold
#'   line and vertical markers for the CV and selected undersmoothed \eqn{\lambda}.
#'   Default \code{FALSE}.
#'
#' @return A named list with three numeric scalars:
#'   \item{estimate}{Plug-in ATE at the undersmoothed outcome model:
#'     \code{mean(mu_hat_1(W) - mu_hat_0(W))}.}
#'   \item{lower}{Lower endpoint of the \eqn{1-\alpha} Wald CI.}
#'   \item{upper}{Upper endpoint of the \eqn{1-\alpha} Wald CI.}
#'
#' @details
#' The procedure is:
#' \enumerate{
#'   \item Cross-validate the propensity model \code{A ~ W} (binomial) on its
#'     grid and the outcome model \code{Y ~ (A, W)} (gaussian) on the outcome
#'     grid.
#'   \item Fix the propensity at its CV-best \eqn{\lambda} and refit on the
#'     full sample to obtain \eqn{\hat\pi(W_i) = P(A=1 | W_i)}.
#'   \item At the CV-best outcome \eqn{\lambda}, compute the ATE EIF
#'     \deqn{\hat\varphi_i = \frac{A_i}{\hat\pi_i}(Y_i - \hat\mu_1(W_i)) -
#'           \frac{1-A_i}{1-\hat\pi_i}(Y_i - \hat\mu_0(W_i)) +
#'           \hat\mu_1(W_i) - \hat\mu_0(W_i) -
#'           (\bar{\hat\mu_1} - \bar{\hat\mu_0})}
#'     and let \eqn{\sigma = \mathrm{sd}_n(\hat\varphi)}.
#'   \item Threshold \eqn{\tau = \sigma / (\sqrt{n} \log n)}.
#'   \item Walk the outcome \eqn{\lambda} grid in
#'     \emph{decreasing} order; pick the first (largest) \eqn{\lambda} for
#'     which \eqn{|\bar\varphi| \le \tau}.  Call it \eqn{\lambda_u}.  If no
#'     grid point meets the threshold, fall back to the smallest
#'     \eqn{\lambda} in the grid.
#'   \item Plug-in estimate
#'     \eqn{\hat\psi = \mathrm{mean}(\hat\mu_1(W; \lambda_u) -
#'                                    \hat\mu_0(W; \lambda_u))}.
#'     CI \eqn{\hat\psi \pm z_{1-\alpha/2}\, \sigma_u / \sqrt{n}} with
#'     \eqn{\sigma_u} the s.d.\ of the EIF at \eqn{\lambda_u}.
#' }
#'
#' No sample splitting / cross-fitting is performed; bias control comes from
#' the undersmoothing step.
#'
#' @examples
#' \dontrun{
#' n <- 200
#' W <- cbind(runif(n, -2, 2), rnorm(n, sd = 0.5))
#' p <- 1 / (1 + exp(-(W[,1] + 0.5 * W[,2])))
#' A <- rbinom(n, 1, p)
#' Y <- 2 * W[,1] + 0.5 + rnorm(n, sd = 0.5)        # truth: ATE = 0
#' ate.hapc(W, Y, A, alpha = 0.05, max_degree = 2, npcs = 50,
#'          grid_length_prop = 4L, grid_length_out = 4L, nfolds = 3L, norm = "2")
#' }
#'
#' @export
ate.hapc <- function(X, Y, A,
                     alpha = 0.05,
                     max_degree = 1L,
                     npcs = nrow(X),
                     log_lambda_prop_min = -5,
                     log_lambda_prop_max = -3,
                     grid_length_prop = 10L,
                     log_lambda_out_min = -5,
                     log_lambda_out_max = -3,
                     grid_length_out = 10L,
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
                     plot_diagnostics = FALSE) {

  norm <- match.arg(norm)
  crit <- match.arg(crit)
  ini  <- match.arg(ini)

  if (!is.numeric(alpha) || length(alpha) != 1L ||
      !(alpha > 0 && alpha < 1)) {
    stop("alpha must be a single numeric in (0,1).")
  }

  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  Y <- as.numeric(Y)
  A <- as.numeric(A)
  n <- nrow(X)
  if (length(Y) != n || length(A) != n) {
    stop("X, Y and A must all have the same number of rows.")
  }

  uA <- sort(unique(A))
  if (all(uA %in% c(0, 1))) {
    A01 <- A
  } else if (all(uA %in% c(-1, 1))) {
    A01 <- (A + 1) / 2
  } else {
    stop("A must be binary {0,1} or {-1,+1}; found: ",
         paste(uA, collapse = ", "))
  }

  max_degree          <- as.integer(max_degree)
  npcs                <- as.integer(npcs)
  log_lambda_prop_min <- as.numeric(log_lambda_prop_min)
  log_lambda_prop_max <- as.numeric(log_lambda_prop_max)
  grid_length_prop    <- as.integer(grid_length_prop)
  log_lambda_out_min  <- as.numeric(log_lambda_out_min)
  log_lambda_out_max  <- as.numeric(log_lambda_out_max)
  grid_length_out     <- as.integer(grid_length_out)
  nfolds              <- as.integer(nfolds)
  max_iter            <- as.integer(max_iter)
  tol                 <- as.numeric(tol)
  step_factor         <- as.numeric(step_factor)
  verbose             <- as.logical(verbose)
  center              <- as.logical(center)
  approx              <- as.logical(approx)

  lambdas_out <- exp(seq(log_lambda_out_min, log_lambda_out_max,
                         length.out = grid_length_out))

  cv_args_base <- list(
    max_degree = max_degree, npcs = npcs, nfolds = nfolds, norm = norm,
    predict = NULL, max_iter = max_iter, tol = tol, step_factor = step_factor,
    verbose = verbose, crit = crit, center = center, approx = approx, ini = ini
  )

  cv_prop <- do.call(
    cv.hapc,
    c(list(
      X = X, Y = A01, family = "binomial",
      log_lambda_min = log_lambda_prop_min,
      log_lambda_max = log_lambda_prop_max,
      grid_length = grid_length_prop
    ), cv_args_base)
  )
  lam_prop_cv <- as.numeric(cv_prop$best_lambda)

  prop_fit <- hapc(
    X, A01, family = "binomial",
    max_degree = max_degree, npcs = npcs,
    lambda = lam_prop_cv, norm = norm, predict = X,
    max_iter = max_iter, tol = tol, step_factor = step_factor,
    verbose = verbose, crit = crit, center = center, approx = approx,
    ini = ini
  )
  pi1 <- .ate_propensity_probs(prop_fit)
  pi1 <- pmin(pmax(pi1, 1e-8), 1 - 1e-8)

  Xout <- cbind(A01, X)
  colnames(Xout) <- NULL
  storage.mode(Xout) <- "double"

  cv_out <- do.call(
    cv.hapc,
    c(list(
      X = Xout, Y = Y, family = "gaussian",
      log_lambda_min = log_lambda_out_min,
      log_lambda_max = log_lambda_out_max,
      grid_length = grid_length_out
    ), cv_args_base)
  )
  lam_out_cv <- as.numeric(cv_out$best_lambda)

  Xmu1 <- cbind(rep(1, n), X); colnames(Xmu1) <- NULL
  Xmu0 <- cbind(rep(0, n), X); colnames(Xmu0) <- NULL
  storage.mode(Xmu1) <- "double"
  storage.mode(Xmu0) <- "double"
  Xeval <- rbind(Xmu1, Xmu0)
  storage.mode(Xeval) <- "double"

  .mu_pair <- function(lam) {
    res <- hapc(
      Xout, Y, family = "gaussian",
      max_degree = max_degree, npcs = npcs,
      lambda = as.numeric(lam), norm = norm, predict = Xeval,
      max_iter = max_iter, tol = tol, step_factor = step_factor,
      verbose = verbose, crit = crit, center = center, approx = approx,
      ini = ini
    )
    pred <- .ate_outcome_predictions(res)
    if (length(pred) != 2L * n) {
      stop("Outcome prediction returned ", length(pred),
           " values, expected ", 2L * n, ".")
    }
    list(mu1 = pred[seq_len(n)], mu0 = pred[(n + 1L):(2L * n)])
  }

  .eif_diff <- function(mu1, mu0) {
    eif1 <- (A01 / pi1) * (Y - mu1) - (mu1 - mean(mu1))
    eif0 <- ((1 - A01) / (1 - pi1)) * (Y - mu0) - (mu0 - mean(mu0))
    eif1 - eif0
  }

  .pop_sd <- function(x) sqrt(mean((x - mean(x))^2))

  cv_pair <- .mu_pair(lam_out_cv)
  eif_cv <- .eif_diff(cv_pair$mu1, cv_pair$mu0)
  sigma_cv <- .pop_sd(eif_cv)
  threshold <- sigma_cv / (sqrt(n) * log(n))

  lam_grid_dec <- sort(lambdas_out, decreasing = TRUE)

  lam_und <- NA_real_
  eif_und <- NULL
  und_pair <- NULL
  for (lam in lam_grid_dec) {
    pair <- tryCatch(.mu_pair(lam), error = function(e) NULL)
    if (is.null(pair)) next
    eif <- .eif_diff(pair$mu1, pair$mu0)
    if (abs(mean(eif)) <= threshold) {
      lam_und <- lam
      und_pair <- pair
      eif_und <- eif
      break
    }
  }

  if (is.null(eif_und)) {
    lam_und <- min(lambdas_out)
    und_pair <- .mu_pair(lam_und)
    eif_und <- .eif_diff(und_pair$mu1, und_pair$mu0)
  }

  if (isTRUE(plot_diagnostics)) {
    traj_lam <- numeric(0)
    traj_abs <- numeric(0)
    for (lam in sort(lambdas_out)) {
      pair <- tryCatch(.mu_pair(lam), error = function(e) NULL)
      if (is.null(pair)) next
      eif <- .eif_diff(pair$mu1, pair$mu0)
      traj_lam <- c(traj_lam, lam)
      traj_abs <- c(traj_abs, abs(mean(eif)))
    }
    .ate_plot_diagnostics(
      cv_prop, cv_out, traj_lam, traj_abs,
      lam_prop_cv, lam_out_cv, lam_und, threshold
    )
  }

  psi <- mean(und_pair$mu1 - und_pair$mu0)
  sigma_und <- .pop_sd(eif_und)
  z <- stats::qnorm(1 - alpha / 2)
  half <- z * sigma_und / sqrt(n)

  list(
    estimate = as.numeric(psi),
    lower    = as.numeric(psi - half),
    upper    = as.numeric(psi + half)
  )
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Extract probabilities from the result of a binomial single-lambda hapc fit.
# Different norm paths return either probabilities or log-odds in
# `$predictions`.  This helper normalises to probabilities in [0,1].
.ate_propensity_probs <- function(res) {
  if (!is.null(res$probabilities)) {
    return(as.numeric(res$probabilities))
  }
  if (is.null(res$predictions)) {
    stop("Propensity fit did not return predictions; cannot compute pi_hat(W).")
  }
  p <- as.numeric(res$predictions)
  if (any(!is.finite(p))) {
    stop("Propensity predictions contain non-finite values.")
  }
  if (any(p < 0) || any(p > 1)) {
    1 / (1 + exp(-p))
  } else {
    p
  }
}

# Extract continuous outcome predictions from a gaussian single-lambda hapc
# fit.  Predictions are stored in `$predictions` for all supported norms.
.ate_outcome_predictions <- function(res) {
  if (is.null(res$predictions)) {
    stop("Outcome fit did not return predictions; ",
         "ate.hapc requires `predict=` to be honoured by the underlying solver.")
  }
  as.numeric(res$predictions)
}

# Base-graphics diagnostic figure (propensity CV, outcome CV, undersmoothing path).
.ate_plot_diagnostics <- function(cv_prop, cv_out, traj_lambdas, traj_abs_mean,
                                   lam_prop_cv, lam_out_cv, lam_und, threshold) {
  lam_p <- as.numeric(cv_prop$lambdas)
  sc_p  <- if (!is.null(cv_prop$deviances)) {
    as.numeric(cv_prop$deviances)
  } else {
    as.numeric(cv_prop$mses)
  }
  lam_o <- as.numeric(cv_out$lambdas)
  sc_o  <- as.numeric(cv_out$mses)

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::layout(1L)
    graphics::par(old_par)
  })
  graphics::layout(matrix(c(1L, 2L, 3L, 3L), nrow = 2L, ncol = 2L, byrow = TRUE))
  graphics::par(cex = 0.9, mgp = c(2, 0.6, 0))

  o_p <- order(lam_p)
  graphics::plot(lam_p[o_p], sc_p[o_p], log = "x", type = "b", pch = 16,
                 col = grDevices::hcl(h = 40, c = 80, l = 50),
                 xlab = expression(lambda ~ "(propensity)"),
                 ylab = "Mean CV logistic deviance",
                 main = "Propensity CV (A ~ W, binomial)")
  graphics::grid()
  graphics::abline(v = lam_prop_cv, col = "#d62728", lty = 2L, lwd = 1.6)

  o_o <- order(lam_o)
  graphics::plot(lam_o[o_o], sc_o[o_o], log = "x", type = "b", pch = 16,
                 col = grDevices::hcl(h = 140, c = 70, l = 45),
                 xlab = expression(lambda ~ "(outcome)"),
                 ylab = "Mean CV MSE",
                 main = "Outcome CV (Y ~ (A,W), gaussian)")
  graphics::grid()
  graphics::abline(v = lam_out_cv, col = "#d62728", lty = 2L, lwd = 1.6)

  ord <- order(traj_lambdas)
  tl <- as.numeric(traj_lambdas[ord])
  ta <- as.numeric(traj_abs_mean[ord])
  ok <- is.finite(tl) & is.finite(ta) & (tl > 0)
  tl <- tl[ok]
  ta <- ta[ok]

  xlab_tr <- "Outcome lambda (undersmoothing grid)"
  ylab_tr <- expression(group("|", bar(varphi), "|"))
  main_tr <- "Undersmoothing trajectory (propensity fixed at CV-lambda)"

  if (length(tl)) {
    graphics::plot(tl, ta, log = "x", type = "b", pch = 16, lwd = 1.5,
                   col = "#1f77b4",
                   xlab = xlab_tr, ylab = ylab_tr, main = main_tr)
    graphics::grid()
    graphics::polygon(
      x = c(tl, rev(tl)),
      y = c(rep(0, length(tl)), rep(threshold, length(tl))),
      col = grDevices::adjustcolor("gray50", alpha.f = 0.12),
      border = NA
    )
    graphics::lines(tl, ta, type = "b", pch = 16, col = "#1f77b4", lwd = 1.5)
  } else {
    xr <- range(lam_o, finite = TRUE)
    if (!all(is.finite(xr)) || length(xr) != 2L || xr[1] <= 0) {
      xr <- c(1e-5, 1)
    } else if (identical(xr[1], xr[2])) {
      xr <- c(xr[1] * 0.5, xr[1] * 2)
    }
    ymax <- max(threshold, 1e-8, na.rm = TRUE)
    graphics::plot(NA_real_, NA_real_, log = "x",
                   xlim = xr, ylim = c(0, ymax * 1.05),
                   xlab = xlab_tr, ylab = ylab_tr, main = main_tr)
    graphics::grid()
    graphics::text(exp(mean(log(xr))), ymax * 0.5, "No valid outcome fits on lambda grid")
  }
  graphics::abline(h = threshold, col = "gray35", lwd = 2)
  graphics::abline(v = lam_out_cv, col = "#d62728", lty = 2L, lwd = 1.6)
  graphics::abline(v = lam_und, col = "#9467bd", lwd = 2)

  graphics::legend(
    "topright",
    legend = c("|mean(EIF)|",
               "Threshold (sigma_CV / (sqrt(n) log n))",
               "Outcome CV lambda", "Undersmoothed lambda"),
    col = c("#1f77b4", "gray35", "#d62728", "#9467bd"),
    lty = c(1L, 1L, 2L, 1L), lwd = c(1.5, 2, 1.6, 2), pch = c(16L, NA, NA, NA),
    bty = "n", cex = 0.72, ncol = 1L
  )
}
