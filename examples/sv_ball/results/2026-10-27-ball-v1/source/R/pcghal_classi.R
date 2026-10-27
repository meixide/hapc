#' Direct PC-HAL classification optimiser (precomputed basis)
#'
#' Runs projected gradient descent for the binomial / logistic-loss HAPC
#' problem given pre-computed PC ingredients (\code{Xtilde = U D},
#' \code{E_Nn = V}) and an initial \eqn{\alpha_0}.  Mirrors Python's
#' \code{hapc.pcghal_classification()}.
#'
#' Useful when the design has already been built (e.g. when iterating across
#' many lambdas while keeping \eqn{H} fixed).
#'
#' @param Y Numeric \{-1, +1\} response vector.
#' @param Xtilde \eqn{n \times k} matrix \eqn{U D} (left singular vectors scaled
#'   by the top-\eqn{k} singular values).
#' @param E_Nn \eqn{B \times k} matrix \eqn{V} from \code{\link{design.hapc}}
#'   (right singular vectors; \eqn{B} is the HAL basis dimension).
#' @param alpha Numeric vector of initial coefficients.
#' @param max_iter Integer; maximum PGD iterations. Default \code{5000L}.
#' @param tol Numeric convergence tolerance. Default \code{1e-3}.
#' @param step_factor Numeric line-search factor. Default \code{0.8}.
#' @param verbose Logical; print progress. Default \code{FALSE}.
#'
#' @return A list with \code{alpha}, \code{alphaiters}, \code{beta},
#'   \code{risk}, and \code{iter}.
#'
#' @examples
#' \dontrun{
#' n <- 80; p <- 3
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' Y <- sample(c(-1, 1), n, replace = TRUE)
#' des <- design.hapc(X, max_degree = 2, npcs = 10)
#' k <- min(10L, length(des$d))
#' Xtilde <- des$U[, 1:k, drop = FALSE] * rep(des$d[1:k], each = nrow(des$U))
#' E_Nn <- des$V[, 1:k, drop = FALSE]
#' a0 <- rep(0, k)
#' fit <- pc_hal_classi_cpp(Y, Xtilde, E_Nn, a0, max_iter = 100, tol = 1e-2)
#' str(fit)
#' }
#'
#' @export
pc_hal_classi_cpp <- function(Y, Xtilde, E_Nn, alpha,
                              max_iter = 5000L,
                              tol = 1e-3,
                              step_factor = 0.8,
                              verbose = FALSE) {

  if (!is.numeric(Y))     stop("Y must be numeric")
  if (!is.matrix(Xtilde)) stop("Xtilde must be a matrix")
  if (!is.matrix(E_Nn))   stop("E_Nn must be a matrix")
  if (!is.numeric(alpha)) stop("alpha must be numeric")

  .Call(
    "pcghal_classi_call",
    as.numeric(Y), as.matrix(Xtilde), as.matrix(E_Nn), as.numeric(alpha),
    as.integer(max_iter), as.numeric(tol),
    as.numeric(step_factor), as.logical(verbose),
    PACKAGE = "hapc"
  )
}
