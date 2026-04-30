#' Cross kernel matrix between train and test points
#'
#' Computes the cross-kernel \eqn{K_{te,tr} = H_{te} H_{tr}^\top} of the HAL
#' basis induced by \code{X} (train) at the points \code{Xte} (test). Used
#' under the hood by HAPC predictions.
#'
#' Counterpart to Python \code{hapc.cross_kernel_hapc()}.
#'
#' @param X Numeric matrix of training features (used to define the basis).
#' @param Xte Numeric matrix of test features. Must have the same number of
#'   columns as \code{X}.
#' @param max_degree Integer; maximum interaction order (must match the value
#'   used to fit). Default \code{1L}.
#' @param center Logical; whether to apply the same column centring used in
#'   training. Default \code{TRUE}.
#'
#' @return An \eqn{n_{te} \times n_{tr}} numeric matrix.
#'
#' @examples
#' \dontrun{
#' X  <- matrix(rnorm(100 * 5), 100, 5)
#' Xn <- matrix(rnorm( 20 * 5),  20, 5)
#' Kx <- cross_kernel.hapc(X, Xn, max_degree = 1)
#' }
#'
#' @export
cross_kernel.hapc <- function(X, Xte, max_degree = 1L, center = TRUE) {
  .Call("kernel_cross_call", X, Xte, as.integer(max_degree),
        as.logical(center), PACKAGE = "hapc")
}
