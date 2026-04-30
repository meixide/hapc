#' Kernel matrix for the HAPC HAL basis
#'
#' Computes the (centred or uncentred) kernel matrix \eqn{K = H H^\top}, where
#' \eqn{H} is the HAL (Highly Adaptive Lasso) zero/one design matrix induced
#' by tensor-product knot indicators of order \eqn{\le} \code{max_degree}.
#' \eqn{H} is never materialised; the kernel is computed implicitly using the
#' technique of Schuler et al. ("Highly Adaptive Ridge").
#'
#' Counterpart to Python \code{hapc.kernel_hapc()}.
#'
#' @param X Numeric matrix of features.
#' @param max_degree Integer; maximum interaction order. Default \code{1L}.
#' @param center Logical; whether to centre \eqn{H} (i.e. subtract column
#'   means before \eqn{HH^\top}). Default \code{TRUE}.
#'
#' @return An \eqn{n \times n} numeric kernel matrix.
#'
#' @examples
#' \dontrun{
#' X <- matrix(rnorm(100 * 5), 100, 5)
#' K <- kernel.hapc(X, max_degree = 2, center = TRUE)
#' }
#'
#' @export
kernel.hapc <- function(X, max_degree = 1L, center = TRUE) {
  .Call("mkernel_call", X, as.integer(max_degree), as.logical(center),
        PACKAGE = "hapc")
}
