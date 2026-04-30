#' Design matrix generation for PC-HA 
#'
#' Generates the design matrix ingredients on which empirical risk minimization
#' is run for the PC-HA family.
#'
#' Counterpart to Python \code{hapc.design_hapc()}.
#'
#' @seealso \code{\link{hapc}}, \code{\link{kernel.hapc}}
#'
#' @param X Numeric matrix of features (\code{nrow} = observations,
#'   \code{ncol} = covariates).
#' @param max_degree Integer; maximum HAL interaction order. Default \code{1L}.
#' @param npcs Integer; number of singular triplets to retain. Default
#'   \code{nrow(X)} (internally capped when \code{center = TRUE}).
#' @param center Logical; centre \code{H} before the SVD. Default \code{TRUE}.
#'
#' @return A list containing:
#' \describe{
#'   \item{H}{HAL design matrix}
#'   \item{U}{\eqn{n \times k} matrix of left singular vectors (first \eqn{k} columns).}
#'   \item{d}{Length-\eqn{k} vector of singular values of \eqn{H}.}
#'   \item{V}{\eqn{B \times k} matrix of right singular vectors (\eqn{B} = number of HAL basis columns).}
#' }
#'
#' @examples
#' \dontrun{
#' # Create sample data
#' X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
#' 
#' # Generate design matrix with default parameters
#' design_mat <- design.hapc(X)
#' 
#' # Generate design matrix with quadratic features
#' design_mat <- design.hapc(X, max_degree = 2, npcs = 50)
#' }
#'
#' @export
design.hapc <- function(X, max_degree = 1, npcs = nrow(X), center=TRUE) {
  .Call("pchal_des", X, as.integer(max_degree), as.integer(npcs), as.logical(center), PACKAGE = "hapc")
}