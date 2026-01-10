#' Design matrix generation for PC-HA 
#'
#' Generates the design matrix ingredients on which empirical risk minimization
#' is run for the PC-HA family. 
#'
#' @param X A numeric matrix containing the input features.
#' @param max_degree Integer specifying the maximum order of interaction terms (no more than max_degree - way interaction) for which
#' basis functions are generated. Default is 1 (no products) and might be increased until ncol(X).
#' @param npcs Integer specifying the number of principal components to retain.
#' Default is n.
#' @param center Logical indicating whether to center the basis functions before
#' processing. Default is TRUE.
#'
#' @return A list containing:
#' \describe{
#'   \item{H}{HAL design matrix}
#'   \item{U}{Matrix whose npcs columns are the left singular vectors of H.}
#'   \item{d}{Vector of the first npcs singular values of H.}
#'   \item{V}{Matrix whose npcs columns are the right singular vectors of H (principal components).}
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