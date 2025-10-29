#' @export


cross_kernel.hapc <- function(X,Xte,m,center=TRUE) {
  p = ncol(X)
  .Call("kernel_cross_call", X, Xte, as.integer(m), as.logical(center), PACKAGE = "hapc")
}