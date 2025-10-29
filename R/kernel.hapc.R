#' @export


kernel.hapc <- function(X,m,center=TRUE) {
  p = ncol(X)
  .Call("mkernel_call", X, as.integer(m), as.logical(center), PACKAGE = "hapc")
}