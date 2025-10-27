#' @export


kernel.hapc <- function(X,m) {
  p = ncol(X)
  .Call("mkernel_call", X, as.integer(m), PACKAGE = "hapc")
}