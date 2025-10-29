#' @export
design.hapc <- function(X, max_degree = 1, npcs = 100, center=TRUE) {
  .Call("pchal_des", X, as.integer(max_degree), as.integer(npcs), as.logical(center), PACKAGE = "hapc")
}