#' @useDynLib hapc, .registration = TRUE 

.onAttach <- function(libname, pkgname) {
  packageStartupMessage("Loaded hapc")
}