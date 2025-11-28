#' @export
#' 
#' 

pc_hal_classi_cpp <- function(Y, Xtilde, E_Nn, alpha, 
                               max_iter = 100, 
                               tol = 1e-9, 
                               step_factor = 0.5, 
                               verbose = FALSE) {
  
  # Input validation
  if (!is.numeric(Y)) stop("Y must be numeric")
  if (!is.matrix(Xtilde)) stop("Xtilde must be a matrix")
  if (!is.matrix(E_Nn)) stop("E_Nn must be a matrix")
  if (!is.numeric(alpha)) stop("alpha must be numeric")
  
  # Ensure proper types
  Y <- as.numeric(Y)
  Xtilde <- as.matrix(Xtilde)
  E_Nn <- as.matrix(E_Nn)
  alpha <- as.numeric(alpha)
  
  # Call C++ function
  result <- .Call("pcghal_classi_call", 
                  Y, Xtilde, E_Nn, alpha, 
                  as.integer(max_iter), 
                  as.numeric(tol), 
                  as.numeric(step_factor), 
                  as.logical(verbose))
  
  return(result)
}