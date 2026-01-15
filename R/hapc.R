#' HAPC Model Fitting
#'
#' Fits PCHA using either a sectional 
#' variation norm constraint, an L1 norm constraint or an L2 norm constraint.
#'
#' @param X A numeric matrix containing the input features.
#' @param Y A numeric vector containing the response variable. Must have length 
#'   equal to \code{nrow(X)}.
#' @param family Character string specifying the model family. Currently supports 
#'   \code{"gaussian"}. Default is \code{"gaussian"}.
#' @param max_degree Integer specifying the maximum order of interaction terms 
#'   (no more than max_degree-way interaction) for which basis functions are 
#'   generated. Default is 1 (no products) and might be increased until 
#'   \code{ncol(X)}.
#' @param npcs Integer specifying the number of principal components to retain. 
#'   Default is \code{nrow(X)}.
#' @param lambda Numeric regularization parameter for the ridge penalty. 
#'   Default is 0.01.
#' @param norm Character string specifying the norm constraint. Options are:
#'   \describe{
#'     \item{\code{"sv"}}{Sectional variation norm constraint (default)}
#'     \item{\code{"1"}}{L1 norm constraint}
#'     \item{\code{"2"}}{L2 norm constraint}
#'   }
#' @param predict Optional numeric matrix of new observations for which to make 
#'   predictions. Must have the same number of columns as \code{X}. 
#'   Default is \code{NULL} (no predictions).
#' @param max_iter Integer specifying the maximum number of iterations for the 
#'   optimization algorithm. Default is 100. Exclusive to \code{norm = "sv"}.
#' @param tol Numeric tolerance for convergence. Default is 1e-9. Exclusive to \code{norm = "sv"}.
#' @param step_factor Numeric step factor for the optimization algorithm. 
#'   Default is 0.1. Exclusive to \code{norm = "sv"}.
#' @param verbose Logical indicating whether to print progress messages. 
#'   Default is \code{TRUE}. Exclusive to \code{norm = "sv"}.
#' @param crit Character string specifying the stopping criterion. 
#'   Default is \code{"risk"}. Exclusive to \code{norm = "sv"}.
#' @param center Logical indicating whether to center the basis functions before 
#'   processing. Default is \code{TRUE}.
#' @param approx Logical indicating whether to use approximate eigendecomposition 
#'   (power iteration) instead of exact eigendecomposition. Only applies when 
#'   \code{norm = "1"} or \code{norm = "2"}. Default is \code{FALSE}.
#'
#' @return A list containing:
#' \describe{
#'   \item{\code{alpha}}{Coefficient vector on the npcs-dimensional principal component basis}
#'   \item{\code{predictions}}{Matrix of predictions (if \code{predict} is provided)}
#' }
#'
#' @details
#' The function fits an HAPC model using the specified norm constraint. When 
#' \code{norm = "sv"}, the sectional variation norm constraint is applied, with 
#' optional approximate eigendecomposition via power iteration for large datasets.
#'
#' Note on the alpha coefficients: The sign of individual alpha coefficients may 
#' vary across different computations due to the nature of SVD algorithms, and 
#' the sign is not guaranteed to be consistent for every coefficient. However, 
#' this effect vanishes when computing predictions via \eqn{U D \alpha}, since 
#' the columns of U have consistent sign changes that cancel out these variations.
#'
#' @examples
#' \dontrun{
#' # Define a test function
#' f <- function(X, n) {
#'   sin(pi * (X[, 1] * X[, 3])) / X[, 1] + 
#'   sqrt(X[, 2]) * log(X[, 3]) + 
#'   rnorm(n, 0, 0.05)
#' }
#' 
#' # Generate training data
#' n <- 50
#' d <- 3
#' X <- matrix(runif(n * d, 0.1, 1), ncol = d)
#' Y <- f(X, n)
#' 
#' # Generate test data
#' nnew <- 100
#' Xnew <- matrix(runif(nnew * d, 0.1, 1), ncol = d)
#' 
#' # Fit HAPC model with L2 norm constraint
#' fit <- hapc(X, Y,
#'             npcs = n,
#'             lambda = 0.5,
#'             norm = "2",
#'             max_degree = 2,
#'             predict = Xnew,
#'             center = FALSE)
#' 
#' # Extract predictions
#' predictions <- fit$predictions
#' }
#'
#' @export
hapc <- function(X, Y, family='gaussian',
                    max_degree = 1,
                    npcs = nrow(X), lambda=0.01,
                    norm = c("sv", "1", "2"),
                    predict = NULL,
                    max_iter=100,
                    tol=1e-9,
                    step_factor=0.1,
                    verbose=TRUE,
                    crit="risk",
                    center=TRUE,
                    approx=FALSE,
                    single_lambda= NULL) {
  norm <- match.arg(norm)
  p <- ncol(X)

  # --- ensure numeric types ---
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  Y <- as.numeric(Y)

  # ensure numeric scalars too
  max_degree <- as.integer(max_degree)
  npcs <- as.integer(npcs)
  lambda <- as.numeric(lambda)
  max_iter <- as.integer(max_iter)
  tol <- as.numeric(tol)
  step_factor <- as.numeric(step_factor)
  verbose <- as.logical(verbose)
  center <- as.logical(center)
  approx <- as.logical(approx)
  
  # Convert predict to matrix if not NULL
  if (!is.null(predict)) {
    predict <- matrix(predict, ncol=p)
  }

  if (family == 'binomial') {
    message("Binomial family")
    
    res <- .Call("pchal_cv_classi_call",
          X, Y,
          max_degree, npcs,
          as.numeric(lambda), as.integer(1),
          max_iter, tol,
          step_factor, verbose, as.character(crit),
          if (is.null(predict)) NULL else predict, center, as.numeric(lambda), PACKAGE = "hapc")
    
    return(res)
  }

  if (norm == "sv") {
    message("Sectional variation norm constraint")
    res <- .Call("single_pcghal_call",
          X, Y,
          max_degree, npcs,
          lambda, 
          max_iter, tol,
          step_factor, verbose, as.character(crit),
          if (is.null(predict)) NULL else predict, center, PACKAGE = "hapc")
  } else if (norm == "2") {
    message(paste0("L", norm, " norm constraint"))
    res <- .Call("single_lambda_pchar",
          X, Y,
           npcs,
          lambda, if (is.null(predict)) NULL else predict, max_degree,
          
           center,approx,as.logical(0), PACKAGE = "hapc")
  } else if (norm == "1") {
    message(paste0("L", norm, " norm constraint"))
    res <- .Call("single_lambda_pchar",
          X, Y,
           npcs,
          lambda, if (is.null(predict)) NULL else predict, max_degree,
          
           center,approx,as.logical(1), PACKAGE = "hapc")
  } else {
    stop("Unknown norm type, try the cv routine")
  }

  res
}

