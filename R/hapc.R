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
                    approx=FALSE) {
  norm <- match.arg(norm)
       p=ncol(X)

  if(family == 'binomial') {
    message("Binomial family")
    
  }
  # --- ensure numeric types ---
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  Y <- as.numeric(Y)

  # ensure numeric scalars too
  max_degree <- as.numeric(max_degree)
  npcs <- as.numeric(npcs)
    predict <- matrix(predict,ncol=p)
    print(dim(predict))
    p=ncol(X)


  if (norm == "sv") {
    message("Sectional variation norm constraint")
  res=.Call("single_pcghal_call",
        as.matrix(X), as.numeric(Y),
        as.integer(max_degree), as.integer(npcs),
        as.double(lambda), 
        as.integer(max_iter), as.numeric(tol),
        as.numeric(step_factor), as.logical(verbose),as.character(crit),matrix(predict,ncol=p), as.logical(center), PACKAGE = "hapc")
  } else if (norm == "2") {
    message(paste0("L", norm, " norm constraint"))
    res=.Call("single_lambda_pchar",
          as.matrix(X), as.numeric(Y),
           as.integer(npcs),
          as.double(lambda), matrix(predict,ncol=p),as.integer(max_degree),
          as.logical(center), as.logical(approx), PACKAGE = "hapc")
  }
  else {
    stop("Unknown norm type, try the cv routine")
  }

  res
}

