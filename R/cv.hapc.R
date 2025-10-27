#' @export

cv.hapc <- function(X, Y,
                    max_degree = 1,
                    npcs = 100,
                    log_lambda_min = -5,
                    log_lambda_max = -3,
                    grid_length=10,
                    nfolds = 5,
                    norm = c("sv", "1"),
                    predict = NULL,
                    max_iter=100,
                    tol=1e-9,
                    step_factor=0.1,
                    verbose=TRUE,
                    crit="risk") {
  norm <- match.arg(norm)

  # --- ensure numeric types ---
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  Y <- as.numeric(Y)

  # ensure numeric scalars too
  max_degree <- as.numeric(max_degree)
  npcs <- as.numeric(npcs)
  log_lambda_min <- as.numeric(log_lambda_min)
  log_lambda_max <- as.numeric(log_lambda_max)
  nfolds <- as.numeric(nfolds)
    predict <- matrix(predict,ncol=p)
    print(dim(predict))
    p=ncol(X)


  if (norm == "sv") {
    message("Sectional variation norm constraint")
     p=ncol(X)
  res=.Call("pchal_cv_call",
        as.matrix(X), as.numeric(Y),
        as.integer(max_degree), as.integer(npcs),
        as.numeric(2^seq(log_lambda_min, log_lambda_max, length.out = grid_length)), as.integer(nfolds),
        as.integer(max_iter), as.numeric(tol),
        as.numeric(step_factor), as.logical(verbose),as.character(crit),matrix(predict,ncol=p),PACKAGE = "hapc")
  } else {
    message("L1 norm constraint")
    res <- .Call("fasthal_cv_call", X, Y, npcs, as.numeric(2^seq(log_lambda_min, log_lambda_max, length.out = grid_length)),nfolds,predict, PACKAGE = "hapc")
  }

  res
}

