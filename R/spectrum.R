#' Demo: spectrum plot (does not run at package load)
#' @export
spectrum_demo <- function(outfile = NULL) {
  set.seed(1)
  n  <- 50
  X  <- matrix(sort(runif(n), decreasing = FALSE), ncol = 1)
  des <- design.hapc(X, npcs = n-1, center = FALSE)
  H   <- des$H; U <- des$U; V <- des$V; D <- des$d

  u_m_j <- function(j, m, n) {
    (-1)^(m+1) * sqrt(2/(n+1)) * sin(((2*m - 1) * j * pi) / (2*(n+1)))
  }
  f_m <- function(m, n) 1 / (2 * sin((2 * m - 1) * pi / (4 * n + 2)))
  theo_eigens <- sapply(1:n, function(k) f_m(k, n))

  if (!is.null(outfile)) png(outfile, width = 15, height = 12, units = "cm", res = 1000)
  par(mfrow = c(2, 3), mar = c(3, 3, 2, 1))
  for (i in 1:5) {
    vj <- sapply(1:n, function(j) u_m_j(j, i, n))
    plot(U[, i], col = "black", xlab = "Observation index", ylab = "U", main = paste("U", i))
    lines(vj, col = "red", lwd = 2)
  }
  plot(log(D), main = "Log singular values", ylab = "log(sqrt(D))", xlab = "Eigenvalue index")
  lines(log(theo_eigens), col = "blue", lwd = 2)
  if (!is.null(outfile)) dev.off()
  invisible(list(H = H, U = U, d = D, V = V))
}