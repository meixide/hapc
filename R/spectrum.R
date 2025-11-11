#' @export

set.seed(1)
n  <- 50

X  <- matrix(sort(runif(n), decreasing = FALSE), ncol = 1)

library(hapc)

des <- design.hapc(X, npcs = n-1, center = FALSE)
H   <- des$H
U   <- des$U
V   <- des$V
D   <- des$d

# theoretical eigenvector form
u_m_j <- function(j, m, n) {
  (-1)^(m+1) * sqrt(2/(n+1)) * sin(((2*m - 1) * j * pi) / (2*(n+1)))
}



f_m <- function(m, n) {
  1 / (2 * sin((2 * m - 1) * pi / (4 * n + 2)))
}

# careful with sqrt!!!!!

theo_eigens=sapply(1:n,function(k) f_m(k,n))



png("/Users/cgmeixide/Dropbox/Aplicaciones/Overleaf/PCHA (PIKACHU)/eigenvectors_2x3.png", width = 15, height = 12, units = "cm", res = 1000)

par(mfrow = c(2, 3), mar = c(3, 3, 2, 1))
for (i in 1:5) {
  
  # theoretical vector
  vj <- sapply(1:n, function(j) u_m_j(j, i, n))
  
  # plot empirical eigenvector
  plot(U[,i],
        col="black",
       xlab="Observation index", ylab="U",
       main = paste("U", i))
  
  # overlay theoretical form
  lines(vj, col="red", lwd=2)
  
 
  
 
}
plot(log(D), main='Log singular values',ylab='log(sqrt(D))',xlab='Eigenvalue index')
lines(log(theo_eigens), col="red", lwd=2)

dev.off()


