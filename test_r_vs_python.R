setwd("~/Projects/hapc")
library(hapc)

set.seed(42)
n <- 100
p <- 1
X <- matrix(runif(n*p, -1, 1), n, p)
Y <- 2*sin(8*pi*(X[,1]^2))/X[,1] + 10 + rnorm(n, 0, 2)

Xnew <- seq(-1, 1, length.out=10)

rescv <- cv.hapc(X, Y,
                 npcs = n,
                 log_lambda_min = -3,
                 log_lambda_max = 0,
                 norm = "sv",
                 predict = Xnew,
                 center = TRUE,
                 max_iter = 10,
                 nfolds = 3
)

print("R Predictions:")
print(rescv$predictions)
print("R Best Lambda:")
print(rescv$best_lambda)
print("R Res Opt Structure:")
str(rescv$res_opt)
