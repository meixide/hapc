library(testthat)
library(hapc)
# Match the existing R test harness on installations with lazy waldo loading.
for (ns in c("glue", "waldo")) invisible(loadNamespace(ns))

fixture <- function(n=100L, binary=TRUE, seed=11) {
  set.seed(seed)
  X <- cbind(runif(n),runif(n))
  A <- rep(0:1, length.out=n)
  eta <- -.6+.5*A+.7*X[,1]
  Y <- if (binary) rbinom(n,1,plogis(eta)) else .7*A+X[,1]+rnorm(n)
  Q <- cbind(plogis(-.6+.7*X[,1]),plogis(-.1+.7*X[,1]))
  list(X=X,A=A,Y=Y,Q=Q)
}

test_that("binary targeting matches independent offset GLM and solves both scores", {
  d <- fixture()
  g <- seq(.25,.75,length.out=nrow(d$X))
  f <- hapc.ate.tmle(d$X,d$Y,d$A,Q=d$Q,g=g,cf_folds=2)
  H0 <- (1-d$A)/(1-g); H1 <- d$A/g
  offset <- qlogis(d$Q[cbind(seq_along(d$Y),d$A+1)])
  ref <- glm(d$Y ~ -1+H0+H1+offset(offset),family=binomial(),
             control=glm.control(epsilon=1e-12,maxit=100))
  expected <- cbind(plogis(qlogis(d$Q[,1])+coef(ref)[1]/(1-g)),
                    plogis(qlogis(d$Q[,2])+coef(ref)[2]/g))
  expect_equal(f$Q_star,unname(expected),tolerance=1e-7)
  expect_equal(f$estimate,mean(expected[,2]-expected[,1]),tolerance=1e-8)
  expect_lt(max(abs(f$diagnostics$score)),1e-8)
  expect_lt(abs(mean(f$eif)),1e-8)
  expect_true(all(f$Q_star>0 & f$Q_star<1))
  expect_equal(f$se,sd(f$eif)/sqrt(length(d$Y)))
  expect_equal(f$upper-f$estimate,qnorm(.975)*f$se)
})

test_that("gaussian fluctuation matches least squares and is scale equivariant", {
  d <- fixture(binary=FALSE)
  g <- rep(.5,length(d$Y)); Q <- cbind(d$X[,1],.5+d$X[,1])
  f <- hapc.ate.tmle(d$X,d$Y,d$A,Q=Q,g=g)
  H0 <- (1-d$A)/(1-g); H1 <- d$A/g
  ref <- lm(I(d$Y-Q[cbind(seq_along(d$Y),d$A+1)]) ~ -1+H0+H1)
  expected <- Q+cbind(coef(ref)[1]/(1-g),coef(ref)[2]/g)
  expect_equal(f$Q_star,unname(expected),tolerance=1e-9)
  scaled <- hapc.ate.tmle(d$X,3*d$Y+8,d$A,Q=3*Q+8,g=g)
  expect_equal(scaled$estimate,3*f$estimate,tolerance=1e-9)
  expect_equal(scaled$se,3*f$se,tolerance=1e-9)
})

test_that("bounded continuous targeting preserves externally specified bounds", {
  d <- fixture()
  d$Y <- .1+.8*d$Q[,1]+.02*d$A
  f <- hapc.ate.tmle(d$X,d$Y,d$A,Q=d$Q,g=.5,outcome_bounds=c(0,1))
  scaled <- hapc.ate.tmle(d$X,10+4*d$Y,d$A,Q=10+4*d$Q,g=.5,
                         outcome_bounds=c(10,14))
  expect_equal(scaled$estimate,4*f$estimate,tolerance=1e-8)
  expect_equal(scaled$se,4*f$se,tolerance=1e-8)
  expect_true(all(scaled$Q_star>=10 & scaled$Q_star<=14))
})

test_that("inputs and non-finite logistic targeting fail explicitly", {
  d <- fixture()
  expect_error(hapc.ate.tmle(d$X,replace(d$Y,1,NA),d$A),"finite")
  expect_error(hapc.ate.tmle(d$X,d$Y,rep(0,100)),"both treatment")
  expect_error(hapc.ate.tmle(d$X,d$Y,d$A,g=0,Q=d$Q),"strictly")
  expect_error(hapc.ate.tmle(d$X,d$Y,d$A,cf_folds=51),"fold")
  expect_error(hapc.ate.tmle(d$X,d$Y,d$A,norm="sv"),"arg")
  y <- d$Y; y[d$A==0] <- 0
  expect_error(hapc.ate.tmle(d$X,y,d$A,Q=d$Q,g=.5),"finite logistic")
  f <- hapc.ate.tmle(d$X,d$Y,d$A,Q=d$Q,g=.001)
  expect_equal(f$diagnostics$g_clipped_fraction,1)
})

test_that("native HAPC fits work, preserve RNG, and do not train on held-out outcomes", {
  d <- fixture(n=100L,binary=FALSE)
  args <- list(X=d$X,Y=d$Y,A=d$A,g=.5,cf_folds=2,seed=5,
               nfolds=2,grid_length=3,npcs=20,norm="1")
  set.seed(7); before <- .Random.seed
  f <- do.call(hapc.ate.tmle,args)
  expect_identical(.Random.seed,before)
  heldout <- which(f$folds==0)
  args$Y[heldout] <- args$Y[heldout]+3
  other <- do.call(hapc.ate.tmle,args)
  expect_equal(f$Q_initial[heldout,],other$Q_initial[heldout,],tolerance=1e-10)
  expect_true(is.finite(f$se))
  # Fit both binomial nuisance models using their native probability contract.
  b <- fixture(n=100L,binary=TRUE,seed=28)
  fit <- suppressWarnings(hapc.ate.tmle(b$X,b$Y,b$A,cf_folds=2,
                         nfolds=2,grid_length=3,npcs=20,norm="1"))
  expect_true(all(is.finite(fit$Q_initial)))
  expect_true(all(fit$g>0 & fit$g<1))
  expect_lt(abs(mean(fit$eif)),1e-8)
})
