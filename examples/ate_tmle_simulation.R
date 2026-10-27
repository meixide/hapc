# Synthetic point-treatment ATE smoke campaign. Run from any working directory:
# Rscript examples/ate_tmle_simulation.R OUTPUT_DIR [REPLICATIONS]
# Records every attempted fit and warning; no silent exclusions.
library(hapc)
args <- commandArgs(trailingOnly=TRUE)
out <- if (length(args)) args[1] else stop("Supply a new output directory.")
R <- if (length(args)>1) as.integer(args[2]) else 40L
if (dir.exists(out)) stop("Output directory exists; choose a new run.")
dir.create(out,recursive=TRUE)
binary_truth <- integrate(function(w1) vapply(w1,function(x)
  integrate(function(w2) plogis(.2+sin(2*pi*x)+.5*w2)-
              plogis(-.6+sin(2*pi*x)+.5*w2),0,1,rel.tol=1e-10)$value,
  numeric(1)),0,1,rel.tol=1e-9)$value
config <- list(reps=R, sizes=c(100L,300L), seed_start=20261023L,
               norm="1",npcs="floor(n_training^0.8)", cf_folds=3L,nfolds=3L,
               max_degree=2L,log_lambda_min=-8,log_lambda_max=2,grid_length=8L,
               g_bound=.025,binary_truth=binary_truth,gaussian_truth=.8)
saveRDS(config,file.path(out,"config.rds"))
capture.output(sessionInfo(),file=file.path(out,"sessionInfo.txt"))
raw <- list(); idx <- 0L
for (family in c("binomial","gaussian")) for (n in config$sizes) {
  truth <- if (family=="binomial") binary_truth else .8
  for (r in seq_len(R)) {
    seed <- config$seed_start+r+1000L*n+if (family=="binomial") 0L else 10000000L
    set.seed(seed)
    W <- cbind(runif(n),runif(n))
    g <- .15+.7*plogis(1.8*(W[,1]-.5)-(W[,2]-.5))
    A <- rbinom(n,1,g)
    base <- sin(2*pi*W[,1])+.5*W[,2]
    Q <- if (family=="binomial") cbind(plogis(-.6+base),plogis(.2+base)) else cbind(base,base+.8)
    Y <- if (family=="binomial") rbinom(n,1,Q[cbind(seq_len(n),A+1)]) else base+.8*A+rnorm(n,sd=.7)
    for (learner in c("hapc","oracle")) {
      warnings <- character()
      fitargs <- list(X=W,Y=Y,A=A,family=family,cf_folds=3L,seed=seed,
                      nfolds=3L,npcs=floor((n*2/3)^.8),max_degree=2L,norm="1",
                      log_lambda_min=-8,log_lambda_max=2,grid_length=8L)
      if (learner=="oracle") { fitargs$Q <- Q; fitargs$g <- g }
      fit <- tryCatch(withCallingHandlers(do.call(hapc.ate.tmle,fitargs),
        warning=function(w) { warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning") }),error=identity)
      for (method in c("plugin","one_step","tmle")) {
        idx <- idx+1L
        ok <- !inherits(fit,"error")
        est <- if (!ok) NA_real_ else switch(method,plugin=fit$plugin,one_step=fit$one_step,tmle=fit$estimate)
        se <- if (!ok || method=="plugin") NA_real_ else if (method=="tmle") fit$se else fit$one_step_se
        raw[[idx]] <- data.frame(family=family,n=n,rep=r,seed=seed,learner=learner,
          method=method,truth=truth,estimate=est,se=se,lower=est-qnorm(.975)*se,
          upper=est+qnorm(.975)*se,ok=ok,
          clipped=if(ok) fit$diagnostics$g_clipped_fraction else NA_real_,
          score=if(ok) max(abs(fit$diagnostics$score)) else NA_real_,
          warning=paste(unique(warnings),collapse=" | "),
          error=if(ok) "" else conditionMessage(fit))
      }
    }
    # Checkpoint after each seed.
    write.csv(do.call(rbind,raw),file.path(out,"replications.csv"),row.names=FALSE)
  }
  cat(family,"n=",n,"finished",R,"seeds\n"); flush.console()
}
d <- do.call(rbind,raw)
summary <- do.call(rbind,lapply(split(d,list(d$family,d$n,d$learner,d$method),drop=TRUE),function(z) {
  good <- z[z$ok,,drop=FALSE]; m <- nrow(good)
  p <- if (m && any(is.finite(good$se))) mean(good$lower<=good$truth & good$upper>=good$truth) else NA_real_
  data.frame(family=z$family[1],n=z$n[1],learner=z$learner[1],method=z$method[1],
    attempted=nrow(z),valid=m,failures=sum(!z$ok),
    bias=if(m)mean(good$estimate-good$truth) else NA_real_,
    rmse=if(m)sqrt(mean((good$estimate-good$truth)^2)) else NA_real_,
    empirical_sd=if(m>1)sd(good$estimate) else NA_real_,
    mean_se=if(m && any(is.finite(good$se)))mean(good$se) else NA_real_,
    coverage=p,coverage_mcse=if(is.finite(p))sqrt(p*(1-p)/m) else NA_real_,
    mean_width=if(m && any(is.finite(good$se)))mean(good$upper-good$lower) else NA_real_,
    warning_runs=sum(nzchar(z$warning)))
}))
# Wilson intervals quantify Monte Carlo uncertainty even at 0% or 100% coverage.
z <- qnorm(.975)
p <- summary$coverage; m <- summary$valid
ctr <- (p+z^2/(2*m))/(1+z^2/m)
half <- z*sqrt(p*(1-p)/m+z^2/(4*m^2))/(1+z^2/m)
summary$coverage_mc_lower <- ctr-half
summary$coverage_mc_upper <- ctr+half
write.csv(summary,file.path(out,"summary.csv"),row.names=FALSE)
print(summary,row.names=FALSE,digits=3)
