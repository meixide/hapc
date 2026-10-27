library(hapc)
set.seed(77);X<-matrix(runif(100),50,2);Y<-sin(6*X[,1])+rnorm(50,sd=.1)
des<-design.hapc(X,max_degree=1,npcs=12,center=TRUE)
Z<-sweep(des$U,2,des$d,'*');V<-des$V
for(ini in c('1','2')) {
 f<-hapc(X,Y,max_degree=1,npcs=12,norm='sv',ini=ini,lambda=.01,predict=X,tol=1e-8)
 h<-f$alphaiters; rr<-rowSums(abs(h%*%t(V)))
 stopifnot(all(rr<=rr[1]+1e-11),isTRUE(attr(f,"sv_converged")))
 risks<-rowMeans((matrix(Y-mean(Y),nrow(h),length(Y),byrow=TRUE)-h%*%t(Z))^2)
 stopifnot(all(diff(risks)<=1e-12),max(abs(f$predictions-(Z%*%f$alpha+mean(Y))))<1e-9)
 stopifnot(!is.null(attr(f,'sv_status')))
}
a<-c(1,1);rot<-matrix(c(1,1,-1,1),2,2)/sqrt(2)
for(logistic in c(FALSE,TRUE)) {
 yy<-if(logistic) c(-1.,1.) else c(10.,0.)
 args<-list(yy,diag(2),rot,a,300L,1e-10,.8,FALSE)
 if(!logistic)args<-c(args,list('grad'))
 out<-do.call('.Call',c(list(if(logistic)'pcghal_classi_call' else 'pcghal_call'),args,list(PACKAGE='hapc')))
 norms<-rowSums(abs(out$alphaiters%*%t(rot)))
 stopifnot(all(norms<=norms[1]+1e-11),isTRUE(attr(out,"sv_converged")))
}
cat('R SV checks passed: native gaussian/logistic, public gaussian paths, predictions, diagnostics\n')

for (factor in c(0,1,NA_real_)) {
  bad <- try(hapc(X,Y,max_degree=1,npcs=12,norm="sv",step_factor=factor),silent=TRUE)
  stopifnot(inherits(bad,"try-error"))
}
# Public binomial output remains finite and calibrated after the shared
# safeguarded intercept solve.
yb<-as.numeric(Y>median(Y))
fb<-hapc(X,yb,family="binomial",norm="sv",max_degree=1,npcs=10,lambda=.01,predict=X)
stopifnot(all(is.finite(fb$predictions)),abs(mean(fb$predictions)-mean(yb))<1e-9)
cat("R public SV validation and binomial checks passed\n")

# The old locked-zero example is solved on the ball.
ball<-.Call("pcghal_call",c(0.,2.),diag(2),diag(2),c(1.,0.),1000L,1e-10,.8,FALSE,"grad",PACKAGE="hapc")
stopifnot(max(abs(ball$alpha-c(0,1)))<1e-7,isTRUE(attr(ball,"sv_converged")),attr(ball,"sv_optimality_gap")<1e-8)
# Radius can be slack at the unconstrained optimum.
inside<-.Call("pcghal_call",c(.2,-.1),diag(2),diag(2),c(2.,2.),5000L,1e-9,.8,FALSE,"risk",PACKAGE="hapc")
stopifnot(sum(abs(inside$beta))<4,max(abs(inside$alpha-c(.2,-.1)))<1e-5)
for (family in c("gaussian","binomial")) {
 yy<-if(family=="gaussian") Y else yb
 cv<-cv.hapc(X,yy,max_degree=1,npcs=8,norm="sv",family=family,nfolds=3,
             log_lambda_min=log(.01),log_lambda_max=log(.1),grid_length=2,tol=1e-6)
 stopifnot(length(attr(cv,"sv_fold_status"))==6,all(attr(cv,"sv_fold_status")=="optimal"))
 if(family=="gaussian")stopifnot(isTRUE(attr(cv$res_opt,"sv_converged")))
 else stopifnot(isTRUE(attr(cv,"sv_converged")))
}
cat("R SV-ball certificates, zero activation, interior solution and CV diagnostics passed\n")
