#ifndef HAPC_LOGISTIC_INTERCEPT_HPP
#define HAPC_LOGISTIC_INTERCEPT_HPP
#include "hapc_core.hpp"
#include <cmath>
#include <stdexcept>
inline double hapc_expit(double x) {
    if (x>=0) return 1.0/(1.0+std::exp(-x));
    const double e=std::exp(x); return e/(1+e);
}
inline double profile_logistic_intercept(const VectorXd& y, const VectorXd& eta) {
    if (y.size()!=eta.size() || y.size()==0) throw std::runtime_error("Intercept dimensions mismatch");
    const double prevalence=y.mean();
    // A one-class sample has no finite MLE. Return a finite limiting offset.
    if (prevalence<=0) return -40.0-eta.maxCoeff();
    if (prevalence>=1) return 40.0-eta.minCoeff();
    const double logit=std::log(prevalence/(1-prevalence));
    double lo=logit-eta.maxCoeff(), hi=logit-eta.minCoeff(), b=lo/2+hi/2;
    for (int it=0; it<120; ++it) {
        double g=0,h=0;
        for (int i=0;i<y.size();++i) { double p=hapc_expit(eta[i]+b); g+=p-y[i]; h+=p*(1-p); }
        if (std::abs(g)<1e-12*y.size()) break;
        if (g>0) hi=b; else lo=b;
        const double next=h>0 ? b-g/h : lo/2+hi/2;
        b=std::isfinite(next) && next>lo && next<hi ? next : lo/2+hi/2;
    }
    return b;
}
#endif
