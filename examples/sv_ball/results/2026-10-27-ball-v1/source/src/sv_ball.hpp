#ifndef HAPC_SV_BALL_HPP
#define HAPC_SV_BALL_HPP
#include "sv_descent.hpp"
#include "logistic_intercept.hpp"
#include <functional>

// Euclidean projection, NOT soft thresholding at a fixed penalty.
inline VectorXd project_sv_ball(const VectorXd& x, double C) {
    if (C==0) return VectorXd::Zero(x.size());
    if (x.lpNorm<1>()<=C) return x;
    std::vector<double> sorted(x.size());
    for (int i=0;i<x.size();++i) sorted[i]=std::abs(x[i]);
    std::sort(sorted.begin(),sorted.end(),std::greater<double>());
    double sum=0, threshold=0;
    for (int i=0;i<x.size();++i) {
        sum+=sorted[i];
        const double t=(sum-C)/(i+1);
        if (i+1==x.size() || t>=sorted[i+1]) { threshold=t; break; }
    }
    return x.unaryExpr([threshold](double z) {
        return std::copysign(std::max(0.0,std::abs(z)-threshold),z);
    });
}

struct SVLoss {
    const VectorXd& y;
    const MatrixXd& Z;
    bool logistic;
    MatrixXd quadratic;
    VectorXd linear, y01;
    SVLoss(const VectorXd& response,const MatrixXd& scores,bool binary)
        :y(response),Z(scores),logistic(binary) {
        if (logistic) y01=(y.array()+1)/2;
        else { quadratic=2.0*Z.transpose()*Z/y.size(); linear=2.0*Z.transpose()*y/y.size(); }
    }
    VectorXd probabilities(const VectorXd& a) const {
        VectorXd eta=Z*a;
        eta.array()+=profile_logistic_intercept(y01,eta);
        return eta.unaryExpr([](double x){return hapc_expit(x);});
    }
    double risk(const VectorXd& a) const {
        if (!logistic) return (y-Z*a).squaredNorm()/y.size();
        VectorXd eta=Z*a;
        eta.array()+=profile_logistic_intercept(y01,eta);
        double sum=0;
        for(int i=0;i<y.size();++i) {
            const double t=-y[i]*eta[i];
            sum+=std::max(0.0,t)+std::log1p(std::exp(-std::abs(t)));
        }
        return sum/y.size();
    }
    VectorXd gradient(const VectorXd& a) const {
        if (!logistic) return quadratic*a-linear;
        return Z.transpose()*(probabilities(a)-y01)/y.size();
    }
    MatrixXd hessian(const VectorXd& a) const {
        if (!logistic) return quadratic;
        const VectorXd p=probabilities(a), w=p.array()*(1-p.array());
        MatrixXd H=Z.transpose()*w.asDiagonal()*Z;
        const double s=w.sum();
        if(s>0) { const VectorXd v=Z.transpose()*w; H-=v*v.transpose()/s; }
        return H/y.size();
    }
    // Dual of the logistic loss: choose probabilities with exactly the target
    // total (up to summation roundoff), independent of intercept root accuracy.
    VectorXd dual_gradient(const VectorXd& a, double& conjugate_bound, double& intercept_residual) const {
        if(!logistic) {
            const VectorXd w=2.0*(Z*a-y)/y.size();
            conjugate_bound=-y.dot(w)-y.size()*w.squaredNorm()/4;
            intercept_residual=0;
            return Z.transpose()*w;
        }
        VectorXd p=probabilities(a);
        if(y01.sum()==0 || y01.sum()==y01.size()) p=y01;
        for(int it=0;it<4;++it) {
            const double remainder=y01.sum()-p.sum();
            if(remainder==0) break;
            VectorXd capacity=remainder>0 ? VectorXd::Ones(p.size())-p : p;
            if(capacity.sum()==0) break;
            p+=remainder*capacity/capacity.sum();
            p=p.cwiseMax(0).cwiseMin(1);
        }
        double entropy=0;
        for(int i=0;i<p.size();++i) {
            if(p[i]>0) entropy-=p[i]*std::log(p[i]);
            if(p[i]<1) entropy-=(1-p[i])*std::log1p(-p[i]);
        }
        conjugate_bound=entropy/y.size();
        intercept_residual=std::abs((p-y01).sum()/y.size());
        return Z.transpose()*(p-y01)/y.size();
    }
};

// ADMM splitting: b=V alpha, b in the L1 ball. Gaussian alpha updates solve
// one cached quadratic system; logistic updates use safeguarded Newton on the
// profiled intercept objective. The reported history contains feasible BEST
// incumbents, not the possibly infeasible raw ADMM alpha iterates.
inline OptimizerOutput sv_ball_fit(const VectorXd& y,const MatrixXd& Z,
                                  const MatrixXd& V,const VectorXd& initial,
                                  int max_iter,double tol,double step_factor,
                                  bool verbose,const std::string& crit,bool logistic) {
    validate_sv_data(y,Z,V,initial);
    if(max_iter<0 || !std::isfinite(tol) || tol<=0 ||
       !std::isfinite(step_factor) || step_factor<=0 || step_factor>=1 ||
       (crit!="grad" && crit!="risk"))
        throw std::runtime_error("Invalid SV controls: require max_iter>=0, tol>0, 0<step_factor<1, valid crit");
    SVLoss loss(y,Z,logistic);
    const int k=Z.cols();
    const double C=(V*initial).lpNorm<1>();
    if(!std::isfinite(C)) throw std::runtime_error("Non-finite initial SV radius");
    VectorXd a=initial, best=initial, b=V*a, u=VectorXd::Zero(V.rows());
    double best_risk=loss.risk(best);
    if(!std::isfinite(best_risk)) throw std::runtime_error("Non-finite initial risk");
    MatrixXd history(max_iter+1,k);history.row(0)=best.transpose();
    OptimizerOutput out;
    out.sv_radius=C;out.constraint="ball";out.status="max_iter";
    out.tangent_norm=std::numeric_limits<double>::quiet_NaN();
    const MatrixXd gram=V.transpose()*V;
    Eigen::CompleteOrthogonalDecomposition<MatrixXd> gram_solver(gram);
    Eigen::SelfAdjointEigenSolver<MatrixXd> spectrum(gram);
    const double sigma_min=std::sqrt(std::max(0.0,spectrum.eigenvalues().minCoeff()));
    const bool full_rank=gram_solver.rank()==k && sigma_min>0;
    const double alpha_bound=full_rank ? C/sigma_min : 0;
    double intercept_bound=0;
    if(logistic && loss.y01.mean()>0 && loss.y01.mean()<1) {
        const double prevalence=loss.y01.mean();
        intercept_bound=std::abs(std::log(prevalence/(1-prevalence)))+
            alpha_bound*Z.rowwise().norm().maxCoeff();
    }
    // C=0 is a singleton only when V has full column rank. Otherwise retain
    // the general ADMM path; a null-V direction need not be a null-Z direction.
    double rho=std::max(1e-6,loss.hessian(a).trace()/std::max(1e-12,gram.trace()));
    rho=std::min(1e6,rho);
    Eigen::LDLT<MatrixXd> system;
    double factored_rho=-1;
    double best_dual_bound=-std::numeric_limits<double>::infinity();
    double best_dual_residual=std::numeric_limits<double>::infinity();
    auto certificate=[&](const VectorXd& dual) {
        double bound=0,intercept=0;
        const VectorXd g=loss.dual_gradient(best,bound,intercept);
        VectorXd corrected=dual-V*gram_solver.solve(g+V.transpose()*dual);
        const VectorXd stationarity=g+V.transpose()*corrected;
        const double dual_residual=std::max(intercept,stationarity.lpNorm<Eigen::Infinity>());
        // Account explicitly for floating-point dual feasibility error:
        // ||alpha||2<=C/sigma_min(V), |b*(alpha)|<=intercept_bound.
        const double dual_bound=bound-C*corrected.lpNorm<Eigen::Infinity>()-
            alpha_bound*stationarity.norm()-intercept_bound*intercept;
        if(!full_rank) {
            out.dual_residual=dual_residual;
            out.dual_bound=-std::numeric_limits<double>::infinity();
            out.optimality_gap=std::numeric_limits<double>::infinity();
            out.converged=false;
            return;
        }
        if(std::isfinite(dual_bound) && dual_bound>best_dual_bound) {
            best_dual_bound=dual_bound;best_dual_residual=dual_residual;
        }
        out.dual_bound=best_dual_bound;out.dual_residual=best_dual_residual;
        const double raw_gap=best_risk-out.dual_bound;
        out.optimality_gap=std::max(0.0,raw_gap);
        out.constraint_violation=std::max(0.0,(V*best).lpNorm<1>()-C);
        const double roundoff=256*std::numeric_limits<double>::epsilon()*(1+std::abs(best_risk)+std::abs(out.dual_bound));
        // A deficient V cannot turn a small stationarity residual into a
        // rigorous lower bound in uncontrolled directions. Do not certify it.
        out.converged=full_rank && std::isfinite(raw_gap) && raw_gap>=-roundoff &&
            out.optimality_gap<=tol*(1+std::abs(best_risk)) &&
            out.dual_residual<=1e-10*(1+g.lpNorm<Eigen::Infinity>()) &&
            out.constraint_violation<=roundoff;
    };
    auto consider=[&](const VectorXd& candidate,const VectorXd& dual) {
        const double candidate_value=loss.risk(candidate);
        certificate(dual);
        const double objective_roundoff=16*std::numeric_limits<double>::epsilon()*(1+std::abs(best_risk));
        if(std::isfinite(candidate_value) && candidate_value<=best_risk+objective_roundoff) {
            const VectorXd previous_best=best;
            const double previous_risk=best_risk;
            const OptimizerOutput previous_diagnostics=out;
            best=candidate;best_risk=candidate_value;
            certificate(dual);
            // Risk differences below rounding cannot rank nearly optimal
            // iterates. Keep the tighter certificate instead of freezing at
            // an O(sqrt(eps)) coefficient error with an O(sqrt(eps)) gap.
            if(candidate_value>=previous_risk && out.optimality_gap>=previous_diagnostics.optimality_gap) {
                best=previous_best;best_risk=previous_risk;out=previous_diagnostics;certificate(dual);
            }
        }
    };
    // A sparse optimum gives a useful dual candidate without dividing by
    // tiny barrier slacks. Solve its active-face stationarity equations. Any
    // candidate remains subject to the same independent dual-gap certificate;
    // an incorrect support guess cannot manufacture convergence.
    auto polish_dual=[&]() {
        if(!full_rank || C==0 || (V*best).lpNorm<1>()<.9*C) return;
        const VectorXd beta=V*best,g=loss.gradient(best);
        for(double relative:{1e-4,1e-6,1e-8}) {
            std::vector<int> zeros;
            VectorXd active=VectorXd::Zero(k);
            for(int j=0;j<V.rows();++j) {
                if(std::abs(beta[j])<=relative*C) zeros.push_back(j);
                else active+=std::copysign(1.0,beta[j])*V.row(j).transpose();
            }
            MatrixXd equations(k,zeros.size()+1);equations.col(0)=active;
            for(int j=0;j<(int)zeros.size();++j) equations.col(j+1)=V.row(zeros[j]).transpose();
            const VectorXd multipliers=equations.completeOrthogonalDecomposition().solve(-g);
            VectorXd dual=beta.unaryExpr([&](double z){return std::copysign(multipliers[0],z);});
            for(int j=0;j<(int)zeros.size();++j) dual[zeros[j]]=multipliers[j+1];
            if(dual.allFinite()) certificate(dual);
        }
    };
    certificate(VectorXd::Zero(V.rows()));
    int done=0;
    if(out.converged) out.status="optimal";
    for(int it=0;it<std::min(max_iter,500) && !out.converged;++it) {
        const VectorXd target=b-u;
        if(!logistic) {
            if(factored_rho!=rho) { system.compute(loss.quadratic+rho*gram);factored_rho=rho; }
            const VectorXd rhs=loss.linear+rho*V.transpose()*target;
            a=system.solve(rhs);
            if(!a.allFinite()) a=(loss.quadratic+rho*gram).completeOrthogonalDecomposition().solve(rhs);
        } else {
            for(int inner=0;inner<60;++inner) {
                const VectorXd g=loss.gradient(a)+rho*V.transpose()*(V*a-target);
                if(g.lpNorm<Eigen::Infinity>()<1e-11*(1+rho*(V.transpose()*target).norm())) break;
                const MatrixXd H=loss.hessian(a)+rho*gram;
                VectorXd direction=H.ldlt().solve(-g);
                if(!direction.allFinite() || g.dot(direction)>=0) direction=-g/std::max(1.0,H.norm());
                // Near the Newton solution, objective differences round to
                // zero long before the gradient is accurate enough for the
                // certificate. A tiny predictor displacement is in the local
                // quadratic regime; take that Newton correction directly.
                if((Z*direction).lpNorm<Eigen::Infinity>()<1e-7 && direction.norm()<1e-7) {
                    a+=direction;
                    continue;
                }
                const double old=loss.risk(a)+rho*(V*a-target).squaredNorm()/2;
                double step=1;bool accepted=false;
                for(int bt=0;bt<50;++bt,step*=.5) {
                    const VectorXd candidate=a+step*direction;
                    const double value=loss.risk(candidate)+rho*(V*candidate-target).squaredNorm()/2;
                    if(std::isfinite(value) && value<=old+1e-4*step*g.dot(direction)) { a=candidate;accepted=true;break; }
                }
                if(!accepted) break;
            }
        }
        if(!a.allFinite()) {out.status="numerical_failure";break;}
        const VectorXd Va=V*a, previous_b=b;
        b=project_sv_ball(Va+u,C);
        u+=Va-b;
        out.primal_residual=(Va-b).norm();
        VectorXd feasible=a;
        const double candidate_C=(V*feasible).lpNorm<1>();
        if(candidate_C>C) feasible*=C/candidate_C; // inequality: never enlarge an interior fit
        consider(feasible,rho*u);
        history.row(++done)=best.transpose();
        if(done%25==0) polish_dual();
        if(out.converged) out.status="optimal";
        if(verbose && (done%25==0 || out.converged))
            std::cout<<"SV ball iter="<<done<<" risk="<<best_risk<<" gap="<<out.optimality_gap<<std::endl;
        if(done%25==0) {
            const double primal=(Va-b).norm(),dual=(rho*V.transpose()*(b-previous_b)).norm();
            if(primal>10*dual && rho<1e8) {rho*=2;u/=2;}
            else if(dual>10*primal && rho>1e-8) {rho/=2;u*=2;}
        }
    }
    // ADMM is inexpensive but can approach a sparse optimum very slowly.
    // Polish that SAME convex problem with a logarithmic epigraph barrier.
    // Eliminate the q slack variables analytically: only a k x k solve remains.
    // Convergence is still decided solely by the independent feasible/dual gap.
    if(!out.converged && full_rank && C>0 && done<max_iter) {
        a=best;
        const double norm=(V*a).lpNorm<1>();
        if(norm>.95*C) a*=.95*C/norm;
        VectorXd t=(V*a).cwiseAbs();
        t.array()+=(C-t.sum())/(2*V.rows());
        double mu=.01*(1+std::abs(best_risk))/(2*V.rows()+1);
        auto barrier=[&](const VectorXd& x,const VectorXd& slack,double parameter) {
            const VectorXd v=V*x;
            if((slack-v).minCoeff()<=0 || (slack+v).minCoeff()<=0 || C-slack.sum()<=0)
                return std::numeric_limits<double>::infinity();
            return loss.risk(x)-parameter*((slack-v).array().log().sum()+
                (slack+v).array().log().sum()+std::log(C-slack.sum()));
        };
        for(int stage=0;stage<25 && !out.converged && done<max_iter;++stage,mu*=.2) {
            for(int inner=0;inner<80 && !out.converged && done<max_iter;++inner) {
                const VectorXd v=V*a,minus=t-v,plus=t+v;
                const double slack=C-t.sum();
                const VectorXd inv_minus=minus.cwiseInverse(),inv_plus=plus.cwiseInverse();
                const VectorXd d=inv_minus.array().square()+inv_plus.array().square();
                const VectorXd mixed=(inv_plus.array().square()-inv_minus.array().square())/d.array();
                const VectorXd dinv=d.cwiseInverse();
                const double denominator=slack*slack+dinv.sum();
                const VectorXd ga=loss.gradient(a)+mu*V.transpose()*(inv_minus-inv_plus);
                const VectorXd gt=mu*(-inv_minus.array()-inv_plus.array()+1/slack).matrix();
                const VectorXd r=V.transpose()*mixed;
                const VectorXd stable_diagonal=4.0/(minus.array().square()+plus.array().square());
                MatrixXd H=loss.hessian(a)+mu*V.transpose()*stable_diagonal.asDiagonal()*V+
                    (mu/denominator)*r*r.transpose();
                const VectorXd rhs=-ga+V.transpose()*(mixed.array()*gt.array()-
                    mixed.array()*(dinv.dot(gt)/denominator)).matrix();
                const VectorXd da=H.ldlt().solve(rhs);
                const VectorXd Va=V*da;
                const VectorXd dt=-(dinv.array()*gt.array()/mu+mixed.array()*Va.array()-
                    dinv.array()*((dinv.dot(gt)/mu+mixed.dot(Va))/denominator)).matrix();
                if(!da.allFinite() || !dt.allFinite()) break;
                const double derivative=ga.dot(da)+gt.dot(dt);
                if(-derivative<.1*mu) break;
                if(derivative>=0) break;
                const double old=barrier(a,t,mu);
                double step=1;bool accepted=false;
                for(int bt=0;bt<60;++bt,step*=.5) {
                    const double value=barrier(a+step*da,t+step*dt,mu);
                    if(std::isfinite(value) && value<=old+1e-4*step*derivative) {accepted=true;break;}
                }
                if(!accepted) break;
                a+=step*da;t+=step*dt;
                const VectorXd newv=V*a;
                const VectorXd dual=mu*((t-newv).cwiseInverse()-(t+newv).cwiseInverse());
                consider(a,dual);
                out.primal_residual=0;
                polish_dual();
                history.row(++done)=best.transpose();
                if(out.converged) out.status="optimal";
            }
            if(verbose) std::cout<<"Barrier mu="<<mu<<" risk="<<best_risk<<" gap="<<out.optimality_gap<<std::endl;
        }
        if(!out.converged && done<max_iter) out.status="polish_stalled";
    }
    out.alpha=best;out.beta=V*best;out.risk=best_risk;
    out.alphaiters=history.topRows(done+1);out.iter=done;
    return out;
}
#endif
