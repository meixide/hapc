#include "sv_ball.hpp"

#include "logistic_intercept.hpp"

OptimizerOutput pcghal_boundary_classi_call(const VectorXd& Y, const MatrixXd& Xtilde,
                                  const MatrixXd& ENn, const VectorXd& alpha0,
                                  int max_iter, double tol, double step_factor,
                                  bool verbose) {
    validate_sv_data(Y, Xtilde, ENn, alpha0);
    if (!((Y.array()==1.0) || (Y.array()==-1.0)).all())
        throw std::runtime_error("Logistic SV solver requires labels -1 or +1");
    const VectorXd y01=(Y.array()+1.0)/2.0;
    auto profile = [&](const VectorXd& alpha) -> VectorXd {
        VectorXd offset=Xtilde*alpha;
        offset.array() += profile_logistic_intercept(y01, offset);
        return offset;
    };
    auto risk = [&](const VectorXd& alpha) {
        const VectorXd eta=profile(alpha);
        double sum=0;
        for (int i=0; i<Y.size(); ++i) {
            const double z=-Y[i]*eta[i];
            sum += std::max(0.0,z)+std::log1p(std::exp(-std::abs(z)));
        }
        return sum/Y.size();
    };
    auto negative_gradient = [&](const VectorXd& alpha) -> VectorXd {
        const VectorXd eta=profile(alpha);
        VectorXd residual(Y.size());
        for (int i=0; i<Y.size(); ++i) residual[i]=Y[i]*hapc_expit(-Y[i]*eta[i]);
        return Xtilde.transpose()*residual/Y.size();
    };
    return sv_descent(ENn, alpha0, max_iter, tol, step_factor, verbose, "grad",
                      risk, negative_gradient);
}

OptimizerOutput pcghal_classi_call(const VectorXd& Y,const MatrixXd& Z,const MatrixXd& V,
                                  const VectorXd& initial,int max_iter,double tol,
                                  double step_factor,bool verbose) {
    if (!((Y.array()==1.0) || (Y.array()==-1.0)).all())
        throw std::runtime_error("Logistic SV solver requires labels -1 or +1");
    return sv_ball_fit(Y,Z,V,initial,max_iter,tol,step_factor,verbose,"grad",true);
}
