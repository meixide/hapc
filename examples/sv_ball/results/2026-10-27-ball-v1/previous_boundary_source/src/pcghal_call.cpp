#include "sv_descent.hpp"
OptimizerOutput pcghal_call(const VectorXd& Y, const MatrixXd& Xtilde,
                           const MatrixXd& ENn, const VectorXd& alpha0,
                           int max_iter, double tol, double step_factor,
                           bool verbose, const std::string& crit) {
    validate_sv_data(Y, Xtilde, ENn, alpha0);
    const double n = Y.size();
    auto risk = [&](const VectorXd& alpha) {
        return (Y-Xtilde*alpha).squaredNorm()/n;
    };
    auto negative_gradient = [&](const VectorXd& alpha) -> VectorXd {
        return (2.0/n)*Xtilde.transpose()*(Y-Xtilde*alpha);
    };
    return sv_descent(ENn, alpha0, max_iter, tol, step_factor, verbose, crit,
                      risk, negative_gradient);
}
