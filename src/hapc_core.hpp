#ifndef HAPC_CORE_HPP
#define HAPC_CORE_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <limits>

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

struct DesignOutput {
    MatrixXd H;
    MatrixXd U;
    VectorXd d;
    MatrixXd V;
};

struct OptimizerOutput {
    VectorXd alpha;
    MatrixXd alphaiters;
    VectorXd beta;
    double risk;
    int iter;
};

DesignOutput pchal_des(const MatrixXd& X, int maxdeg, int npc, bool center);
VectorXd ridge_call(const VectorXd& Y, const MatrixXd& U, const VectorXd& D2, double lambda);
MatrixXd mkernel_call(const MatrixXd& X, int m, bool center);
MatrixXd kernel_cross_call(const MatrixXd& Xtr, const MatrixXd& Xte, int m, bool center);
OptimizerOutput pcghal_call(const VectorXd& Y, const MatrixXd& Xtilde, 
                            const MatrixXd& ENn, const VectorXd& alpha0,
                            int max_iter, double tol, double step_factor, 
                            bool verbose, const std::string& crit);
OptimizerOutput pcghal_classi_call(const VectorXd& Y, const MatrixXd& Xtilde,
                                    const MatrixXd& ENn, const VectorXd& alpha0,
                                    int max_iter, double tol, double step_factor, 
                                    bool verbose);
VectorXd fast_pchal_call(const MatrixXd& U, const VectorXd& D2, const VectorXd& Y, double lambda);

// Single lambda fit with PC-GHAL optimizer (C++ wrapper for Python)
struct SinglePcghalOutput {
    VectorXd alpha;
    VectorXd predictions;
    double lambda_val;  // renamed from 'lambda' to avoid Python keyword issues
    double risk;
    int iter;
};

SinglePcghalOutput single_pcghal_fit(const MatrixXd& X, const VectorXd& Y, 
                                     int maxdeg, int npc, double single_lambda,
                                     const MatrixXd& predict_data,
                                     int max_iter, double tol, double step_factor,
                                     bool verbose, const std::string& crit,
                                     bool center, bool approx, const std::string& ini);

// Cross-validation output structure for Python
struct CVOutput {
    std::vector<double> mses;
    std::vector<double> lambdas;
    double best_lambda;
    VectorXd best_alpha;
    VectorXd predictions;
};

// CV with PC-GHAL optimizer (C++ wrapper for Python)
CVOutput pcghal_cv_fit(const MatrixXd& X, const VectorXd& Y,
                       int maxdeg, int npc, const std::vector<double>& lambdas,
                       int nfolds, const MatrixXd& predict_data,
                       int max_iter, double tol, double step_factor,
                       bool verbose, const std::string& crit,
                       bool center, bool approx, const std::string& ini);

// CV output structure for Python
struct FastCVOutput {
    std::vector<double> mses;
    std::vector<double> lambdas;
    double best_lambda;
    VectorXd best_alpha;
    VectorXd predictions;
};

// Python-friendly CV function (matches R fasthal_cv_call parameter order)
FastCVOutput fasthal_cv_python(const MatrixXd& X, const VectorXd& Y, int npc,
                                const std::vector<double>& lambdas, int nfolds,
                                const MatrixXd& predict, int maxdeg,
                                bool center, bool approx, bool l1);

// Pure C++ Newton-Raphson logistic ridge initializer.
// Y must contain values in {-1, +1}. lambda is the *unscaled* penalty
// (internally multiplied by n, matching logistic_call).
VectorXd logistic_ridge_init(const VectorXd& Y_pm1, const MatrixXd& X, double lambda);

// Cross-validation output for binomial (logistic) HAPC.
struct CVClassiOutput {
    std::vector<double> deviances;
    std::vector<double> lambdas;
    double best_lambda;
    VectorXd best_alpha;
    VectorXd predictions;          // predicted probabilities on `predict`
};

// Python-friendly binomial CV (mirrors R `pchal_cv_classi_call`).
// Y must contain only 0 or 1 values.
//
// When `with_pgd == true` (default): per fold runs logistic-ridge initialiser
// followed by projected gradient descent on logistic loss (norm="sv").
// When `with_pgd == false`: per fold runs only the logistic-ridge initialiser
// (norm="2"; logistic ridge in the PC basis), then evaluates logistic deviance
// on the held-out fold. Folds and risk are *always* logistic; no MSE on Y.
CVClassiOutput pcghal_cv_classi_python(const MatrixXd& X, const VectorXd& Y,
                                        int maxdeg, int npc,
                                        const std::vector<double>& lambdas,
                                        int nfolds, const MatrixXd& predict_data,
                                        int max_iter, double tol, double step_factor,
                                        bool verbose, bool center,
                                        bool with_pgd = true);

#endif
