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
    std::string status = "not_reported";
    double sv_radius = 0;
    double tangent_norm = 0;
    std::string constraint = "boundary";
    bool converged = false;
    double optimality_gap = std::numeric_limits<double>::infinity();
    double dual_bound = -std::numeric_limits<double>::infinity();
    double dual_residual = std::numeric_limits<double>::infinity();
    double primal_residual = 0;
    double constraint_violation = 0;
};

DesignOutput pchal_des(const MatrixXd& X, int maxdeg, int npc, bool center);
VectorXd ridge_call(const VectorXd& Y, const MatrixXd& U, const VectorXd& D2, double lambda);
MatrixXd mkernel_call(const MatrixXd& X, int m, bool center);
MatrixXd kernel_cross_call(const MatrixXd& Xtr, const MatrixXd& Xte, int m, bool center);
// Uncentred HAL kernel between rows of Xa and Xb with knots at rows of Xtr
// (multi-threaded; HAPC_NUM_THREADS overrides the thread count).
MatrixXd hal_kernel_uncentered(const MatrixXd& Xtr, const MatrixXd& Xa,
                               const MatrixXd& Xb, int m, bool symmetric);
int hapc_num_threads();
OptimizerOutput pcghal_call(const VectorXd& Y, const MatrixXd& Xtilde, 
                            const MatrixXd& ENn, const VectorXd& alpha0,
                            int max_iter, double tol, double step_factor, 
                            bool verbose, const std::string& crit);
OptimizerOutput pcghal_classi_call(const VectorXd& Y, const MatrixXd& Xtilde,
                                    const MatrixXd& ENn, const VectorXd& alpha0,
                                    int max_iter, double tol, double step_factor, 
                                    bool verbose);
OptimizerOutput pcghal_boundary_call(const VectorXd&,const MatrixXd&,const MatrixXd&,const VectorXd&,
                                      int,double,double,bool,const std::string&);
OptimizerOutput pcghal_boundary_classi_call(const VectorXd&,const MatrixXd&,const MatrixXd&,const VectorXd&,
                                           int,double,double,bool);
VectorXd fast_pchal_call(const MatrixXd& U, const VectorXd& D2, const VectorXd& Y, double lambda);

// Single lambda fit with PC-GHAL optimizer (C++ wrapper for Python)
struct SinglePcghalOutput {
    VectorXd alpha;
    VectorXd predictions;
    double lambda_val;  // renamed from 'lambda' to avoid Python keyword issues
    double risk;
    int iter;
    OptimizerOutput optimizer_output;
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
    OptimizerOutput optimizer_output;
    std::vector<std::string> sv_fold_status;
    std::vector<double> sv_fold_gap;
    std::vector<double> sv_fold_radius;
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

// Gaussian norm "1"/"2" CV with an optional inductive basis per training part
// (src/cv_inductive.cpp). An empty `lambdas` asks for the data-scaled grid
// with `nlambda` points. selection: "refit" (full-sample refit at the selected
// lambda), "fold_average" (average of the V fold fits) or "holdout" (single
// split, fit on the training part).
struct GeneralCVOutput {
    std::vector<double> mses;
    std::vector<double> lambdas;
    double best_lambda;
    int best_index;                    // 0-based
    MatrixXd fold_mses;                // parts x L
    MatrixXd cv_predictions;           // n x L held-out predictions (NaN elsewhere)
    std::vector<int> foldid;           // V-fold: 1..V; hold-out: 0 train, 1 validation
    std::vector<MatrixXd> fold_alphas; // per part, r_k x L
    std::vector<double> fold_intercepts;
    std::vector<int> fold_ranks;
    VectorXd best_alpha;               // empty for fold_average
    double intercept;                  // NaN for fold_average
    int rank;                          // PCs of the returned basis (0 for fold_average)
    VectorXd predictions;
    std::string grid;                  // "user" or "scaled"
};

GeneralCVOutput fasthal_cv_general(const MatrixXd& X, const VectorXd& Y, int npc,
                                   const std::vector<double>& lambdas, int nlambda,
                                   int nfolds, const MatrixXd& predict, int maxdeg,
                                   bool center, bool l1, bool inductive,
                                   const std::string& selection, double train_frac,
                                   bool clip);

// Pure C++ Newton-Raphson logistic ridge with an unpenalised intercept
// (only the slopes are returned). Y must contain values in {-1, +1}.
// Objective as glmnet(alpha = 0, standardize = FALSE):
// -(1/n) loglik + (lambda / 2) ||beta||^2.
VectorXd logistic_ridge_init(const VectorXd& Y_pm1, const MatrixXd& X, double lambda);

// Soft-label variant: target `y01` may take any value in [0, 1] (hard {0,1}
// labels or fractional EM-HAL E-step posteriors). On hard {0,1} inputs the
// result is identical to logistic_ridge_init. lambda has the same scaling.
VectorXd logistic_ridge_init_y01(const VectorXd& y01, const MatrixXd& X, double lambda);

// Cross-validation output for binomial (logistic) HAPC.
struct CVClassiOutput {
    std::vector<double> deviances;
    std::vector<double> lambdas;
    double best_lambda;
    VectorXd best_alpha;
    VectorXd predictions;          // predicted probabilities on `predict`
    OptimizerOutput optimizer_output;
    std::vector<std::string> sv_fold_status;
    std::vector<double> sv_fold_gap;
    std::vector<double> sv_fold_radius;
};

// Python-friendly binomial CV (mirrors R `pchal_cv_classi_call`).
// Y must lie in [0,1]: hard {0,1} labels or soft EM-HAL posteriors. Soft
// labels are supported only when with_pgd == false (norm="2"); with_pgd ==
// true (norm="sv") rejects soft labels.
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
