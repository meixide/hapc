#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "hapc_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hapc_core, m) {
    m.doc() = "HAPC: Highly Adaptive Principal Components";
    
    py::class_<DesignOutput>(m, "DesignOutput")
        .def_readonly("H", &DesignOutput::H)
        .def_readonly("U", &DesignOutput::U)
        .def_readonly("d", &DesignOutput::d)
        .def_readonly("V", &DesignOutput::V);
    
    py::class_<OptimizerOutput>(m, "OptimizerOutput")
        .def_readonly("alpha", &OptimizerOutput::alpha)
        .def_readonly("alphaiters", &OptimizerOutput::alphaiters)
        .def_readonly("beta", &OptimizerOutput::beta)
        .def_readonly("risk", &OptimizerOutput::risk)
        .def_readonly("iter", &OptimizerOutput::iter);
    
    py::class_<SinglePcghalOutput>(m, "SinglePcghalOutput")
        .def_readonly("alpha", &SinglePcghalOutput::alpha)
        .def_readonly("predictions", &SinglePcghalOutput::predictions)
        .def_readonly("lambda_val", &SinglePcghalOutput::lambda_val)
        .def_readonly("risk", &SinglePcghalOutput::risk)
        .def_readonly("iter", &SinglePcghalOutput::iter);
    
    py::class_<CVOutput>(m, "CVOutput")
        .def_readonly("mses", &CVOutput::mses)
        .def_readonly("lambdas", &CVOutput::lambdas)
        .def_readonly("best_lambda", &CVOutput::best_lambda)
        .def_readonly("best_alpha", &CVOutput::best_alpha)
        .def_readonly("predictions", &CVOutput::predictions);

    py::class_<FastCVOutput>(m, "FastCVOutput")
        .def_readonly("mses", &FastCVOutput::mses)
        .def_readonly("lambdas", &FastCVOutput::lambdas)
        .def_readonly("best_lambda", &FastCVOutput::best_lambda)
        .def_readonly("best_alpha", &FastCVOutput::best_alpha)
        .def_readonly("predictions", &FastCVOutput::predictions);

    py::class_<CVClassiOutput>(m, "CVClassiOutput")
        .def_readonly("deviances", &CVClassiOutput::deviances)
        .def_readonly("lambdas", &CVClassiOutput::lambdas)
        .def_readonly("best_lambda", &CVClassiOutput::best_lambda)
        .def_readonly("best_alpha", &CVClassiOutput::best_alpha)
        .def_readonly("predictions", &CVClassiOutput::predictions);
    
    m.def("pchal_des", &pchal_des,
          py::arg("X"), py::arg("maxdeg"), py::arg("npc"), py::arg("center") = true);
    
    m.def("ridge_call", &ridge_call,
          py::arg("Y"), py::arg("U"), py::arg("D2"), py::arg("lambda"));
    
    m.def("mkernel_call", &mkernel_call,
          py::arg("X"), py::arg("m"), py::arg("center") = true);
    
    m.def("kernel_cross_call", &kernel_cross_call,
          py::arg("Xtr"), py::arg("Xte"), py::arg("m"), py::arg("center") = true);
    
    m.def("pcghal_call", &pcghal_call,
          py::arg("Y"), py::arg("Xtilde"), py::arg("ENn"), py::arg("alpha0"),
          py::arg("max_iter") = 5000, py::arg("tol") = 0.001, 
          py::arg("step_factor") = 0.8, py::arg("verbose") = false,
          py::arg("crit") = "grad");
    
    m.def("pcghal_classi_call", &pcghal_classi_call,
          py::arg("Y"), py::arg("Xtilde"), py::arg("ENn"), py::arg("alpha0"),
          py::arg("max_iter") = 5000, py::arg("tol") = 0.001,
          py::arg("step_factor") = 0.8, py::arg("verbose") = false);
    
    m.def("fast_pchal_call", &fast_pchal_call,
          py::arg("U"), py::arg("D2"), py::arg("Y"), py::arg("lambda"));
    
    m.def("single_pcghal_fit", &single_pcghal_fit,
          py::arg("X"), py::arg("Y"), py::arg("maxdeg"), py::arg("npc"),
          py::arg("single_lambda"), py::arg("predict_data"),
          py::arg("max_iter") = 5000, py::arg("tol") = 0.001,
          py::arg("step_factor") = 0.8, py::arg("verbose") = false,
          py::arg("crit") = "grad", py::arg("center") = true, py::arg("approx") = false,
          py::arg("ini") = "1");
    
    m.def("pcghal_cv_fit", &pcghal_cv_fit,
          py::arg("X"), py::arg("Y"), py::arg("maxdeg"), py::arg("npc"),
          py::arg("lambdas"), py::arg("nfolds"),
          py::arg("predict_data"),
          py::arg("max_iter") = 5000, py::arg("tol") = 0.001,
          py::arg("step_factor") = 0.8, py::arg("verbose") = false,
          py::arg("crit") = "grad", py::arg("center") = true, py::arg("approx") = false,
          py::arg("ini") = "1");

    // Pure-C++ k-fold CV for norm in {"1", "2"} (LASSO/Ridge in PC basis).
    // Mirrors R's fasthal_cv_call exactly (same fold partitioning with
    // mt19937(12345) + shuffle), so R and Python produce identical mses /
    // best_lambda / best_alpha for the same inputs.
    m.def("fasthal_cv_fit", &fasthal_cv_python,
          py::arg("X"), py::arg("Y"), py::arg("npc"),
          py::arg("lambdas"), py::arg("nfolds"),
          py::arg("predict"), py::arg("maxdeg"),
          py::arg("center") = true, py::arg("approx") = false,
          py::arg("l1") = true);

    // Pure-C++ k-fold logistic CV (binomial + sv).  Mirrors R's
    // pchal_cv_classi_call: same logistic-ridge initialiser, same PGD step
    // selection and same fold partitioning (mt19937(12345)).
    m.def("pcghal_cv_classi_fit", &pcghal_cv_classi_python,
          py::arg("X"), py::arg("Y"), py::arg("maxdeg"), py::arg("npc"),
          py::arg("lambdas"), py::arg("nfolds"),
          py::arg("predict_data"),
          py::arg("max_iter") = 5000, py::arg("tol") = 1e-3,
          py::arg("step_factor") = 0.8, py::arg("verbose") = false,
          py::arg("center") = true,
          py::arg("with_pgd") = true);

    m.def("logistic_ridge_init", &logistic_ridge_init,
          py::arg("Y"), py::arg("X"), py::arg("lambda"));
}
