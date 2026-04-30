"""HAPC: Highly Adaptive Principal Components.

Public API
----------
High-level entry points (mirror the R package):

- :func:`hapc` — single-λ fit (gaussian / binomial; norm in {"sv","1","2"}).
- :func:`cv_hapc` — k-fold cross-validated fit.

Lower-level building blocks:

- :func:`design_hapc`, :func:`kernel_hapc`, :func:`cross_kernel_hapc`
- :func:`ridge_regression`, :func:`fast_pchal`
- :func:`pcghal`, :func:`pcghal_classification`, :func:`pc_hal_classi`
- :func:`single_pcghal`, :func:`single_lambda_fit`,
  :func:`single_pcghal_classification`,
  :func:`single_pcghal_classification_ridge_only`
- :func:`pcghal_cv`, :func:`pcghal_cv_classi`, :func:`fasthal_cv`
- :func:`ate_hapc` — ATE estimate + Wald CI via HAPC + outcome undersmoothing.
"""

__version__ = "0.3.0"

from .core import (
    DesignOutput,
    OptimizerOutput,
    cross_kernel_hapc,
    design_hapc,
    fast_pchal,
    kernel_cross,         # alias for cross_kernel_hapc (backward compat)
    kernel_hapc,
    mkernel,              # alias for kernel_hapc       (backward compat)
    pchal_design,         # alias for design_hapc       (backward compat)
    pcghal,
    pcghal_classification,
    pc_hal_classi,
    ridge_regression,
)
from .single import (
    SingleLambdaResult,
    SinglePcghalClassificationResult,
    SinglePcghalResult,
    hapc,
    single_lambda_fit,
    single_pcghal,
    single_pcghal_classification,
    single_pcghal_classification_lasso,
    single_pcghal_classification_ridge_only,
)
from .cv import (
    CVResult,
    cv_hapc,
    fasthal_cv,
    pcghal_cv,
    pcghal_cv_classi,
    pcghal_cv_classi_lasso,
)
from .ate import ATEResult, ate_hapc

__all__ = [
    "__version__",
    # high level
    "hapc",
    "cv_hapc",
    "ate_hapc",
    # design & kernels
    "design_hapc",
    "kernel_hapc",
    "cross_kernel_hapc",
    # solvers
    "ridge_regression",
    "fast_pchal",
    "pcghal",
    "pcghal_classification",
    "pc_hal_classi",
    # single-λ
    "single_pcghal",
    "single_lambda_fit",
    "single_pcghal_classification",
    "single_pcghal_classification_ridge_only",
    "single_pcghal_classification_lasso",
    # CV
    "pcghal_cv",
    "pcghal_cv_classi",
    "pcghal_cv_classi_lasso",
    "fasthal_cv",
    # result types
    "ATEResult",
    "CVResult",
    "DesignOutput",
    "OptimizerOutput",
    "SingleLambdaResult",
    "SinglePcghalResult",
    "SinglePcghalClassificationResult",
    # backward-compat aliases
    "pchal_design",
    "mkernel",
    "kernel_cross",
]
