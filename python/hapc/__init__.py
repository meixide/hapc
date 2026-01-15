"""HAPC: Highly Adaptive Principal Components."""

__version__ = "0.2.0"

from .core import (
    pchal_design,
    ridge_regression,
    mkernel,
    kernel_cross,
    pcghal,
    pcghal_classification,
    fast_pchal,
)
from .single import single_lambda_fit, single_pcghal, hapc, SinglePcghalResult
from .cv import pcghal_cv, CVResult, fasthal_cv, cv_hapc

__all__ = [
    "pchal_design",
    "ridge_regression",
    "mkernel",
    "kernel_cross",
    "pcghal",
    "pcghal_classification",
    "fast_pchal",
    "single_lambda_fit",
    "single_pcghal",
    "hapc",
    "SinglePcghalResult",
    "pcghal_cv",
    "cv_hapc",
    "fasthal_cv",
    "CVResult",
]
