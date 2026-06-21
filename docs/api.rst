API reference
=============

This page is generated directly from the package docstrings.

High-level model fitting
------------------------

.. autofunction:: hapc.hapc
.. autofunction:: hapc.cv_hapc
.. autofunction:: hapc.hazard_hapc
   :no-index:
.. autofunction:: hapc.ate_hapc

Result types
------------

.. autoclass:: hapc.HazardResult
   :no-index:
.. autoclass:: hapc.CVResult
   :exclude-members: count, index, _make, _asdict, _replace, _fields,
                     _field_defaults
.. autoclass:: hapc.ATEResult
   :exclude-members: count, index, _make, _asdict, _replace, _fields,
                     _field_defaults

Design & kernels
----------------

.. autofunction:: hapc.design_hapc
.. autofunction:: hapc.kernel_hapc
.. autofunction:: hapc.cross_kernel_hapc

Solvers & single-λ fits
-----------------------

.. autofunction:: hapc.ridge_regression
.. autofunction:: hapc.fast_pchal
.. autofunction:: hapc.single_pcghal
.. autofunction:: hapc.single_lambda_fit
.. autofunction:: hapc.single_pcghal_classification
.. autofunction:: hapc.single_pcghal_classification_ridge_only
.. autofunction:: hapc.single_pcghal_classification_lasso

Cross-validation
----------------

.. autofunction:: hapc.pcghal_cv
.. autofunction:: hapc.pcghal_cv_classi
.. autofunction:: hapc.pcghal_cv_classi_lasso
.. autofunction:: hapc.fasthal_cv
