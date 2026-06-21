Discrete-time survival (``family = "logit-hazard"``)
====================================================

:func:`hapc.hazard_hapc` fits a discrete-time **logistic hazard** model with
HAPC from right-censored survival data ``(X, T, Delta)``.  It performs the
person-period expansion, prepends the visit time as the first HAL covariate, and
cross-validates the binomial fit.  The full statistical derivation (model,
person-period likelihood, survival function) and references are in the function
docstring below.

The same routine is reachable from the cross-validation dispatcher as
``cv_hapc(X, T, family="logit-hazard", Delta=Delta, norm="1")``.

.. autofunction:: hapc.hazard_hapc

Result type
-----------

The field-by-field description is rendered from the class docstring.

.. autoclass:: hapc.HazardResult
