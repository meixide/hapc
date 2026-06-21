# hapc — Highly Adaptive Principal Components

`hapc` provides fast HAL projection-based regression, classification,
cross-validation, average treatment effect (ATE) estimation, and discrete-time
survival (logistic hazard) models, with matching R and Python APIs backed by a
shared C++ core.

```{toctree}
:maxdepth: 2
:caption: Contents

survival
api
```

## Installation

```bash
pip install hapc
```

## Quick start

```python
import numpy as np
from hapc import cv_hapc

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 3))
Y = np.sin(np.pi * X[:, 0]) + X[:, 1] + rng.standard_normal(200) * 0.1

cv = cv_hapc(X, Y, max_degree=2, npcs=199, nfolds=5, norm="sv")
print(cv.best_lambda)
```

For discrete-time survival, see {doc}`survival`. The full project README with
installation notes for Linux/HPC clusters and the R package lives on
[GitHub](https://github.com/meixide/hapc#readme).

## Indices

- {ref}`genindex`
- {ref}`modindex`
