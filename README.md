# HAPC: Highly Adaptive Prinicipal Components

A fast and flexible machine learning library for nonparametric high-dimensional regression and classification with guarantees.

## Documentation

- **Python API** (rendered from docstrings): https://hapc.readthedocs.io —
  configured via [`.readthedocs.yaml`](.readthedocs.yaml) and
  [`docs/`](docs/) (Sphinx + autodoc). Build locally with
  `pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html`.
- **R API** (rendered from roxygen): a [pkgdown](https://pkgdown.r-lib.org)
  site built by [`.github/workflows/pkgdown.yaml`](.github/workflows/pkgdown.yaml)
  (config in [`_pkgdown.yml`](_pkgdown.yml)). Build locally with
  `Rscript -e 'pkgdown::build_site()'`.

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler (g++, clang, or MSVC)
- CMake 3.15+
- Eigen3

### Quick Install

```bash
pip install hapc
```

Prebuilt wheels are published for Linux (manylinux2014, x86_64), macOS
(Intel + Apple Silicon) and Windows, for CPython 3.8–3.12. No compiler,
CMake or Eigen is needed when a wheel is available.

### Linux / HPC clusters

The Linux wheels use the **manylinux2014** baseline (glibc 2.17), so
`pip install hapc` works out of the box on HPC login/compute nodes —
no `conda` toolchain, `devtoolset`, or sysroot setup required:

```bash
pip install hapc
```

If you must build from the source distribution (niche architecture, very
old Python, or an air-gapped node), provide a C++17 compiler and either
let CMake fetch Eigen automatically (needs network) or install Eigen and
let `find_package(Eigen3)` find it:

```bash
# with conda compilers (recommended on HPC)
conda install -c conda-forge cxx-compiler cmake eigen
pip install hapc --no-binary hapc
```

### Install from GitHub (latest development version)

```bash
pip install git+https://github.com/meixide/hapc.git
```

Or with editable install for development:

```bash
git clone https://github.com/meixide/hapc.git
cd hapc
pip install -e .
```

### Install build dependencies

If installation fails, you may need to install build dependencies:

**macOS:**
```bash
brew install cmake eigen
```

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake libeigen3-dev build-essential
```

**Windows:**
```bash
pip install cmake
# Install Visual Studio Build Tools or use conda
conda install -c conda-forge eigen
```

## Quick Start

```python
import numpy as np
from hapc.single import single_pcghal
from hapc.cv import pcghal_cv

# Generate sample data
X = np.random.randn(100, 5)
Y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

# Single fit with fixed lambda
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01)
print(f"Risk: {result.optimizer_output.risk:.6f}")

# Cross-validation to select lambda
lambdas = np.logspace(-4, 0, 10)
cv_result = pcghal_cv(X, Y, maxdeg=2, npc=5, lambdas=lambdas, nfolds=5)
print(f"Best lambda: {cv_result.best_lambda:.6f}")

# Make predictions
X_test = np.random.randn(20, 5)
result = single_pcghal(X, Y, maxdeg=2, npc=5, single_lambda=0.01, predict=X_test)
print(f"Predictions: {result.predictions}")
```

## Usage

### Regression

```python
from hapc.single import single_pcghal

result = single_pcghal(
    X, Y,
    maxdeg=2,        # Maximum degree of interactions
    npc=10,          # Number of principal components
    single_lambda=0.01,
    predict=X_test   # Optional: test data for predictions
)
```

### Classification

```python
from hapc.single import single_pcghal

result = single_pcghal(
    X, Y_binary,
    maxdeg=2,
    npc=10,
    single_lambda=0.01,
    predict=X_test
)
```

### Cross-Validation

```python
from hapc.cv import pcghal_cv

cv_result = pcghal_cv(
    X, Y,
    maxdeg=2,
    npc=10,
    lambdas=np.logspace(-4, 0, 20),
    nfolds=5
)
print(cv_result.best_lambda)
```

### Average Treatment Effect (ATE)

Estimate the ATE `E[Y(1)] − E[Y(0)]` with HAPC nuisance models and a
doubly-robust (AIPW) efficient influence function. `ate_hapc` returns a point
estimate and a `(1 − alpha)` Wald confidence interval.

```python
from hapc import ate_hapc

# W: covariates (n, p); A: binary treatment in {0,1} or {-1,+1}; Y: outcome
res = ate_hapc(W, Y, A, alpha=0.05, method="undersmooth")
print(res.estimate, res.lower, res.upper)
```

Two bias-control strategies are available through `method`:

- **`method="undersmooth"`** (default) — single-sample estimator. The outcome
  model is undersmoothed (λ pushed below the CV-optimal value) until the
  empirical influence function is within `σ / (√n · log n)`. This requires the
  **full PC basis** (`npcs = n`, the default) and a λ grid that reaches small λ
  (defaults `log_lambda_out_min = -10`); otherwise the gate never reaches the
  low-bias regime and `ate_hapc` emits a warning. Pass
  `report_undersmoothing=True` to print the `|mean(EIF)|`-vs-λ path.
- **`method="crossfit"`** — DML-style K-fold cross-fitting (`cf_folds`, default
  5, stratified by treatment). Both nuisances are fit on the training folds and
  the influence function is evaluated out-of-fold, giving honest point estimates
  and coverage without undersmoothing. Recommended under good overlap.

### Discrete-time survival (`family = "logit-hazard"`)

Fit a discrete-time **logistic hazard** model with HAPC. You supply only the
observed right-censored data — baseline covariates `X`, the observed time
`T = min(T_event, C)`, and the event indicator `Delta = 1(T_event <= C)` — and
the wrapper performs the person-period expansion (one row per
subject-per-interval-at-risk, hazard label = 1 at the event interval), prepends
the visit time as the first HAL covariate, and cross-validates the binomial fit.

**Model.** The discrete hazard is the conditional event probability in interval
`t` given survival up to `t`, modelled on the logit scale by a HAPC fit `f` of
the augmented covariate `(t, x)`:

```text
lambda(t | x) = P(T_event = t | T_event >= t, X = x)
logit lambda(t | x) = f(t, x)
```

**Person-period likelihood.** Under independent right-censoring the observed-data
likelihood factorises over the at-risk intervals,

```text
prod_i prod_{t <= T_i}  lambda(t|x_i)^Y_it * (1 - lambda(t|x_i))^(1 - Y_it),
with  Y_it = 1(T_event_i = t),
```

which is exactly the Bernoulli (logistic) likelihood of the expanded
person-period table — so a binomial HAPC fit of `Y_it` on `(t, x_i)` estimates
the discrete hazard (Cox 1972; Brown 1975; Allison 1982).

**Survival.** The conditional survival function follows by the product-limit
relation `S(t | x) = prod_{s <= t} (1 - lambda(s | x))`, returned for new
subjects when `predict=` is supplied.

```python
from hapc import hazard_hapc
import numpy as np

# X: baseline covariates (n, p); T: observed times; Delta: 0/1 event indicator
fit = hazard_hapc(X, T, Delta, norm="1", max_degree=2, time_grid=np.arange(1, 7))
fit.hazard        # estimated hazard per person-period row (CV predictions)
fit.best_lambda, fit.interior   # CV-selected lambda; is it interior to the grid?

# survival curves S(t|x) for new subjects
fit = hazard_hapc(X, T, Delta, norm="1", predict=X_new)
fit.predict_survival            # (m, K) survival probabilities over the grid
```

```r
library(hapc)
# equivalent to cv.hapc(X, T, family = "logit-hazard", Delta = Delta, norm = "1")
fit <- hazard.hapc(X, T, Delta, norm = "1", max_degree = 2, time_grid = 1:6)
fit$hazard; fit$best_lambda; fit$interior
```

`norm` must be `"1"` (logistic LASSO) or `"2"` (logistic ridge); `norm = "sv"`
is **not implemented** for this family and is flagged.

**Returns** (Python `HazardResult` / R `hapc_hazard`):

- `hazard` — cross-validated discrete hazard for each person-period row
- `lambdas`, `risk`, `best_lambda` — CV grid, mean logistic deviance, selected λ
- `interior` — whether `best_lambda` is strictly inside the grid (sanity check)
- `time_grid`, `ids`/`id`, `Y` — the discrete grid and person-period bookkeeping
- `predict_hazard`, `predict_survival` — hazard surface and survival curves for
  new subjects (only when `predict=` is given)
- `cv` — the underlying cross-validation result

Worked end-to-end examples (five hazard data-generating processes, with
true-vs-estimated hazard scatters and CV risk-vs-λ curves verifying an interior
optimum) are in
[`examples/hazard_logit_hazard_examples.R`](examples/hazard_logit_hazard_examples.R)
and
[`examples/hazard_logit_hazard_examples.py`](examples/hazard_logit_hazard_examples.py).

**References.** Cox (1972, *JRSS B*); Brown (1975, *Biometrics*); Allison (1982,
*Sociological Methodology*); Singer & Willett (2003, *Applied Longitudinal Data
Analysis*); Benkeser & van der Laan (2016, *IEEE DSAA*).

## API Reference

### `hapc.single.single_pcghal()`

Fit PC-GHAL with a single lambda value.

**Parameters:**
- `X` (ndarray, shape (n, p)): Input features
- `Y` (ndarray, shape (n,)): Response variable
- `maxdeg` (int): Maximum degree of interactions
- `npc` (int): Number of principal components
- `single_lambda` (float): Regularization parameter
- `max_iter` (int, default=100): Maximum iterations
- `tol` (float, default=1e-6): Convergence tolerance
- `verbose` (bool, default=False): Print progress
- `predict` (ndarray, optional): Test data for predictions
- `center` (bool, default=True): Center the design matrix

**Returns:**
- `result.optimizer_output.alpha`: Coefficients
- `result.optimizer_output.risk`: Final risk
- `result.optimizer_output.iter`: Iterations until convergence
- `result.predictions`: Predictions on test data (if provided)

### `hapc.cv.pcghal_cv()`

Cross-validation to select lambda.

**Parameters:**
- `lambdas` (ndarray): Grid of lambda values to test
- `nfolds` (int, default=5): Number of CV folds
- ...other parameters same as `single_pcghal`

**Returns:**
- `cv_result.best_lambda`: Optimal lambda
- `cv_result.mses`: CV errors for each lambda
- `cv_result.best_model`: Fitted model with best lambda
- `cv_result.predictions`: Predictions on test data (if provided)

## Contributing

Contributions welcome! The C++ core is shared between R and Python packages.

```bash
git clone https://github.com/meixide/hapc.git
cd hapc
pip install -e .
pytest
```

## License

MIT License - see LICENSE file
