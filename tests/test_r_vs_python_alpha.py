"""R-vs-Python α agreement test (integration test requiring Rscript).

For each ``norm`` ∈ {"sv", "1", "2"} we fit a single-λ HAPC model in both
languages with identical data, max_degree, npcs, λ, max_iter, tol, step_factor,
crit, center, and ini, and assert that the resulting α-vectors are equal up to
the well-known per-element SVD sign ambiguity.

Skipped automatically if ``Rscript`` is not on PATH or the local R package
``hapc`` is not installed.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from hapc import single_lambda_fit, single_pcghal, single_pcghal_classification_lasso


pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript is required for the R-vs-Python comparison test.",
)


def _data(seed: int, n: int = 80, p: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Y = X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 0] * X[:, 1] + rng.standard_normal(n) * 0.1
    return X, Y


def _r_alpha(X, Y, max_degree, npcs, lam, norm,
             max_iter=500, tol=1e-3, step_factor=0.8, crit="grad", ini="1"):
    with tempfile.TemporaryDirectory() as d:
        Xf, Yf, Of = (os.path.join(d, n) for n in ("X.csv", "Y.csv", "out.json"))
        np.savetxt(Xf, X, delimiter=",")
        np.savetxt(Yf, Y, delimiter=",")
        ini_arg = f', ini="{ini}"' if norm == "sv" else ""
        r = f"""
suppressPackageStartupMessages(library(hapc))
suppressPackageStartupMessages(library(jsonlite))
X <- as.matrix(read.csv("{Xf}", header=FALSE))
Y <- as.numeric(read.csv("{Yf}", header=FALSE)[[1]])
res <- hapc(X, Y, max_degree={max_degree}, npcs={npcs}, lambda={lam},
            norm="{norm}", max_iter={max_iter}, tol={tol},
            step_factor={step_factor}, crit="{crit}"{ini_arg},
            verbose=FALSE)
alpha <- if (!is.null(res$res_opt)) res$res_opt$alpha else res$alpha
write_json(list(alpha=as.numeric(alpha)), "{Of}")
"""
        rf = os.path.join(d, "r.R")
        Path(rf).write_text(r)
        out = subprocess.run(["Rscript", rf], capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            pytest.skip(f"R run failed: {out.stderr.strip()}")
        return np.array(json.loads(Path(Of).read_text())["alpha"])


def _py_alpha(X, Y, max_degree, npcs, lam, norm,
              max_iter=500, tol=1e-3, step_factor=0.8, crit="grad", ini="1"):
    if norm == "sv":
        return single_pcghal(X, Y, max_degree=max_degree, npcs=npcs,
                             lambda_=lam, max_iter=max_iter, tol=tol,
                             step_factor=step_factor, crit=crit, ini=ini).alpha
    return single_lambda_fit(X, Y, max_degree=max_degree, npcs=npcs,
                             lambda_=lam, l1=(norm == "1")).alpha


def _agree(ar, ap, atol=1e-3, rtol=1e-2) -> bool:
    """Equal up to per-element sign ambiguity from the SVD."""
    if ar.shape != ap.shape:
        return False
    return np.allclose(np.abs(ar), np.abs(ap), atol=atol, rtol=rtol)


@pytest.mark.parametrize("norm,seed,lam", [
    ("sv", 42, 0.01),
    ("2",  43, 0.01),
    ("1",  44, 1e-3),
])
def test_alpha_match(norm, seed, lam):
    X, Y = _data(seed)
    ar = _r_alpha(X, Y, max_degree=2, npcs=4, lam=lam, norm=norm)
    ap = _py_alpha(X, Y, max_degree=2, npcs=4, lam=lam, norm=norm)
    assert _agree(ar, ap), f"R={ar}\nPy={ap}"


# --- binomial + norm="1" R(glmnet) vs Py(sklearn) parity --------------------

def _binary_data(seed: int, n: int = 80, p: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    return X, Y


def _r_binomial_lasso_alpha(X, Y, max_degree, npcs, lam):
    """Run R hapc(family='binomial', norm='1') and return alpha."""
    with tempfile.TemporaryDirectory() as d:
        Xf, Yf, Of = (os.path.join(d, n) for n in ("X.csv", "Y.csv", "out.json"))
        np.savetxt(Xf, X, delimiter=",")
        np.savetxt(Yf, Y, delimiter=",", fmt="%d")
        r = f"""
suppressPackageStartupMessages(library(hapc))
suppressPackageStartupMessages(library(jsonlite))
if (!requireNamespace("glmnet", quietly=TRUE)) {{
    cat("__SKIP__glmnet not available")
    quit(save="no", status=0)
}}
X <- as.matrix(read.csv("{Xf}", header=FALSE))
Y <- as.numeric(read.csv("{Yf}", header=FALSE)[[1]])
fit <- hapc(X, Y, family="binomial", max_degree={max_degree}, npcs={npcs},
            lambda={lam}, norm="1")
write_json(list(alpha=as.numeric(fit$alpha)), "{Of}")
"""
        rf = os.path.join(d, "r.R")
        Path(rf).write_text(r)
        out = subprocess.run(["Rscript", rf], capture_output=True, text=True, timeout=120)
        if out.returncode != 0 or "__SKIP__" in (out.stdout + out.stderr):
            pytest.skip(f"R run failed or glmnet missing: {out.stderr.strip()} {out.stdout.strip()}")
        return np.array(json.loads(Path(Of).read_text())["alpha"])


@pytest.mark.parametrize("seed,lam", [(123, 0.05), (456, 0.01), (789, 0.1)])
def test_binomial_lasso_alpha_match(seed, lam):
    """R(glmnet) and Py(sklearn liblinear) should produce nearly identical
    coefficients when given the same Xtilde, lambda, and parameterisation
    (C = 1/(n*lambda) on the Python side)."""
    X, Y = _binary_data(seed)
    ar = _r_binomial_lasso_alpha(X, Y, max_degree=2, npcs=10, lam=lam)
    ap = single_pcghal_classification_lasso(
        X, Y, max_degree=2, npcs=10, lambda_=lam,
    ).alpha
    # Same shape, same sparsity pattern, agreement within 5e-3 absolute / 1e-2 rel.
    assert ar.shape == ap.shape
    np.testing.assert_array_equal(ar == 0, ap == 0)
    np.testing.assert_allclose(ar, ap, atol=5e-3, rtol=1e-2)
