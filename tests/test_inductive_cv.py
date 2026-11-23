"""Inductive cross-validation for gaussian norm "1"/"2" (and binomial "2"/"sv").

Checks, against an independent construction from the explicit HAL dictionary
of the training rows only:

1. a fold fit does not depend on the covariates of its held-out rows;
2. the fold fits are the exact minimisers of the training-fold problems
   (ridge by its normal equations, lasso by coordinate descent);
3. the original CV is reproduced bit for bit with ``inductive=False``;
4. the data-scaled grid, hold-out selection, fold averaging, clipping and a
   user-supplied grid behave as documented.
"""

import itertools

import numpy as np
import pytest

from hapc import cross_kernel_hapc, cv_hapc, design_hapc, hapc_core
from hapc.cv import fasthal_cv


def dictionary(Xknots, Xeval, max_degree):
    """H[i, (s, t)] = 1{Xknots[t, s] <= Xeval[i, s]} for sections |s| <= max_degree."""
    p = Xknots.shape[1]
    blocks = []
    for k in range(1, max_degree + 1):
        for s in itertools.combinations(range(p), k):
            s = list(s)
            blocks.append(np.all(Xeval[:, None, s] >= Xknots[None, :, s], axis=2))
    return np.hstack(blocks).astype(float)


def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def direct_fold_prediction(X, Y, tr, Xeval, lam, l1, r, max_degree):
    """Prediction at Xeval of the exact training-fold minimiser.

    Basis: SVD of the explicit centred training dictionary (no kernel, no
    eigendecomposition of the package). r=None means all PCs; then ridge is
    solved in kernel-free form on the dictionary itself.
    """
    H = dictionary(X[tr], X[tr], max_degree)
    hbar = H.mean(axis=0)
    Hc = H - hbar
    ybar = Y[tr].mean()
    yc = Y[tr] - ybar
    nt = len(tr)
    He = dictionary(X[tr], Xeval, max_degree) - hbar
    if r is None and not l1:
        beta = Hc.T @ np.linalg.solve(Hc @ Hc.T + nt * lam * np.eye(nt), yc)
        return ybar + He @ beta
    _, s, vt = np.linalg.svd(Hc, full_matrices=False)
    Z, Ze = Hc @ vt[:r].T, He @ vt[:r].T
    if not l1:
        a = np.linalg.solve(Z.T @ Z + nt * lam * np.eye(r), Z.T @ yc)
    else:  # coordinate descent for (1/2n)||yc - Z a||^2 + lam ||a||_1
        a = np.zeros(r)
        col = (Z ** 2).sum(axis=0) / nt
        for _ in range(1000):
            a_old = a.copy()
            for j in range(r):
                rj = yc - Z @ a + Z[:, j] * a[j]
                a[j] = soft(Z[:, j] @ rj / nt, lam) / col[j]
            if np.max(np.abs(a - a_old)) < 1e-15:
                break
    return ybar + Ze @ a


def sim(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, d))
    Y = np.sin(2 * np.pi * X[:, 0]) + (X[:, -1] > 0.5) + 0.3 * rng.standard_normal(n)
    return X, Y, rng


LAMS = list(np.geomspace(1e-5, 1e-1, 6))


@pytest.mark.parametrize("l1", [False, True])
def test_fold_fit_does_not_depend_on_heldout_covariates(l1):
    X, Y, rng = sim(60, 2)
    kw = dict(npcs=60, lambdas=LAMS, nfolds=5, max_degree=2, l1=l1,
              selection="fold_average")
    a = fasthal_cv(X, Y, inductive=True, **kw)
    X2 = X.copy()
    te = a.foldid == 1
    X2[te] = rng.uniform(size=(te.sum(), 2))
    b = fasthal_cv(X2, Y, inductive=True, **kw)
    np.testing.assert_array_equal(a.foldid, b.foldid)
    np.testing.assert_allclose(b.fold_alphas[0], a.fold_alphas[0], rtol=0, atol=1e-12)
    assert b.fold_intercepts[0] == a.fold_intercepts[0]
    # the original (full-sample basis) fold fit does move
    a0 = fasthal_cv(X, Y, inductive=False, **kw)
    b0 = fasthal_cv(X2, Y, inductive=False, **kw)
    assert np.max(np.abs(b0.fold_alphas[0] - a0.fold_alphas[0])) > 1e-6


@pytest.mark.parametrize("d,max_degree,npcs", [(1, 1, None), (1, 1, 10), (2, 2, None), (2, 2, 25)])
@pytest.mark.parametrize("l1", [False, True])
def test_fold_fits_are_exact_minimisers(d, max_degree, npcs, l1):
    if l1 and npcs is None:
        npcs = 30  # lasso is checked by coordinate descent on a fixed PC set
    n = 60 if d == 1 else 50
    X, Y, _ = sim(n, d, seed=d)
    res = fasthal_cv(X, Y, npcs=n if npcs is None else npcs, lambdas=LAMS, nfolds=5,
                     max_degree=max_degree, l1=l1, inductive=True)
    err = 0.0
    for k in range(1, 6):
        tr = np.flatnonzero(res.foldid != k)
        te = np.flatnonzero(res.foldid == k)
        assert res.fold_ranks[k - 1] == (len(tr) - 1 if npcs is None else npcs)
        for j, lam in enumerate(LAMS):
            ref = direct_fold_prediction(X, Y, tr, X[te], lam, l1, npcs, max_degree)
            err = max(err, np.max(np.abs(res.cv_predictions[te, j] - ref)))
    assert err < 1e-8


def test_original_fold_fits_are_not_exact():
    """Documents the defect fixed by inductive=True (d = 1, n = 60, r = 10)."""
    X, Y, _ = sim(60, 1, seed=1)
    res = fasthal_cv(X, Y, npcs=10, lambdas=LAMS, nfolds=5, l1=False,
                     inductive=False, selection="fold_average")
    tr = np.flatnonzero(res.foldid != 1)
    te = np.flatnonzero(res.foldid == 1)
    ref = direct_fold_prediction(X, Y, tr, X[te], LAMS[0], False, 10, 1)
    assert np.max(np.abs(res.cv_predictions[te, 0] - ref)) > 1e-3


@pytest.mark.parametrize("l1", [False, True])
def test_inductive_false_reproduces_original_cv(l1):
    X, Y, rng = sim(70, 2)
    Xp = rng.uniform(size=(9, 2))
    old = hapc_core.fasthal_cv_fit(X, Y, 70, LAMS, 5, Xp, 2, True, False, l1)
    new = hapc_core.fasthal_cv_general_fit(X, Y, 70, LAMS, 0, 5, Xp, 2, True, l1,
                                           False, "refit", 0.8, False)
    np.testing.assert_allclose(new.mses, old.mses, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(new.best_alpha, old.best_alpha, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(new.predictions, old.predictions, rtol=1e-12, atol=1e-14)
    # inductive=False takes the original route
    cv = cv_hapc(X, Y, norm="1" if l1 else "2", max_degree=2, npcs=70, predict=Xp,
                 lambdas=LAMS, inductive=False)
    np.testing.assert_array_equal(cv.mses, old.mses)
    assert cv.fold_mses is None
    # defaults: inductive CV on the scaled 40-point grid; giving the
    # log-lambda bounds selects the fixed grid; approx=True keeps the original
    norm = "1" if l1 else "2"
    d = cv_hapc(X, Y, norm=norm, npcs=70)
    assert d.inductive and d.lambda_grid == "scaled" and d.lambdas.size == 40
    f = cv_hapc(X, Y, norm=norm, npcs=70, log_lambda_min=-5)
    assert f.inductive
    np.testing.assert_allclose(f.lambdas, np.exp(np.linspace(-5, -3, 10)))
    o = cv_hapc(X, Y, norm=norm, npcs=70, approx=True)
    assert o.fold_mses is None
    np.testing.assert_allclose(o.lambdas, np.exp(np.linspace(-5, -3, 10)))


def test_scaled_grid():
    X, Y, _ = sim(80, 2)
    ridge = cv_hapc(X, Y, norm="2", max_degree=2, inductive=True, lambda_grid="scaled")
    lasso = cv_hapc(X, Y, norm="1", max_degree=2, inductive=True, lambda_grid="scaled")
    for r in (ridge, lasso):
        assert r.lambdas.size == 40 and r.lambda_grid == "scaled"
        assert np.all(np.diff(np.log(r.lambdas)) > 0)
    np.testing.assert_allclose(ridge.lambdas[-1] / ridge.lambdas[0], 1e10, rtol=1e-10)
    np.testing.assert_allclose(lasso.lambdas[-1] / lasso.lambdas[0], 1e5, rtol=1e-10)
    # ridge scale: max over folds of s_1^2 / n_train
    s1 = []
    for k in range(1, 6):
        tr = lasso.foldid != k
        K = hapc_core.mkernel_call(X[tr], 2, True)
        s1.append(np.linalg.eigvalsh(K)[-1] / tr.sum())
    np.testing.assert_allclose(ridge.lambdas[-1], 10 * max(s1), rtol=1e-10)
    # lasso: lambda_max zeroes every fold fit; the next value does not
    top = np.array([np.abs(a[:, -1]).max() for a in lasso.fold_alphas])
    nxt = np.array([np.abs(a[:, -2]).max() for a in lasso.fold_alphas])
    assert np.all(top == 0) and np.any(nxt > 0)
    # grid scales with n
    Xb, Yb, _ = sim(160, 2, seed=5)
    big = cv_hapc(Xb, Yb, norm="2", max_degree=2, inductive=True, lambda_grid="scaled")
    assert big.lambdas[-1] > 1.5 * ridge.lambdas[-1]


def test_holdout_selection_and_clip():
    X, Y, rng = sim(100, 2)
    Xp = rng.uniform(-0.5, 1.5, size=(50, 2))
    r = cv_hapc(X, Y, norm="2", max_degree=2, inductive=True, lambda_grid="scaled",
                selection="holdout", clip=True, predict=Xp)
    tr = r.foldid == 0
    assert tr.sum() == 80 and r.fold_mses.shape == (1, 40)
    assert np.all(np.isnan(r.cv_predictions[tr]))
    assert np.all(np.isfinite(r.cv_predictions[~tr]))
    np.testing.assert_allclose(r.mses, r.fold_mses[0])
    np.testing.assert_allclose(r.best_model_alpha, r.fold_alphas[0][:, r.best_index])
    assert r.intercept == pytest.approx(Y[tr].mean())
    assert Y[tr].min() <= r.predictions.min() and r.predictions.max() <= Y[tr].max()
    vals = r.cv_predictions[~tr]
    assert Y[tr].min() <= vals.min() and vals.max() <= Y[tr].max()
    # the returned fit is the training-part fit (no refit)
    ref = direct_fold_prediction(X, Y, np.flatnonzero(tr), Xp, r.best_lambda, False,
                                 None, 2)
    np.testing.assert_allclose(r.predictions, np.clip(ref, Y[tr].min(), Y[tr].max()),
                               atol=1e-8)


def test_fold_average_predictions():
    X, Y, rng = sim(60, 1)
    Xp = rng.uniform(size=(12, 1))
    r = cv_hapc(X, Y, norm="2", inductive=True, selection="fold_average",
                lambdas=LAMS, predict=Xp)
    assert r.best_model_alpha.size == 0 and np.isnan(r.intercept)
    ref = np.mean([direct_fold_prediction(X, Y, np.flatnonzero(r.foldid != k), Xp,
                                          r.best_lambda, False, None, 1)
                   for k in range(1, 6)], axis=0)
    np.testing.assert_allclose(r.predictions, ref, atol=1e-8)


def test_user_grid_and_argument_checks():
    X, Y, _ = sim(50, 1)
    for ind in (False, True):
        r = cv_hapc(X, Y, norm="1", lambdas=[0.3, 0.01, 0.001], inductive=ind)
        np.testing.assert_array_equal(r.lambdas, [0.3, 0.01, 0.001])
    with pytest.raises(ValueError):
        cv_hapc(X, Y, norm="1", lambdas=[0.1], lambda_grid="scaled")
    with pytest.raises(ValueError):
        cv_hapc(X, Y, norm="sv", inductive=True)
    with pytest.raises(ValueError):
        cv_hapc(X, (Y > 0).astype(float), family="binomial", norm="2", clip=True)
    with pytest.raises(ValueError):
        cv_hapc(X, Y, norm="2", inductive=True, approx=True)


def _logistic_intercept(y, eta):
    b = 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-(eta + b)))
        step = (p.sum() - y.sum()) / (p * (1 - p)).sum()
        b -= step
        if abs(step) < 1e-14:
            break
    return b


def test_binomial_ridge_inductive_matches_training_only_design():
    X, _, rng = sim(80, 2, seed=3)
    Y = rng.binomial(1, 1 / (1 + np.exp(-(3 * X[:, 0] - 1.5)))).astype(float)
    lams = [1e-3, 1e-2]
    npcs = 15
    r = cv_hapc(X, Y, family="binomial", norm="2", max_degree=2, npcs=npcs,
                lambdas=lams, inductive=True)
    foldid = fasthal_cv(X, Y, npcs=npcs, lambdas=lams, inductive=True).foldid
    dev = np.zeros((5, len(lams)))
    for k in range(1, 6):
        tr, te = foldid != k, foldid == k
        des = design_hapc(X[tr], max_degree=2, npcs=npcs, center=True)
        U, d = np.asarray(des.U), np.asarray(des.d)
        Z = U * d
        Zte = cross_kernel_hapc(X[tr], X[te], max_degree=2, center=True) @ U / d
        for j, lam in enumerate(lams):
            a = np.asarray(hapc_core.logistic_ridge_init_y01(Y[tr], Z, lam))
            b0 = _logistic_intercept(Y[tr], Z @ a)
            pr = np.clip(1 / (1 + np.exp(-(Zte @ a + b0))), 1e-15, 1 - 1e-15)
            dev[k - 1, j] = -np.mean(Y[te] * np.log(pr) + (1 - Y[te]) * np.log(1 - pr))
    np.testing.assert_allclose(r.mses, dev.mean(axis=0), rtol=1e-7)
    orig = cv_hapc(X, Y, family="binomial", norm="2", max_degree=2, npcs=npcs,
                   lambdas=lams, inductive=False)
    assert np.max(np.abs(orig.mses - r.mses)) > 1e-6
    d = cv_hapc(X, Y, family="binomial", norm="2", max_degree=2, npcs=npcs,
                lambdas=lams)
    np.testing.assert_array_equal(d.mses, r.mses)  # inductive by default


def test_binomial_sv_inductive_runs():
    X, _, rng = sim(60, 2, seed=4)
    Y = rng.binomial(1, 1 / (1 + np.exp(-(3 * X[:, 0] - 1.5)))).astype(float)
    r = cv_hapc(X, Y, family="binomial", norm="sv", npcs=20, lambdas=[1e-2, 1e-1],
                inductive=True, max_iter=500)
    assert np.all(np.isfinite(r.mses))
