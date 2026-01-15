"""
Test that R and Python produce matching alpha coefficients (up to sign)
for fixed lambda single-lambda fitting routines.

Tests all three norm types:
- norm="sv": gradient descent optimizer (PC-GHAL)
- norm="1": L1 penalty (LASSO)
- norm="2": L2 penalty (Ridge)
"""

import numpy as np
import subprocess
import json
import tempfile
import os
from pathlib import Path

# Add hapc to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from hapc.single import single_pcghal, single_lambda_fit


def generate_test_data(seed=42, n=100, p=5, maxdeg=2):
    """Generate reproducible test data."""
    np.random.seed(seed)
    X = np.random.randn(n, p)
    # Simple linear response with interactions
    Y = X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 0] * X[:, 1] + np.random.randn(n) * 0.1
    return X, Y


def run_r_single_lambda(X, Y, maxdeg, npc, single_lambda, norm, center=True, 
                       max_iter=500, tol=1e-1, step_factor=0.5, crit='grad'):
    """Call R hapc() and extract alpha coefficients."""
    
    # Write data to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        X_file = os.path.join(tmpdir, "X.csv")
        Y_file = os.path.join(tmpdir, "Y.csv")
        out_file = os.path.join(tmpdir, "result.json")
        
        np.savetxt(X_file, X, delimiter=",")
        np.savetxt(Y_file, Y, delimiter=",")
        
        # R script
        r_code = f"""
library(hapc)
library(jsonlite)

X <- as.matrix(read.csv("{X_file}", header=FALSE))
Y <- as.numeric(read.csv("{Y_file}", header=FALSE)[[1]])

norm_param <- "{norm}"

res <- hapc(X, Y,
            npcs = {npc},
            lambda = {single_lambda},
            norm = norm_param,
            max_degree = {maxdeg},
            center = {str(center).upper()},
            max_iter = {max_iter},
            tol = {tol},
            step_factor = {step_factor},
            crit = "{crit}",
            verbose = FALSE)

# Debug: print structure
cat("Result structure:\\n")
str(res)
cat("\\nResult names:\\n")
print(names(res))

# Extract alpha from result
if (norm_param == "sv") {{
    if (!is.null(res$res_opt)) {{
        alpha <- res$res_opt$alpha
    }} else {{
        alpha <- res$alpha
    }}
}} else {{
    alpha <- res$alpha
}}

cat("\\nExtracted alpha length:", length(alpha), "\\n")
print(alpha)

# Save to JSON
write_json(list(alpha = as.numeric(alpha)), "{out_file}")
"""
        
        r_file = os.path.join(tmpdir, "test.R")
        with open(r_file, 'w') as f:
            f.write(r_code)
        
        # Run R
        result = subprocess.run(['Rscript', r_file], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"R stderr: {result.stderr}")
            raise RuntimeError(f"R execution failed: {result.stderr}")
        
        # Load result
        with open(out_file, 'r') as f:
            data = json.load(f)
        
        return np.array(data['alpha'])


def run_python_single_lambda(X, Y, maxdeg, npc, single_lambda, norm, center=True,
                            max_iter=500, tol=1e-1, step_factor=0.5, crit='grad'):
    """Call Python hapc() and extract alpha coefficients."""
    
    if norm == "sv":
        result = single_pcghal(X, Y, maxdeg, npc, single_lambda,
                              predict=None, center=center, approx=False,
                              verbose=False, max_iter=max_iter, tol=tol)
        return result.alpha
    elif norm == "1":
        result = single_lambda_fit(X, Y, maxdeg, npc, single_lambda,
                                  predict=None, center=center, approx=False, l1=True)
        return result.alpha
    elif norm == "2":
        result = single_lambda_fit(X, Y, maxdeg, npc, single_lambda,
                                  predict=None, center=center, approx=False, l1=False)
        return result.alpha
    else:
        raise ValueError(f"Unknown norm: {norm}")


def compare_alphas(alpha_r, alpha_py, norm, tol_rel=1e-2, tol_abs=1e-4):
    """
    Compare R and Python alphas (allowing for sign flips per element).
    
    Returns:
        match: bool - whether alphas match up to sign
        corr: float - maximum absolute correlation
        details: dict - detailed comparison info
    """
    if len(alpha_r) == 0 or len(alpha_py) == 0:
        return False, 0.0, {
            "error": f"Empty alpha vector: R length {len(alpha_r)}, Python length {len(alpha_py)}"
        }
    
    if len(alpha_r) != len(alpha_py):
        return False, 0.0, {"error": f"Length mismatch: {len(alpha_r)} vs {len(alpha_py)}"}
    
    # Try multiple sign flip patterns
    alpha_r_arr = np.array(alpha_r)
    alpha_py_arr = np.array(alpha_py)
    
    # Check absolute values (allowing element-wise sign flips)
    match_abs = np.allclose(np.abs(alpha_r_arr), np.abs(alpha_py_arr), atol=tol_abs, rtol=tol_rel)
    
    # Compute correlation of absolute values
    corr_abs = np.corrcoef(np.abs(alpha_r_arr), np.abs(alpha_py_arr))[0, 1]
    
    # Check if signs match for each element
    signs_match = np.sign(alpha_r_arr) == np.sign(alpha_py_arr)
    n_matching_signs = np.sum(signs_match)
    
    # Check if same sign or opposite sign
    match_same_sign = np.allclose(alpha_r_arr, alpha_py_arr, atol=tol_abs, rtol=tol_rel)
    match_flip_sign = np.allclose(alpha_r_arr, -alpha_py_arr, atol=tol_abs, rtol=tol_rel)
    match_mixed = match_abs and not match_same_sign and not match_flip_sign
    
    # Match if absolute values match (regardless of signs)
    match = match_abs
    
    abs_diff = np.abs(np.abs(alpha_r_arr) - np.abs(alpha_py_arr))
    rel_err = abs_diff / (np.abs(alpha_r_arr) + 1e-12)
    
    details = {
        "n_coef": len(alpha_r),
        "match_same_sign": bool(match_same_sign),
        "match_flip_sign": bool(match_flip_sign),
        "match_mixed_signs": bool(match_mixed),
        "correlation_abs": float(corr_abs),
        "n_matching_signs": int(n_matching_signs),
        "max_abs_diff": float(abs_diff.max()),
        "max_rel_error": float(rel_err.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "alpha_r_norm": float(np.linalg.norm(alpha_r_arr)),
        "alpha_py_norm": float(np.linalg.norm(alpha_py_arr)),
    }
    
    return match, corr_abs, details


def test_single_lambda_norm_sv():
    """Test norm='sv' (gradient descent PC-GHAL)."""
    print("\n" + "="*70)
    print("TEST: norm='sv' (PC-GHAL Gradient Descent)")
    print("="*70)
    
    X, Y = generate_test_data(seed=42, n=80, p=4, maxdeg=2)
    maxdeg, npc, single_lambda = 2, 4, 0.01
    
    print(f"Data: n={X.shape[0]}, p={X.shape[1]}, maxdeg={maxdeg}, npc={npc}, lambda={single_lambda}")
    
    try:
        alpha_r = run_r_single_lambda(X, Y, maxdeg, npc, single_lambda, norm="sv")
        print(f"✓ R executed successfully, alpha shape: {alpha_r.shape}")
        print(f"  R alpha: {alpha_r}")
    except Exception as e:
        print(f"✗ R failed: {e}")
        return False
    
    try:
        alpha_py = run_python_single_lambda(X, Y, maxdeg, npc, single_lambda, norm="sv")
        print(f"✓ Python executed successfully, alpha shape: {alpha_py.shape}")
        print(f"  Python alpha: {alpha_py}")
    except Exception as e:
        print(f"✗ Python failed: {e}")
        return False
    
    match, corr, details = compare_alphas(alpha_r, alpha_py, "sv")
    
    print(f"\nComparison:")
    print(f"  Match (up to sign): {match}")
    if "error" not in details:
        print(f"  Correlation (absolute values): {corr:.6f}")
        print(f"  Max absolute difference: {details['max_abs_diff']:.2e}")
        print(f"  Mean absolute difference: {details['mean_abs_diff']:.2e}")
        print(f"  R norm: {details['alpha_r_norm']:.6f}, Python norm: {details['alpha_py_norm']:.6f}")
        print(f"  Sign match info:")
        print(f"    - Same sign: {details['match_same_sign']}")
        print(f"    - All opposite sign: {details['match_flip_sign']}")
        print(f"    - Mixed signs: {details['match_mixed_signs']}")
        print(f"    - Coefficients with matching signs: {details['n_matching_signs']}/{details['n_coef']}")
    
    if match:
        if details['match_flip_sign']:
            print(f"  ✓ Alphas match with UNIFORM SIGN FLIP (SVD ambiguity)")
        elif details['match_mixed_signs']:
            print(f"  ✓ Alphas match with MIXED SIGN FLIPS (element-wise ambiguity)")
        else:
            print(f"  ✓ Alphas match with SAME SIGN")
        return True
    else:
        print(f"  ✗ Alphas DO NOT MATCH")
        if "error" not in details:
            print(f"    Error details: max_abs_diff={details['max_abs_diff']:.2e}")
        return False


def test_single_lambda_norm_2():
    """Test norm='2' (Ridge regression L2 penalty)."""
    print("\n" + "="*70)
    print("TEST: norm='2' (Ridge L2 Penalty)")
    print("="*70)
    
    X, Y = generate_test_data(seed=43, n=80, p=4, maxdeg=2)
    maxdeg, npc, single_lambda = 2, 4, 0.01
    
    print(f"Data: n={X.shape[0]}, p={X.shape[1]}, maxdeg={maxdeg}, npc={npc}, lambda={single_lambda}")
    
    try:
        alpha_r = run_r_single_lambda(X, Y, maxdeg, npc, single_lambda, norm="2")
        print(f"✓ R executed successfully, alpha shape: {alpha_r.shape}")
        print(f"  R alpha: {alpha_r}")
    except Exception as e:
        print(f"✗ R failed: {e}")
        return False
    
    try:
        alpha_py = run_python_single_lambda(X, Y, maxdeg, npc, single_lambda, norm="2")
        print(f"✓ Python executed successfully, alpha shape: {alpha_py.shape}")
        print(f"  Python alpha: {alpha_py}")
    except Exception as e:
        print(f"✗ Python failed: {e}")
        return False
    
    match, corr, details = compare_alphas(alpha_r, alpha_py, "2")
    
    print(f"\nComparison:")
    print(f"  Match (up to sign): {match}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Max absolute difference: {details['max_abs_diff']:.2e}")
    print(f"  Mean absolute difference: {details['mean_abs_diff']:.2e}")
    print(f"  R norm: {details['alpha_r_norm']:.6f}, Python norm: {details['alpha_py_norm']:.6f}")
    
    if match:
        if details['match_flip_sign']:
            print(f"  ✓ Alphas match with SIGN FLIP (expected due to SVD)")
        else:
            print(f"  ✓ Alphas match with SAME SIGN")
        return True
    else:
        print(f"  ✗ Alphas DO NOT MATCH")
        return False


def test_single_lambda_norm_1():
    """Test norm='1' (LASSO L1 penalty)."""
    print("\n" + "="*70)
    print("TEST: norm='1' (LASSO L1 Penalty)")
    print("="*70)
    
    X, Y = generate_test_data(seed=44, n=80, p=4, maxdeg=2)
    maxdeg, npc, single_lambda = 2, 4, 0.001
    
    print(f"Data: n={X.shape[0]}, p={X.shape[1]}, maxdeg={maxdeg}, npc={npc}, lambda={single_lambda}")
    
    try:
        alpha_r = run_r_single_lambda(X, Y, maxdeg, npc, single_lambda, norm="1")
        print(f"✓ R executed successfully, alpha shape: {alpha_r.shape}")
        print(f"  R alpha: {alpha_r}")
    except Exception as e:
        print(f"✗ R failed: {e}")
        return False
    
    try:
        alpha_py = run_python_single_lambda(X, Y, maxdeg, npc, single_lambda, norm="1")
        print(f"✓ Python executed successfully, alpha shape: {alpha_py.shape}")
        print(f"  Python alpha: {alpha_py}")
    except Exception as e:
        print(f"✗ Python failed: {e}")
        return False
    
    match, corr, details = compare_alphas(alpha_r, alpha_py, "1")
    
    print(f"\nComparison:")
    print(f"  Match (up to sign): {match}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Max absolute difference: {details['max_abs_diff']:.2e}")
    print(f"  Mean absolute difference: {details['mean_abs_diff']:.2e}")
    print(f"  R norm: {details['alpha_r_norm']:.6f}, Python norm: {details['alpha_py_norm']:.6f}")
    
    if match:
        if details['match_flip_sign']:
            print(f"  ✓ Alphas match with SIGN FLIP (expected due to SVD)")
        else:
            print(f"  ✓ Alphas match with SAME SIGN")
        return True
    else:
        print(f"  ✗ Alphas DO NOT MATCH")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("R vs Python Alpha Coefficient Comparison")
    print("Testing fixed-lambda single fitting routines")
    print("="*70)
    
    results = {}
    results["norm='sv'"] = test_single_lambda_norm_sv()
    results["norm='2'"] = test_single_lambda_norm_2()
    results["norm='1'"] = test_single_lambda_norm_1()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for norm, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {norm}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    exit(0 if all_passed else 1)
