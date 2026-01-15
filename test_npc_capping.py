"""
Test that npc is correctly capped at n-1 when center=True
and at n when center=False.
"""

import numpy as np
import subprocess
import json
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / "python"))

from hapc.single import single_pcghal, single_lambda_fit


def test_python_npc_capping():
    """Test Python npc capping logic."""
    print("\n" + "="*70)
    print("Testing Python NPC Capping Logic")
    print("="*70)
    
    np.random.seed(42)
    n, p = 50, 5
    X = np.random.randn(n, p)
    Y = np.random.randn(n)
    maxdeg = 2
    single_lambda = 0.01
    
    # Test 1: center=True, npc > n (should cap to n-1)
    print("\nTest 1: center=True, requested npc=60 (n=50)")
    try:
        result = single_pcghal(X, Y, maxdeg, npc=60, single_lambda=single_lambda, 
                              center=True, verbose=False)
        actual_npc = len(result.alpha)
        expected_max = n - 1
        print(f"  Requested npc: 60")
        print(f"  n: {n}")
        print(f"  Actual alpha length: {actual_npc}")
        print(f"  Max expected (n-1): {expected_max}")
        assert actual_npc <= expected_max, f"Alpha length {actual_npc} exceeds n-1={expected_max}"
        print(f"  ✓ PASS: alpha capped to {actual_npc} <= {expected_max}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False
    
    # Test 2: center=False, npc > n (should cap to n)
    print("\nTest 2: center=False, requested npc=60 (n=50)")
    try:
        result = single_lambda_fit(X, Y, maxdeg, npc=60, single_lambda=single_lambda,
                                  center=False, l1=False)
        actual_npc = len(result.alpha)
        expected_max = n
        print(f"  Requested npc: 60")
        print(f"  n: {n}")
        print(f"  Actual alpha length: {actual_npc}")
        print(f"  Max expected (n): {expected_max}")
        assert actual_npc <= expected_max, f"Alpha length {actual_npc} exceeds n={expected_max}"
        print(f"  ✓ PASS: alpha capped to {actual_npc} <= {expected_max}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False
    
    # Test 3: center=True, npc=n (should cap to n-1)
    print("\nTest 3: center=True, requested npc=50 (n=50, should cap to 49)")
    try:
        result = single_pcghal(X, Y, maxdeg, npc=50, single_lambda=single_lambda,
                              center=True, verbose=False)
        actual_npc = len(result.alpha)
        expected_max = n - 1
        print(f"  Requested npc: 50")
        print(f"  n: {n}")
        print(f"  Actual alpha length: {actual_npc}")
        print(f"  Max expected (n-1): {expected_max}")
        assert actual_npc <= expected_max, f"Alpha length {actual_npc} exceeds n-1={expected_max}"
        print(f"  ✓ PASS: alpha capped to {actual_npc} <= {expected_max}")
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False
    
    return True


def test_r_npc_capping():
    """Test R npc capping logic."""
    print("\n" + "="*70)
    print("Testing R NPC Capping Logic")
    print("="*70)
    
    np.random.seed(42)
    n, p = 50, 5
    X = np.random.randn(n, p)
    Y = np.random.randn(n)
    maxdeg = 2
    single_lambda = 0.01
    
    with tempfile.TemporaryDirectory() as tmpdir:
        X_file = os.path.join(tmpdir, "X.csv")
        Y_file = os.path.join(tmpdir, "Y.csv")
        out_file = os.path.join(tmpdir, "result.json")
        
        np.savetxt(X_file, X, delimiter=",")
        np.savetxt(Y_file, Y, delimiter=",")
        
        # Test 1: center=TRUE, npc > n
        print("\nTest 1: center=TRUE, requested npc=60 (n=50)")
        r_code = f"""
library(hapc)
library(jsonlite)

X <- as.matrix(read.csv("{X_file}", header=FALSE))
Y <- as.numeric(read.csv("{Y_file}", header=FALSE)[[1]])

res <- hapc(X, Y, npcs=60, lambda={single_lambda}, norm="sv", 
            max_degree={maxdeg}, center=TRUE, verbose=FALSE)

alpha <- res$res_opt$alpha
cat("Alpha length:", length(alpha), "\\n")

write_json(list(alpha_len = length(alpha), n = nrow(X)), "{out_file}")
"""
        
        r_file = os.path.join(tmpdir, "test1.R")
        with open(r_file, 'w') as f:
            f.write(r_code)
        
        result = subprocess.run(['Rscript', r_file], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  ✗ R execution failed: {result.stderr}")
            return False
        
        with open(out_file, 'r') as f:
            data = json.load(f)
        
        alpha_len = data['alpha_len']
        if isinstance(alpha_len, list):
            alpha_len = alpha_len[0] if alpha_len else 0
        
        expected_max = n - 1
        print(f"  Requested npc: 60")
        print(f"  n: {n}")
        print(f"  Actual alpha length: {alpha_len}")
        print(f"  Max expected (n-1): {expected_max}")
        
        if alpha_len > expected_max:
            print(f"  ✗ FAIL: alpha length {alpha_len} exceeds n-1={expected_max}")
            return False
        else:
            print(f"  ✓ PASS: alpha capped to {alpha_len} <= {expected_max}")
        
        # Test 2: center=FALSE, npc > n
        print("\nTest 2: center=FALSE, requested npc=60 (n=50)")
        r_code = f"""
library(hapc)
library(jsonlite)

X <- as.matrix(read.csv("{X_file}", header=FALSE))
Y <- as.numeric(read.csv("{Y_file}", header=FALSE)[[1]])

res <- hapc(X, Y, npcs=60, lambda={single_lambda}, norm="2",
            max_degree={maxdeg}, center=FALSE, verbose=FALSE)

alpha <- res$alpha
cat("Alpha length:", length(alpha), "\\n")

write_json(list(alpha_len = length(alpha), n = nrow(X)), "{out_file}")
"""
        
        r_file = os.path.join(tmpdir, "test2.R")
        with open(r_file, 'w') as f:
            f.write(r_code)
        
        result = subprocess.run(['Rscript', r_file], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  ✗ R execution failed: {result.stderr}")
            return False
        
        with open(out_file, 'r') as f:
            data = json.load(f)
        
        alpha_len = data['alpha_len']
        if isinstance(alpha_len, list):
            alpha_len = alpha_len[0] if alpha_len else 0
        
        expected_max = n
        print(f"  Requested npc: 60")
        print(f"  n: {n}")
        print(f"  Actual alpha length: {alpha_len}")
        print(f"  Max expected (n): {expected_max}")
        
        if alpha_len > expected_max:
            print(f"  ✗ FAIL: alpha length {alpha_len} exceeds n={expected_max}")
            return False
        else:
            print(f"  ✓ PASS: alpha capped to {alpha_len} <= {expected_max}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NPC Capping Verification Tests")
    print("="*70)
    
    py_pass = test_python_npc_capping()
    r_pass = test_r_npc_capping()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Python NPC capping: {'✓ PASS' if py_pass else '✗ FAIL'}")
    print(f"R NPC capping: {'✓ PASS' if r_pass else '✗ FAIL'}")
    
    if py_pass and r_pass:
        print(f"\n✓ ALL TESTS PASSED: NPC capping working correctly in both R and Python")
        exit(0)
    else:
        print(f"\n✗ SOME TESTS FAILED")
        exit(1)
