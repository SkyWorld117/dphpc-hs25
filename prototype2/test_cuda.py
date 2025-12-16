"""
Test suite for the PyTorch dual simplex solver.

Verifies correctness by:
1. Testing simple LP problems with known analytical solutions
2. Comparing results against scipy.optimize.linprog
3. Testing the full Delaunay interpolation workflow
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from scipy.optimize import linprog
import sys
import time

from dualsimplex import (
    solve_lp, DualSimplexError, 
    set_device, get_device, to_numpy, from_numpy,
    DEVICE, DTYPE
)

DEVICE_STR = "cuda"
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

set_device(DEVICE_STR)
print(f"Device set to: {get_device()}\n")

N_VARS = 50
N_CONSTRAINTS = 100

def test_gpu_performance():
    """
    Test GPU performance with larger problems (if GPU available).
    """
    print("=" * 60)
    print("Test 6: GPU Performance Benchmark")
    print("=" * 60)
    
    # Test with a moderately sized problem
    n_vars = N_VARS
    n_constraints = N_CONSTRAINTS
    
    torch.manual_seed(123)
    
    # Generate problem
    A_ub = torch.randn(n_constraints, n_vars, dtype=DTYPE, device=get_device())
    x_feasible = torch.abs(torch.randn(n_vars, dtype=DTYPE, device=get_device()))
    slack = torch.abs(torch.randn(n_constraints, dtype=DTYPE, device=get_device())) + 0.1
    B = A_ub @ x_feasible + slack
    
    A = torch.vstack([A_ub, -torch.eye(n_vars, dtype=DTYPE, device=get_device())])
    B_full = torch.cat([B, torch.zeros(n_vars, dtype=DTYPE, device=get_device())])
    C = torch.randn(n_vars, dtype=DTYPE, device=get_device())
    
    try:
        # Warm up
        _ = solve_lp(A, B_full, C)
        
        # Synchronize before timing
        if DEVICE_STR == "cuda":
            torch.cuda.synchronize()
        
        # Time GPU solve

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            start = time.perf_counter()
            for _ in range(10):
                X, Y, basis, ierr = solve_lp(A, B_full, C)
            
            if DEVICE_STR == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        print(f"Problem size: {n_vars} variables, {n_constraints + n_vars} constraints")
        print(f"10 solves completed in {elapsed:.4f} seconds")
        print(f"Average time per solve: {elapsed/10*1000:.2f} ms")
        print("✓ PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

if __name__ == "__main__":
    sys.exit(test_gpu_performance())
