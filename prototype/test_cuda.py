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
import argparse

from dualsimplex import (
    solve_lp, DualSimplexError, 
    set_device, get_device, to_numpy, from_numpy,
    DEVICE, DTYPE
)

DEVICE_STR = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

set_device(DEVICE_STR)
print(f"Device set to: {get_device()}\n")

N_VARS = 200 * 4
N_CONSTRAINTS = 400 * 3

PROFILE = False

def test_gpu_performance():
    """
    Test GPU performance with larger problems (if GPU available).
    """
    print("=" * 60)
    print("GPU Performance Benchmark")
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

    # Print sparsity of A
    sparsity = 1.0 - (A != 0).sum().item() / (A.numel())
    print(f"Sparsity of constraint matrix A: {sparsity:.2%}")
    
    try:
        # Warm up
        _ = solve_lp(A, B_full, C)
        
        # Synchronize before timing
        if DEVICE_STR == "cuda":
            torch.cuda.synchronize()
        
        # Time GPU solve

        if PROFILE:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                start = time.perf_counter()
                for _ in range(10):
                    X, Y, basis, ierr = solve_lp(A, B_full, C)
                
                if DEVICE_STR == "cuda":
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            start = time.perf_counter()
            for _ in range(10):
                X, Y, basis, ierr = solve_lp(A, B_full, C)
            
            if DEVICE_STR == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
        
        print(f"Problem size: {n_vars} variables, {n_constraints + n_vars} constraints")
        print(f"10 solves completed in {elapsed:.4f} seconds")
        print(f"Average time per solve: {elapsed/10*1000:.2f} ms")
        print("✓ PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def generate_sparse_lp_system(n_vars, n_constraints, sparsity=0.1, seed=None):
    """
    Generate a sparse LP system for testing the dual simplex solver.
    
    Creates a sparse constraint matrix A where only a fraction of entries are non-zero.
    The system is constructed to be feasible.
    
    Parameters
    ----------
    n_vars : int
        Number of decision variables
    n_constraints : int
        Number of inequality constraints (before adding non-negativity)
    sparsity : float
        Fraction of non-zero entries in A (default: 0.1 = 10% non-zero)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    A : torch.Tensor
        Full constraint matrix (n_constraints + n_vars, n_vars) including non-negativity
    B : torch.Tensor
        Right-hand side vector
    C : torch.Tensor
        Objective function coefficients
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = get_device()
    
    # Generate sparse constraint matrix using a mask
    # Create random values
    A_dense = torch.randn(n_constraints, n_vars, dtype=DTYPE, device=device)
    
    # Create sparsity mask - only keep 'sparsity' fraction of entries
    mask = torch.rand(n_constraints, n_vars, device=device) < sparsity
    
    # Ensure each row has at least one non-zero entry
    for i in range(n_constraints):
        if not mask[i].any():
            # Set at least one random entry to True
            j = torch.randint(0, n_vars, (1,)).item()
            mask[i, j] = True
    
    # Ensure each column has at least one non-zero entry (for bounded problem)
    for j in range(n_vars):
        if not mask[:, j].any():
            i = torch.randint(0, n_constraints, (1,)).item()
            mask[i, j] = True
    
    # Apply mask to create sparse matrix (stored as dense for now)
    A_ub = A_dense * mask.float()
    
    # Generate a feasible solution
    x_feasible = torch.abs(torch.randn(n_vars, dtype=DTYPE, device=device)) + 0.1
    
    # Compute b with slack to ensure feasibility
    slack = torch.abs(torch.randn(n_constraints, dtype=DTYPE, device=device)) + 0.1
    B = A_ub @ x_feasible + slack
    
    # Add non-negativity constraints: -x <= 0
    A = torch.vstack([A_ub, -torch.eye(n_vars, dtype=DTYPE, device=device)])
    B_full = torch.cat([B, torch.zeros(n_vars, dtype=DTYPE, device=device)])
    
    # Random objective coefficients
    C = torch.randn(n_vars, dtype=DTYPE, device=device)
    
    # Compute actual sparsity (excluding non-negativity constraints)
    actual_nnz = mask.sum().item()
    actual_sparsity = actual_nnz / (n_constraints * n_vars)
    
    print(f"Generated sparse LP: {n_vars} vars, {n_constraints} constraints")
    print(f"Requested sparsity: {sparsity:.1%}, Actual sparsity: {actual_sparsity:.1%}")
    print(f"Non-zero entries in A_ub: {actual_nnz} / {n_constraints * n_vars}")
    
    return A, B_full, C


def test_sparse_gpu_performance():
    """
    Test GPU performance with sparse LP problems.
    """
    print("=" * 60)
    print("Test: Sparse LP GPU Performance Benchmark")
    print("=" * 60)
    
    n_vars = N_VARS
    n_constraints = N_CONSTRAINTS
    sparsity = 0.15  # 15% non-zero entries
    
    # Generate sparse problem
    A, B, C = generate_sparse_lp_system(n_vars, n_constraints, sparsity=sparsity, seed=456)
    
    try:
        # Warm up
        _ = solve_lp(A, B, C)
        
        # Synchronize before timing
        if DEVICE_STR == "cuda":
            torch.cuda.synchronize()
        
        # Time GPU solve
        if PROFILE:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                start = time.perf_counter()
                for _ in range(10):
                    X, Y, basis, ierr = solve_lp(A, B, C)
                
                if DEVICE_STR == "cuda":
                    torch.cuda.synchronize()
                
                elapsed = time.perf_counter() - start

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        else:
            start = time.perf_counter()
            for _ in range(10):
                X, Y, basis, ierr = solve_lp(A, B, C)
            
            if DEVICE_STR == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
        
        print(f"Problem size: {n_vars} variables, {n_constraints + n_vars} constraints")
        print(f"10 solves completed in {elapsed:.4f} seconds")
        print(f"Average time per solve: {elapsed/10*1000:.2f} ms")
        print("✓ PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_vars', nargs='?', type=int, default=N_VARS,
                        help=f'Number of variables (default: {N_VARS})')
    parser.add_argument('n_constraints', nargs='?', type=int, default=N_CONSTRAINTS,
                        help=f'Number of inequality constraints (default: {N_CONSTRAINTS})')
    args = parser.parse_args()

    # Override module-level defaults with CLI values
    N_VARS = args.n_vars
    N_CONSTRAINTS = args.n_constraints

    print(f"Using problem size from CLI: {N_VARS} variables, {N_CONSTRAINTS} constraints\n")

    # Uncomment to run GPU perf test instead
    # sys.exit(test_gpu_performance())
    sys.exit(test_sparse_gpu_performance())
