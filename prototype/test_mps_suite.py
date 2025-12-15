"""
Test Suite for MPS Files using Dual Revised Simplex Solver

This module tests the dual revised simplex solver on standard MPS benchmark files.
The MPS (Mathematical Programming System) format is a standard format for representing
linear programming problems.

MPS Test Files:
- 01_test.mps: Small test problem
- brazil3.mps: Brazilian electricity distribution problem
- Dual2_5000.mps: Large dual problem with 5000 variables
- L1_sixm250obs.mps: L1 regression problem
- Primal2_1000.mps: Large primal problem with 1000 variables
"""

import sys
import os
import time
import torch
from typing import Tuple, Dict, Any, Optional

# Add parent directory to path to import parse_mps
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parse_mps import parse_self_MPS
from dual_revised_simplex_solver import DualRevisedSimplexSolver

# Disable PyTorch gradient calculations
torch.set_grad_enabled(False)


class MPSTestResult:
    """Container for MPS test results."""
    
    def __init__(self, filename: str):
        self.filename: str = filename
        self.parse_time: float = 0.0
        self.setup_time: float = 0.0
        self.solve_time: float = 0.0
        self.total_time: float = 0.0
        self.status: Optional[str] = None
        self.objective_value: Optional[float] = None
        self.iterations: int = 0
        self.num_vars: int = 0
        self.num_constraints: int = 0
        self.num_nonzeros: int = 0
        self.error_message: Optional[str] = None
        
    def __repr__(self):
        return (f"MPSTestResult({self.filename}, status={self.status}, "
                f"obj={self.objective_value}, iter={self.iterations})")


def build_standard_form_from_mps(objective_coefficients, constraint_matrix_sparse, 
                                  right_hand_side, var_name_to_bounds, device='cpu'):
    """
    Convert MPS parsed data to standard form matrices for the dual simplex solver.
    
    Args:
        objective_coefficients: List of objective coefficients
        constraint_matrix_sparse: List of (row, col, value) tuples
        right_hand_side: List of RHS values
        var_name_to_bounds: Dictionary mapping variable names to bounds
        device: Device to run on
        
    Returns:
        c: Cost vector (torch.Tensor)
        A: Constraint matrix (torch.Tensor)
        b: RHS vector (torch.Tensor)
        metadata: Dictionary with problem information
    """
    dev = torch.device(device)
    
    # Convert objective coefficients to tensor
    c = torch.tensor(objective_coefficients, dtype=torch.float32, device=dev)
    
    # Build constraint matrix from sparse representation
    if len(constraint_matrix_sparse) == 0:
        raise ValueError("Empty constraint matrix")
    
    # Find matrix dimensions
    max_row = max(entry[0] for entry in constraint_matrix_sparse)
    max_col = max(entry[1] for entry in constraint_matrix_sparse)
    
    m = max_row + 1  # Number of constraints
    n = max_col + 1  # Number of variables (including slack)
    
    # Create dense matrix (for small/medium problems)
    # For very large problems, we might want to use sparse matrices
    A = torch.zeros((m, n), dtype=torch.float32, device=dev)
    
    for row, col, value in constraint_matrix_sparse:
        A[row, col] = value
    
    # Convert RHS to tensor
    b = torch.tensor(right_hand_side, dtype=torch.float32, device=dev)
    
    # Ensure dimensions match
    if len(b) != m:
        raise ValueError(f"Dimension mismatch: b has {len(b)} elements, but A has {m} rows")
    
    if len(c) != n:
        # Pad c with zeros if necessary (for slack variables)
        if len(c) < n:
            padding = torch.zeros(n - len(c), dtype=torch.float32, device=dev)
            c = torch.cat([c, padding])
        else:
            raise ValueError(f"Dimension mismatch: c has {len(c)} elements, but A has {n} columns")
    
    metadata = {
        'num_vars': n,
        'num_constraints': m,
        'num_nonzeros': len(constraint_matrix_sparse),
        'sparsity': 1.0 - (len(constraint_matrix_sparse) / (m * n)) if m * n > 0 else 0.0,
        'num_bounds': len(var_name_to_bounds)
    }
    
    return c, A, b, metadata


def test_mps_file(mps_path: str, device='cpu', max_iter=10000, verbose=True) -> MPSTestResult:
    """
    Test the dual simplex solver on a single MPS file.
    
    Args:
        mps_path: Path to MPS file
        device: Device to run on ('cpu' or 'cuda')
        max_iter: Maximum iterations for solver
        verbose: Whether to print detailed output
        
    Returns:
        MPSTestResult object with test results
    """
    filename = os.path.basename(mps_path)
    result = MPSTestResult(filename)
    
    if verbose:
        print("=" * 80)
        print(f"Testing MPS File: {filename}")
        print(f"Path: {mps_path}")
        print(f"Device: {device}")
        print("=" * 80)
    
    total_start = time.time()
    
    try:
        # Step 1: Parse MPS file
        if verbose:
            print("\n[1/3] Parsing MPS file...")
        parse_start = time.time()
        
        objective_coefficients, constraint_matrix_sparse, right_hand_side, var_name_to_bounds = \
            parse_self_MPS(mps_path)
        
        result.parse_time = time.time() - parse_start
        
        if verbose:
            print(f"✓ Parsing completed in {result.parse_time:.4f} seconds")
        
        # Step 2: Build standard form
        if verbose:
            print("\n[2/3] Building standard form...")
        setup_start = time.time()
        
        c, A, b, metadata = build_standard_form_from_mps(
            objective_coefficients, constraint_matrix_sparse, 
            right_hand_side, var_name_to_bounds, device=device
        )
        
        result.num_vars = metadata['num_vars']
        result.num_constraints = metadata['num_constraints']
        result.num_nonzeros = metadata['num_nonzeros']
        
        result.setup_time = time.time() - setup_start
        
        if verbose:
            print(f"✓ Standard form built in {result.setup_time:.4f} seconds")
            print(f"\nProblem statistics:")
            print(f"  Variables: {result.num_vars:,}")
            print(f"  Constraints: {result.num_constraints:,}")
            print(f"  Non-zeros: {result.num_nonzeros:,}")
            print(f"  Sparsity: {metadata['sparsity']*100:.2f}% zeros")
            print(f"  Matrix size: {result.num_constraints} x {result.num_vars}")
            memory_mb = (A.numel() + c.numel() + b.numel()) * 4 / 1024 / 1024
            print(f"  Memory usage: ~{memory_mb:.2f} MB")
        
        # Step 3: Solve with dual simplex
        if verbose:
            print(f"\n[3/3] Solving with Dual Revised Simplex (max_iter={max_iter})...")
        solve_start = time.time()
        
        solver = DualRevisedSimplexSolver(c, A, b, max_iter=max_iter, device=device)
        
        # Use slack variables as initial basis (if available)
        # Typically, the last m variables are slack variables
        m = result.num_constraints
        n = result.num_vars
        if n >= m:
            initial_basis = list(range(n - m, n))
        else:
            # Not enough variables, use whatever we have
            initial_basis = list(range(min(m, n)))
        
        x, obj_val, status = solver.solve(basis=initial_basis)
        
        result.solve_time = time.time() - solve_start
        result.status = status
        result.objective_value = obj_val
        result.iterations = solver.iterations
        
        if verbose:
            print(f"✓ Solving completed in {result.solve_time:.4f} seconds")
            print(f"\nSolution:")
            print(f"  Status: {status}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Objective value: {obj_val:.6f}")
            
            # Verify solution
            if status == 'optimal':
                # Check constraint satisfaction
                residual = torch.norm(A @ x - b).item()
                non_negativity = torch.all(x >= -1e-6).item()
                
                print(f"\nVerification:")
                print(f"  Constraint residual: {residual:.6e}")
                print(f"  Non-negativity satisfied: {non_negativity}")
                
                if residual > 1e-4:
                    print(f"  ⚠ Warning: Large constraint residual!")
                if not non_negativity:
                    print(f"  ⚠ Warning: Non-negativity violated!")
    
    except Exception as e:
        result.error_message = str(e)
        if verbose:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    result.total_time = time.time() - total_start
    
    if verbose:
        print(f"\nTotal time: {result.total_time:.4f} seconds")
        print("=" * 80)
    
    return result


def run_mps_test_suite(mps_dir: str = "../MPS_tests", device='cpu', 
                       max_iter=10000, skip_large=False):
    """
    Run the test suite on all MPS files in the directory.
    
    Args:
        mps_dir: Directory containing MPS files
        device: Device to run on ('cpu' or 'cuda')
        max_iter: Maximum iterations for solver
        skip_large: Skip large problems (> 1000 variables)
        
    Returns:
        List of MPSTestResult objects
    """
    print("\n" + "=" * 80)
    print("MPS TEST SUITE FOR DUAL REVISED SIMPLEX SOLVER")
    print("=" * 80)
    print(f"MPS Directory: {mps_dir}")
    print(f"Device: {device}")
    print(f"Max iterations: {max_iter}")
    if skip_large:
        print("Skipping large problems (> 1000 variables)")
    print("=" * 80)
    
    # Find all MPS files
    if not os.path.exists(mps_dir):
        print(f"\n✗ Error: Directory {mps_dir} does not exist!")
        return []
    
    mps_files = [f for f in os.listdir(mps_dir) if f.endswith('.mps')]
    mps_files.sort()
    
    if not mps_files:
        print(f"\n✗ Error: No MPS files found in {mps_dir}")
        return []
    
    print(f"\nFound {len(mps_files)} MPS file(s):")
    for f in mps_files:
        print(f"  - {f}")
    print()
    
    # Test each file
    results = []
    
    for mps_file in mps_files:
        mps_path = os.path.join(mps_dir, mps_file)
        
        # Optionally skip large problems during quick testing
        if skip_large and any(keyword in mps_file.lower() for keyword in ['5000', 'sixm']):
            print(f"\nSkipping {mps_file} (large problem)")
            continue
        
        result = test_mps_file(mps_path, device=device, max_iter=max_iter, verbose=True)
        results.append(result)
        print()
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results):
    """Print a summary table of all test results."""
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No results to display.")
        return
    
    # Header
    print(f"\n{'Filename':<25} {'Status':<12} {'Iter':>6} {'Obj Value':>15} {'Time (s)':>10}")
    print("-" * 80)
    
    # Results
    for result in results:
        filename = result.filename[:24]
        status = result.status or 'ERROR'
        iterations = result.iterations if result.iterations else 0
        obj_val = f"{result.objective_value:.6f}" if result.objective_value is not None else "N/A"
        total_time = f"{result.total_time:.4f}"
        
        print(f"{filename:<25} {status:<12} {iterations:>6} {obj_val:>15} {total_time:>10}")
    
    # Statistics
    print("-" * 80)
    successful = sum(1 for r in results if r.status == 'optimal')
    failed = sum(1 for r in results if r.status != 'optimal')
    total = len(results)
    
    print(f"\nTotal tests: {total}")
    print(f"Successful (optimal): {successful}")
    print(f"Failed/Non-optimal: {failed}")
    
    if successful > 0:
        avg_time = sum(r.total_time for r in results if r.status == 'optimal') / successful
        avg_iter = sum(r.iterations for r in results if r.status == 'optimal') / successful
        print(f"\nAverage time (successful): {avg_time:.4f} seconds")
        print(f"Average iterations (successful): {avg_iter:.1f}")
    
    # Problem size statistics
    print(f"\nProblem sizes:")
    for result in results:
        if result.num_vars > 0:
            print(f"  {result.filename}: {result.num_constraints} x {result.num_vars} "
                  f"({result.num_nonzeros:,} non-zeros)")


def benchmark_device_comparison(mps_file: str, mps_dir: str = "../MPS_tests", max_iter=5000):
    """
    Compare CPU vs GPU performance on a single MPS file.
    
    Args:
        mps_file: Name of MPS file to test
        mps_dir: Directory containing MPS files
        max_iter: Maximum iterations
    """
    print("\n" + "=" * 80)
    print("CPU vs GPU BENCHMARK")
    print("=" * 80)
    
    mps_path = os.path.join(mps_dir, mps_file)
    
    if not os.path.exists(mps_path):
        print(f"✗ Error: File {mps_path} does not exist!")
        return
    
    # Test on CPU
    print(f"\nTesting on CPU:")
    cpu_result = test_mps_file(mps_path, device='cpu', max_iter=max_iter, verbose=True)
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print(f"\nTesting on GPU:")
        gpu_result = test_mps_file(mps_path, device='cuda', max_iter=max_iter, verbose=True)
        
        # Compare
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        
        if cpu_result.solve_time > 0 and gpu_result.solve_time > 0:
            speedup = cpu_result.solve_time / gpu_result.solve_time
            print(f"\nSolve time speedup: {speedup:.2f}x")
            print(f"  CPU: {cpu_result.solve_time:.4f} seconds")
            print(f"  GPU: {gpu_result.solve_time:.4f} seconds")
        
        if cpu_result.total_time > 0 and gpu_result.total_time > 0:
            total_speedup = cpu_result.total_time / gpu_result.total_time
            print(f"\nTotal time speedup: {total_speedup:.2f}x")
            print(f"  CPU: {cpu_result.total_time:.4f} seconds")
            print(f"  GPU: {gpu_result.total_time:.4f} seconds")
    else:
        print("\n⚠ CUDA not available. Skipping GPU benchmark.")


def main():
    """Main entry point for the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Dual Simplex Solver on MPS files')
    parser.add_argument('--mps-dir', type=str, default='../MPS_tests',
                        help='Directory containing MPS files')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run on')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Maximum iterations for solver')
    parser.add_argument('--skip-large', action='store_true',
                        help='Skip large problems')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Run CPU vs GPU benchmark on specific MPS file')
    parser.add_argument('--file', type=str, default=None,
                        help='Test only a specific MPS file')
    
    args = parser.parse_args()
    
    # Adjust path if running from different directory
    mps_dir = args.mps_dir
    if not os.path.exists(mps_dir):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mps_dir = os.path.join(script_dir, '..', '..', 'MPS_tests')
    
    if args.benchmark:
        benchmark_device_comparison(args.benchmark, mps_dir, args.max_iter)
    elif args.file:
        mps_path = os.path.join(mps_dir, args.file)
        test_mps_file(mps_path, device=args.device, max_iter=args.max_iter, verbose=True)
    else:
        run_mps_test_suite(mps_dir, device=args.device, max_iter=args.max_iter, 
                          skip_large=args.skip_large)


if __name__ == "__main__":
    main()
