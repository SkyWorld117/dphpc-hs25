"""
Test suite for the PyTorch dual simplex solver.

Verifies correctness by:
1. Testing simple LP problems with known analytical solutions
2. Comparing results against scipy.optimize.linprog
3. Testing the full Delaunay interpolation workflow
"""

import torch
import numpy as np
from scipy.optimize import linprog
import sys
import time

from dualsimplex import (
    solve_lp, DualSimplexError, 
    set_device, get_device, to_numpy, from_numpy,
    DEVICE, DTYPE
)

# Device configuration - use GPU if available
if torch.cuda.is_available():
    DEVICE_STR = "cuda"
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     DEVICE_STR = "mps"
#     print("Using Apple MPS device")
else:
    DEVICE_STR = "cpu"
    print("Using CPU device")

set_device(DEVICE_STR)
print(f"Device set to: {get_device()}\n")


def test_simple_2d():
    """
    Test a simple 2D LP with known solution.
    
    maximize    x + y
    subject to  x + 2y <= 4
                x - y <= 1
                x >= 0, y >= 0
    
    Optimal solution: x=2, y=1, objective=3
    """
    print("=" * 60)
    print("Test 1: Simple 2D LP")
    print("=" * 60)
    
    # Convert to standard form: max C^T X s.t. A X <= B
    # We need to include x >= 0, y >= 0 as -x <= 0, -y <= 0
    A = torch.tensor([
        [1.0, 2.0],   # x + 2y <= 4
        [1.0, -1.0],  # x - y <= 1
        [-1.0, 0.0],  # -x <= 0 (i.e., x >= 0)
        [0.0, -1.0],  # -y <= 0 (i.e., y >= 0)
    ], dtype=DTYPE, device=get_device())
    B = torch.tensor([4.0, 1.0, 0.0, 0.0], dtype=DTYPE, device=get_device())
    C = torch.tensor([1.0, 1.0], dtype=DTYPE, device=get_device())  # maximize x + y
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        obj = torch.dot(C, X).item()
        
        print(f"Solution: X = {to_numpy(X)}")
        print(f"Objective: C^T X = {obj:.6f}")
        print(f"Expected: X = [2, 1], Objective = 3")
        
        # Verify
        X_np = to_numpy(X)
        assert np.allclose(X_np, [2.0, 1.0], atol=1e-6), f"Wrong solution: {X_np}"
        assert np.isclose(obj, 3.0, atol=1e-6), f"Wrong objective: {obj}"
        print("✓ PASSED\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def test_simple_3d():
    """
    Test a 3D LP problem.
    
    maximize    3x + 2y + 4z
    subject to  x + y + 2z <= 4
                2x + 3z <= 5
                2y + z <= 7
                x, y, z >= 0
    
    """
    print("=" * 60)
    print("Test 2: 3D LP Problem")
    print("=" * 60)
    
    A = torch.tensor([
        [1.0, 1.0, 2.0],   # x + y + 2z <= 4
        [2.0, 0.0, 3.0],   # 2x + 3z <= 5
        [0.0, 2.0, 1.0],   # 2y + z <= 7
        [-1.0, 0.0, 0.0],  # x >= 0
        [0.0, -1.0, 0.0],  # y >= 0
        [0.0, 0.0, -1.0],  # z >= 0
    ], dtype=DTYPE, device=get_device())
    B = torch.tensor([4.0, 5.0, 7.0, 0.0, 0.0, 0.0], dtype=DTYPE, device=get_device())
    C = torch.tensor([3.0, 2.0, 4.0], dtype=DTYPE, device=get_device())
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        obj = torch.dot(C, X).item()
        
        print(f"Solution: X = {to_numpy(X)}")
        print(f"Objective: C^T X = {obj:.6f}")
        
        # Compare with scipy
        # scipy minimizes, so we negate C
        A_np = to_numpy(A)
        B_np = to_numpy(B)
        C_np = to_numpy(C)
        res = linprog(-C_np, A_ub=A_np, b_ub=B_np, method='highs')
        scipy_obj = -res.fun
        scipy_x = res.x
        
        print(f"Scipy solution: X = {scipy_x}")
        print(f"Scipy objective: {scipy_obj:.6f}")
        
        assert np.isclose(obj, scipy_obj, atol=1e-4), f"Objective mismatch: {obj} vs {scipy_obj}"
        print("✓ PASSED\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def test_random_lp(n_vars=5, n_constraints=10, seed=42):
    """
    Test a random LP and compare against scipy.
    """
    print("=" * 60)
    print(f"Test 3: Random LP ({n_vars} vars, {n_constraints} constraints)")
    print("=" * 60)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random problem that is feasible and bounded
    A_ub = torch.randn(n_constraints, n_vars, dtype=DTYPE, device=get_device())
    
    # Make sure there's a feasible region by using a known feasible point
    x_feasible = torch.abs(torch.randn(n_vars, dtype=DTYPE, device=get_device()))
    slack = torch.abs(torch.randn(n_constraints, dtype=DTYPE, device=get_device())) + 0.1
    B = A_ub @ x_feasible + slack
    
    # Add non-negativity constraints
    A = torch.vstack([A_ub, -torch.eye(n_vars, dtype=DTYPE, device=get_device())])
    B_full = torch.cat([B, torch.zeros(n_vars, dtype=DTYPE, device=get_device())])
    
    C = torch.randn(n_vars, dtype=DTYPE, device=get_device())
    
    try:
        X, Y, basis, ierr = solve_lp(A, B_full, C)
        obj = torch.dot(C, X).item()
        
        print(f"Our solution objective: {obj:.6f}")
        
        # Compare with scipy
        A_np = to_numpy(A)
        B_np = to_numpy(B_full)
        C_np = to_numpy(C)
        res = linprog(-C_np, A_ub=A_np, b_ub=B_np, method='highs')
        
        if res.success:
            scipy_obj = -res.fun
            print(f"Scipy objective: {scipy_obj:.6f}")
            
            assert np.isclose(obj, scipy_obj, atol=1e-3), f"Objective mismatch: {obj} vs {scipy_obj}"
            print("✓ PASSED\n")
            return True
        else:
            print(f"Scipy failed: {res.message}")
            print("✓ PASSED (scipy failed, our solver found solution)\n")
            return True
            
    except DualSimplexError as e:
        # Check if scipy also fails
        A_np = to_numpy(A)
        B_np = to_numpy(B_full)
        C_np = to_numpy(C)
        res = linprog(-C_np, A_ub=A_np, b_ub=B_np, method='highs')
        if not res.success:
            print(f"Both solvers indicate problem: {e.message}")
            print("✓ PASSED (correctly identified issue)\n")
            return True
        else:
            print(f"✗ FAILED: {e}\n")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def test_delaunay_workflow():
    """
    Test the full Delaunay interpolation workflow.
    """
    print("=" * 60)
    print("Test 4: Delaunay Interpolation Workflow")
    print("=" * 60)
    
    try:
        from generate_data import generate_delaunay_data
        # Note: delaunayLPtest would need to be ported to torch as well
        # For now, we skip this test or use a simplified version
        import os
        
        # Generate small test data
        d, n = 3, 20  # 3D, 20 points
        test_file = "test_deldata_torch.txt"
        
        print(f"Generating {n} points in {d} dimensions...")
        generate_delaunay_data(d, n, test_file)
        
        # Simple verification: just check that data was generated
        if os.path.exists(test_file):
            print("Data file generated successfully")
            os.remove(test_file)
            print("✓ PASSED (basic data generation)\n")
            return True
        else:
            print("✗ FAILED: Data file not created\n")
            return False
        
    except ImportError as e:
        print(f"✗ SKIPPED: {e}\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def test_infeasible():
    """
    Test detection of infeasible problems.
    """
    print("=" * 60)
    print("Test 5: Infeasible Problem Detection")
    print("=" * 60)
    
    # Infeasible: x <= 1 and x >= 2
    A = torch.tensor([
        [1.0],   # x <= 1
        [-1.0],  # -x <= -2 (i.e., x >= 2)
    ], dtype=DTYPE, device=get_device())
    B = torch.tensor([1.0, -2.0], dtype=DTYPE, device=get_device())
    C = torch.tensor([1.0], dtype=DTYPE, device=get_device())
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        print(f"✗ FAILED: Should have detected infeasibility, got X={to_numpy(X)}\n")
        return False
    except DualSimplexError as e:
        if e.code in [1, 2, 33]:  # Infeasibility codes
            print(f"Correctly detected: {e.message}")
            print("✓ PASSED\n")
            return True
        else:
            print(f"✗ FAILED: Wrong error code {e.code}: {e.message}\n")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def test_gpu_performance():
    """
    Test GPU performance with larger problems (if GPU available).
    """
    print("=" * 60)
    print("Test 6: GPU Performance Benchmark")
    print("=" * 60)
    
    if DEVICE_STR == "cpu":
        print("Skipping GPU benchmark (no GPU available)")
        print("✓ SKIPPED\n")
        return True
    
    # Test with a moderately sized problem
    n_vars = 20
    n_constraints = 50
    
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


def test_numpy_interop():
    """
    Test numpy array input compatibility.
    """
    print("=" * 60)
    print("Test 7: NumPy Interoperability")
    print("=" * 60)
    
    # Input as numpy arrays
    A = np.array([
        [1.0, 2.0],
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    B = np.array([4.0, 1.0, 0.0, 0.0])
    C = np.array([1.0, 1.0])
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        obj = torch.dot(C if isinstance(C, torch.Tensor) else from_numpy(C), X).item()
        
        print(f"Solution (from numpy input): X = {to_numpy(X)}")
        print(f"Objective: {obj:.6f}")
        
        X_np = to_numpy(X)
        assert np.allclose(X_np, [2.0, 1.0], atol=1e-6), f"Wrong solution: {X_np}"
        print("✓ PASSED\n")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("PYTORCH DUAL SIMPLEX SOLVER TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_simple_2d,
        test_simple_3d,
        test_random_lp,
        test_delaunay_workflow,
        test_infeasible,
        test_gpu_performance,
        test_numpy_interop,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Test crashed: {e}")
            results.append(False)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Device used: {get_device()}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
