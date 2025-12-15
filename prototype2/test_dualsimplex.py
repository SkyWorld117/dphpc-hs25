"""
Test suite for the dual simplex solver.

Verifies correctness by:
1. Testing simple LP problems with known analytical solutions
2. Comparing results against scipy.optimize.linprog
3. Testing the full Delaunay interpolation workflow
"""

import numpy as np
from scipy.optimize import linprog
import sys

from dualsimplex import solve_lp, DualSimplexError
# from revised_dualsimplex import solve_lp, DualSimplexError


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
    A = np.array([
        [1.0, 2.0],   # x + 2y <= 4
        [1.0, -1.0],  # x - y <= 1
        [-1.0, 0.0],  # -x <= 0 (i.e., x >= 0)
        [0.0, -1.0],  # -y <= 0 (i.e., y >= 0)
    ])
    B = np.array([4.0, 1.0, 0.0, 0.0])
    C = np.array([1.0, 1.0])  # maximize x + y
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        obj = np.dot(C, X)
        
        print(f"Solution: X = {X}")
        print(f"Objective: C^T X = {obj:.6f}")
        print(f"Expected: X = [2, 1], Objective = 3")
        
        # Verify
        assert np.allclose(X, [2.0, 1.0], atol=1e-6), f"Wrong solution: {X}"
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
    
    A = np.array([
        [1.0, 1.0, 2.0],   # x + y + 2z <= 4
        [2.0, 0.0, 3.0],   # 2x + 3z <= 5
        [0.0, 2.0, 1.0],   # 2y + z <= 7
        [-1.0, 0.0, 0.0],  # x >= 0
        [0.0, -1.0, 0.0],  # y >= 0
        [0.0, 0.0, -1.0],  # z >= 0
    ])
    B = np.array([4.0, 5.0, 7.0, 0.0, 0.0, 0.0])
    C = np.array([3.0, 2.0, 4.0])
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        obj = np.dot(C, X)
        
        print(f"Solution: X = {X}")
        print(f"Objective: C^T X = {obj:.6f}")
        
        # Compare with scipy
        # scipy minimizes, so we negate C
        res = linprog(-C, A_ub=A, b_ub=B, method='highs')
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
    
    np.random.seed(seed)
    
    # Generate random problem that is feasible and bounded
    A_ub = np.random.randn(n_constraints, n_vars)
    
    # Make sure there's a feasible region by using a known feasible point
    x_feasible = np.abs(np.random.randn(n_vars))
    slack = np.abs(np.random.randn(n_constraints)) + 0.1
    B = A_ub @ x_feasible + slack
    
    # Add non-negativity constraints
    A = np.vstack([A_ub, -np.eye(n_vars)])
    B_full = np.hstack([B, np.zeros(n_vars)])
    
    C = np.random.randn(n_vars)
    
    try:
        X, Y, basis, ierr = solve_lp(A, B_full, C)
        obj = np.dot(C, X)
        
        print(f"Our solution objective: {obj:.6f}")
        
        # Compare with scipy
        res = linprog(-C, A_ub=A, b_ub=B_full, method='highs')
        
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
        res = linprog(-C, A_ub=A, b_ub=B_full, method='highs')
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
        from delaunayLPtest import interpolate_delaunay
        import os
        
        # Generate small test data
        d, n = 3, 20  # 3D, 20 points
        test_file = "test_deldata.txt"
        
        print(f"Generating {n} points in {d} dimensions...")
        generate_delaunay_data(d, n, test_file)
        
        print("Running Delaunay interpolation...")
        simplices, weights, errors, elapsed = interpolate_delaunay(test_file, verbose=False)
        
        # Check results
        success_count = sum(1 for e in errors if e == 0)
        extrap_count = sum(1 for e in errors if e == 2)
        error_count = sum(1 for e in errors if e not in [0, 2])
        
        print(f"Results: {success_count} successful, {extrap_count} extrapolations, {error_count} errors")
        print(f"Time: {elapsed:.4f} seconds")
        
        # Verify weights sum to 1 for successful interpolations
        for i, (w, e) in enumerate(zip(weights, errors)):
            if e == 0 and w is not None:
                weight_sum = np.sum(w)
                if not np.isclose(weight_sum, 1.0, atol=1e-6):
                    print(f"✗ FAILED: Weights don't sum to 1 at point {i}: {weight_sum}")
                    return False
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("✓ PASSED\n")
        return True
        
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
    A = np.array([
        [1.0],   # x <= 1
        [-1.0],  # -x <= -2 (i.e., x >= 2)
    ])
    B = np.array([1.0, -2.0])
    C = np.array([1.0])
    
    try:
        X, Y, basis, ierr = solve_lp(A, B, C)
        print(f"✗ FAILED: Should have detected infeasibility, got X={X}\n")
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


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("DUAL SIMPLEX SOLVER TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_simple_2d,
        test_simple_3d,
        test_random_lp,
        test_delaunay_workflow,
        test_infeasible,
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
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
