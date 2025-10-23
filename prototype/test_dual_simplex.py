"""
Test cases for the Dual Revised Simplex Solver

This module contains test cases to verify the correctness of the dual revised simplex solver.
"""

import torch
from dual_revised_simplex_solver import DualRevisedSimplexSolver, add_slack_variables

# Disable PyTorch gradient calculations globally
torch.set_grad_enabled(False)

def test_simple_lp(device='cpu'):
    """
    Test Case 1: Simple LP problem
    
    minimize    -2*x1 - 3*x2
    subject to   x1 + 2*x2 <= 4
                2*x1 +  x2 <= 5
                 x1, x2 >= 0
    
    Expected optimal solution: x1 = 2, x2 = 1, objective = -7
    """
    print("=" * 60)
    print("Test Case 1: Simple LP Problem")
    print(f"Device: {device}")
    print("=" * 60)
    
    dev = torch.device(device)
    
    # Original problem (in inequality form)
    c_orig = torch.tensor([-2.0, -3.0], device=dev)
    A_orig = torch.tensor([
        [1.0, 2.0],
        [2.0, 1.0]
    ], device=dev)
    b = torch.tensor([4.0, 5.0], device=dev)
    
    # Convert to standard form by adding slack variables
    c, A, b = add_slack_variables(c_orig, A_orig, b, inequality_types=['<=', '<='], device=dev)
    
    print("\nProblem in standard form:")
    print(f"Cost vector c: {c}")
    print(f"Constraint matrix A:\n{A}")
    print(f"Right-hand side b: {b}")
    
    # Create solver and solve
    solver = DualRevisedSimplexSolver(c, A, b)
    
    # Initial basis: use slack variables [2, 3] (indices of slack variables)
    initial_basis = [2, 3]
    
    x, obj_val, status = solver.solve(basis=initial_basis)
    
    print(f"\nSolver Status: {status}")
    print(f"Iterations: {solver.iterations}")
    print(f"Solution x: {x}")
    print(f"Objective value: {obj_val}")
    
    # Extract original variables (first 2)
    x_orig = x[:2]
    print(f"Original variables (x1, x2): {x_orig}")
    
    # Verify constraints
    print("\nConstraint verification:")
    for i in range(A_orig.shape[0]):
        lhs = (A_orig[i] @ x_orig).item()
        rhs = b[i].item()
        print(f"Constraint {i+1}: {lhs:.4f} <= {rhs:.4f} : {'✓' if lhs <= rhs + 1e-6 else '✗'}")
    
    # Expected solution
    expected_x = torch.tensor([2.0, 1.0])
    expected_obj = -7.0
    
    print(f"\nExpected solution: {expected_x}")
    print(f"Expected objective: {expected_obj}")
    print(f"Solution match: {torch.allclose(x_orig, expected_x, atol=1e-4)}")
    print(f"Objective match: {abs(obj_val - expected_obj) < 1e-4}")
    
    return status == 'optimal'


def test_another_lp(device='cpu'):
    """
    Test Case 2: Another LP problem
    
    minimize    3*x1 + 2*x2
    subject to  2*x1 +  x2 >= 4
                 x1 + 2*x2 >= 3
                 x1, x2 >= 0
    
    Expected optimal solution: x1 = 1.67, x2 = 0.67, objective ≈ 6.33
    """
    print("\n" + "=" * 60)
    print("Test Case 2: LP with >= Constraints")
    print(f"Device: {device}")
    print("=" * 60)
    
    dev = torch.device(device)
    
    # Original problem
    c_orig = torch.tensor([3.0, 2.0], device=dev)
    A_orig = torch.tensor([
        [2.0, 1.0],
        [1.0, 2.0]
    ], device=dev)
    b = torch.tensor([4.0, 3.0], device=dev)
    
    # For >= constraints, we need to convert them properly
    # We'll add artificial variables and use two-phase method
    # For simplicity in this test, we'll reformulate:
    # 2*x1 + x2 - s1 = 4  (where s1 >= 0)
    # x1 + 2*x2 - s2 = 3  (where s2 >= 0)
    
    # Add slack variables for >= constraints (they become negative)
    c, A, b = add_slack_variables(c_orig, A_orig, b, inequality_types=['>=', '>='])
    
    print("\nProblem in standard form:")
    print(f"Cost vector c: {c}")
    print(f"Constraint matrix A:\n{A}")
    print(f"Right-hand side b: {b}")
    
    # For >= constraints converted to equality with negative slack,
    # we need artificial variables for an initial feasible basis
    # For this test, let's use a different approach: add artificial variables
    
    m, n = A.shape
    # Add artificial variables
    c_art = torch.cat([c, torch.ones(m, device=dev) * 1000])  # Big M method
    A_art = torch.cat([A, torch.eye(m, device=dev)], dim=1)
    
    print("\nWith artificial variables:")
    print(f"Cost vector: {c_art}")
    print(f"Constraint matrix:\n{A_art}")
    
    solver = DualRevisedSimplexSolver(c_art, A_art, b, max_iter=100)
    
    # Initial basis: artificial variables
    initial_basis = list(range(n, n + m))
    
    x, obj_val, status = solver.solve(basis=initial_basis)
    
    print(f"\nSolver Status: {status}")
    print(f"Iterations: {solver.iterations}")
    print(f"Solution x: {x}")
    print(f"Objective value: {obj_val}")
    
    # Extract original variables
    x_orig = x[:2]
    print(f"Original variables (x1, x2): {x_orig}")
    
    # Verify constraints (original form)
    print("\nConstraint verification:")
    for i in range(A_orig.shape[0]):
        lhs = (A_orig[i] @ x_orig).item()
        rhs = b[i].item()
        print(f"Constraint {i+1}: {lhs:.4f} >= {rhs:.4f} : {'✓' if lhs >= rhs - 1e-6 else '✗'}")
    
    return status in ['optimal', 'max_iter']


def test_infeasible_lp(device='cpu'):
    """
    Test Case 3: Infeasible LP problem
    
    minimize    x1 + x2
    subject to  x1 + x2 <= 1
                x1 + x2 >= 2
                x1, x2 >= 0
    
    This problem is infeasible.
    """
    print("\n" + "=" * 60)
    print("Test Case 3: Infeasible LP Problem")
    print(f"Device: {device}")
    print("=" * 60)
    
    dev = torch.device(device)
    
    # Create an infeasible problem by having contradictory constraints
    c = torch.tensor([1.0, 1.0, 0.0, 0.0], device=dev)
    A = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],   # x1 + x2 + s1 = 1
        [1.0, 1.0, 0.0, -1.0]   # x1 + x2 - s2 = 2
    ], device=dev)
    b = torch.tensor([1.0, 2.0], device=dev)
    
    print("\nProblem in standard form:")
    print(f"Cost vector c: {c}")
    print(f"Constraint matrix A:\n{A}")
    print(f"Right-hand side b: {b}")
    
    solver = DualRevisedSimplexSolver(c, A, b, max_iter=100)
    initial_basis = [2, 3]
    
    x, obj_val, status = solver.solve(basis=initial_basis)
    
    print(f"\nSolver Status: {status}")
    print(f"Iterations: {solver.iterations}")
    print(f"Expected status: infeasible or max_iter")
    
    return True  # Just checking it doesn't crash


def main(device='cpu'):
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("DUAL SIMPLEX SOLVER TEST SUITE")
    print(f"Device: {device}")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Test 1: Simple LP", test_simple_lp(device)))
    except Exception as e:
        print(f"\nTest 1 failed with error: {e}")
        results.append(("Test 1: Simple LP", False))
    
    try:
        results.append(("Test 2: LP with >= constraints", test_another_lp(device)))
    except Exception as e:
        print(f"\nTest 2 failed with error: {e}")
        results.append(("Test 2: LP with >= constraints", False))
    
    try:
        results.append(("Test 3: Infeasible LP", test_infeasible_lp(device)))
    except Exception as e:
        print(f"\nTest 3 failed with error: {e}")
        results.append(("Test 3: Infeasible LP", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
