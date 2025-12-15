"""
Dual Revised Simplex Module

Refactored Python version of dualsimplex.f90 implementing the Revised Simplex method.
This version avoids dense tableau modifications in favor of matrix factorizations.

Original Author: Tyler Chang
Refactored: Gemini
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional

class DualSimplexError(Exception):
    """Exception class for dual simplex errors."""
    ERROR_MESSAGES = {
        0: "Success.",
        1: "Primal Infeasible (Dual Unbounded).",
        2: "Primal Unbounded or Infeasible (Dual Infeasible).",
        10: "Illegal dimensions.",
        32: "Singular Basis encountered (initial).",
        33: "Loss of Dual Feasibility.",
        40: "Iteration limit exceeded.",
        41: "Singular Basis encountered (during pivot).",
        51: "LAPACK error.",
    }
    
    def __init__(self, code: int):
        self.code = code
        self.message = self.ERROR_MESSAGES.get(code, f"Error code: {code}")
        super().__init__(self.message)

def dualsimplex(
    n: int,
    m: int,
    AT: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    ibasis: np.ndarray,
    eps: Optional[float] = None,
    ibudget: int = 50000,
    return_basis: bool = False
) -> Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """
    Solve: minimize B^T Y  s.t. A^T Y = C, Y >= 0
    (Equivalent to Max C^T X s.t. A X <= B)
    """
    if eps is None:
        eps = 1e-9

    # Ensure float64 for precision
    AT = AT.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)
    basis_idx = ibasis.copy().astype(int)

    # Validation
    if AT.shape != (n, m): raise DualSimplexError(12)
    
    for iteration in range(ibudget):
        # 1. Form Basis Matrix
        # basis_idx points to columns of AT (rows of A)
        B_mat = AT[:, basis_idx]
        
        # 2. Factorize Basis
        try:
            lu_piv = linalg.lu_factor(B_mat)
        except linalg.LinAlgError:
            raise DualSimplexError(32 if iteration == 0 else 41)

        # 3. Compute Dual Solution (Y_basis)
        # B_mat * Y_basis = C
        try:
            Y_basis = linalg.lu_solve(lu_piv, C)
        except linalg.LinAlgError:
            raise DualSimplexError(51)

        # Check Dual Feasibility (Y >= 0)
        if np.any(Y_basis < -eps):
             raise DualSimplexError(33)

        # 4. Compute Primal Solution (X)
        # B_mat^T * X = B_subset
        # This recovers the Primal variables X (Lagrange multipliers)
        b_basis = B[basis_idx]
        try:
            X = linalg.lu_solve(lu_piv, b_basis, trans=1)
        except linalg.LinAlgError:
            raise DualSimplexError(51)

        # 5. Compute Slacks (Reduced Costs for Dual)
        # S = B - A^T * X
        # Primal Feasibility check: Ax <= B  =>  B - Ax >= 0
        S = B - AT.T @ X
        
        # Enforce exact zero for basic variables to prevent drift
        S[basis_idx] = 0.0

        # 6. Check Optimality
        min_slack = np.min(S)
        if min_slack >= -eps:
            # Optimal
            Y_full = np.zeros(m)
            Y_full[basis_idx] = Y_basis
            return X, Y_full, 0, (basis_idx if return_basis else None)

        # 7. Select Entering Constraint (Violated Primal Constraint)
        # This constraint will ENTER the basis (become active)
        entering_idx_global = np.argmin(S)

        # 8. Select Leaving Constraint (Relaxed Constraint)
        # We must drop a column from B_mat.
        # Direction d = B^-1 * A_new
        A_col = AT[:, entering_idx_global]
        try:
            d = linalg.lu_solve(lu_piv, A_col)
        except linalg.LinAlgError:
            raise DualSimplexError(51)

        # Ratio Test: min(Y_i / d_i) for d_i > 0
        # We need to find which basic variable drops to 0 first.
        candidates = np.where(d > eps)[0]
        
        if len(candidates) == 0:
            raise DualSimplexError(1) # Dual Unbounded = Primal Infeasible
            
        ratios = Y_basis[candidates] / d[candidates]
        best_candidate_idx = np.argmin(ratios)
        leaving_idx_in_basis = candidates[best_candidate_idx]

        # 9. Update Basis
        basis_idx[leaving_idx_in_basis] = entering_idx_global
        
    raise DualSimplexError(40)


def feasible_basis(
    n: int,
    m: int,
    AT: np.ndarray,
    C: np.ndarray,
    eps: Optional[float] = None,
    ibudget: int = 50000
) -> Tuple[np.ndarray, int]:
    """
    Find a dual feasible basis using the Auxiliary Problem Method.
    """
    if eps is None: eps = 1e-9
    
    # 1. Construct Auxiliary Problem
    # We add N artificial variables with cost 1.0 (in the Dual sense).
    # This bounds the Dual problem to ensure a solution exists.
    A_aux = np.zeros((n, n + m))
    A_aux[:, n:n+m] = AT # Original columns shifted right
    
    # Artificial columns: Diagonal matrix * sign(C)
    # This ensures we can form an initial identity basis that matches signs of C
    for i in range(n):
        sign_c = np.sign(C[i]) if C[i] != 0 else 1.0
        A_aux[i, i] = sign_c

    # B_aux acts as the "Cost" for the Dual Simplex
    # We penalize artificials (cost 1) and prefer original vars (cost 0)
    B_aux = np.zeros(n + m)
    B_aux[:n] = 1.0 
    
    # Initial basis: The artificial variables (0..N-1)
    ibasis = np.arange(n)
    
    try:
        # Solve Auxiliary
        _, Y_aux, ierr, basis = dualsimplex(
            n, n + m, A_aux, B_aux, C, ibasis,
            eps=eps, ibudget=ibudget, return_basis=True
        )
    except DualSimplexError as e:
        return None, e.code

    if ierr != 0: return None, ierr
    
    # Check if we found a valid basis for the original problem
    # Logic: Artificials have cost 1. If Dual Objective > 0, we still use artificials?
    # Actually, we just need to ensure the final basis consists only of original columns.
    
    final_basis = basis.copy()
    
    # Map back to original indices (subtract N)
    # If index < N, it is an artificial variable.
    
    # 2. Clean up Artificials (Degeneracy handling)
    # If artificial variables are still in the basis (with value 0),
    # we must swap them for original columns to get a valid basis for AT.
    
    artificial_indices = np.where(final_basis < n)[0]
    
    if len(artificial_indices) > 0:
        # Get list of potential replacements (original cols not in basis)
        # available_cols = [j for j in range(n, n+m) if j not in final_basis]
        
        # This is the tricky part: We need to swap artificials out 
        # such that the matrix remains non-singular.
        # Simple Heuristic: Iterate through available columns and check rank.
        # (For high performance, use QR updates, but this is setup phase).
        
        # Create pool of available original indices (0..m-1) relative to AT
        current_original_indices = final_basis[final_basis >= n] - n
        available_original = list(set(range(m)) - set(current_original_indices))
        
        for i in artificial_indices:
            replaced = False
            for cand in available_original:
                # Try swapping artificial 'final_basis[i]' with 'cand'
                # Note: 'cand' is 0..m-1. In A_aux, it is cand + n.
                
                # Construct temporary basis to check singularity
                temp_basis = final_basis.copy()
                temp_basis[i] = cand + n
                
                # Extract matrix of this proposed basis
                # We need columns from A_aux
                B_try = A_aux[:, temp_basis]
                
                # Check condition number or rank
                if np.linalg.matrix_rank(B_try) == n:
                    final_basis[i] = cand + n
                    available_original.remove(cand)
                    replaced = True
                    break
            
            if not replaced:
                # Could not remove artificial variable - Primal Constraints likely dependent
                return None, 10 # Error

    # 3. Final mapping
    final_basis = final_basis - n
    
    return final_basis, 0


def solve_lp(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    eps: Optional[float] = None,
    ibudget: int = 50000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Main entry point.
    """
    if eps is None: eps = 1e-9
    
    m, n = A.shape
    # A is (Constraints x Vars). Dual Simplex expects AT (Vars x Constraints).
    AT = A.T
    
    try:
        # Phase 1: Get Feasible Basis
        basis, ierr = feasible_basis(n, m, AT, C, eps=eps, ibudget=ibudget)
        if ierr != 0: return None, None, None, ierr
        
        # Phase 2: Solve
        X, Y, ierr, obasis = dualsimplex(
            n, m, AT, B, C, basis,
            eps=eps, ibudget=ibudget, return_basis=True
        )
        return X, Y, obasis, ierr
        
    except DualSimplexError as e:
        # print(f"Solver Error: {e.message}")
        raise e