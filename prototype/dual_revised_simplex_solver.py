"""
Dual Revised Simplex Solver for Linear Programming using PyTorch

This module implements the dual revised simplex method for solving linear programming 
problems in standard form:
    minimize    c^T x
    subject to  Ax = b
                x >= 0

The dual revised simplex method maintains and updates the basis inverse incrementally,
making it more efficient than the standard dual simplex method. It is particularly 
efficient when starting from a dual feasible but primal infeasible solution.
"""

import torch
from typing import Tuple, Optional, List

# Disable PyTorch gradient calculations globally
torch.set_grad_enabled(False)

class DualRevisedSimplexSolver:
    """
    Dual Revised Simplex Method implementation using PyTorch.
    
    The solver handles LP problems in standard form:
        minimize    c^T x
        subject to  Ax = b
                    x >= 0

    This implementation maintains and updates the basis inverse B_inv incrementally
    using rank-1 updates (eta matrices), which is more efficient than recomputing
    the inverse in each iteration.
    """
   
    def __init__(self, c: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                 max_iter: int = 1000, tol: float = 1e-7, device: str = 'cpu',
                 refactor_frequency: int = 50):
        """
        Initialize the dual revised simplex solver.

        Args:
            c: Cost vector (n,)
            A: Constraint matrix (m, n)
            b: Right-hand side vector (m,)
            max_iter: Maximum number of iterations
            tol: Tolerance for numerical comparisons
            device: Device to run computations on ('cpu', 'cuda', 'cuda:0', etc.)
            refactor_frequency: How often to refactor B_inv from scratch (for numerical stability)
        """
        self.device = torch.device(device)
        self.c = c.clone().float().to(self.device)
        self.A = A.clone().float().to(self.device)
        self.b = b.clone().float().to(self.device)
        self.max_iter = max_iter
        self.tol = tol
        self.refactor_frequency = refactor_frequency
        
        self.m, self.n = A.shape
        
        # Solution variables
        self.x = None
        self.objective_value = None
        self.status = None
        self.iterations = 0
        
        # Revised simplex specific: maintain B_inv
        self.B_inv = None
        
    def _compute_basis_inverse(self) -> torch.Tensor:
        """Compute the basis inverse from scratch."""
        B = self.A[:, self.basis]
        try:
            return torch.linalg.inv(B)
        except:
            raise ValueError("Singular basis matrix")
    
    def _update_basis_inverse(self, leaving_idx: int, entering_var: int) -> None:
        """
        Update B_inv using the eta matrix (rank-1 update).
        
        This is the key efficiency improvement of the revised simplex method.
        Instead of recomputing B_inv from scratch, we update it incrementally.
        
        Args:
            leaving_idx: Index in basis of the leaving variable
            entering_var: Column index of the entering variable
        """
        # Compute the entering column in the basis
        a_entering = self.A[:, entering_var]
        d = self.B_inv @ a_entering  # Direction vector
        
        # Compute the eta matrix inverse
        # E^{-1} is an identity matrix except column leaving_idx
        pivot = d[leaving_idx]
        if abs(pivot) < self.tol:
            raise ValueError("Pivot too small, basis may be singular")
        
        # Update B_inv using the eta matrix
        # B_inv_new = E^{-1} @ B_inv_old
        eta_col = -d / pivot
        eta_col[leaving_idx] = 1.0 / pivot
        
        # Apply the eta transformation
        # This is equivalent to: B_inv_new = E_inv @ B_inv
        B_inv_new = self.B_inv.clone()
        for i in range(self.m):
            if i != leaving_idx:
                B_inv_new[i, :] = self.B_inv[i, :] - (d[i] / pivot) * self.B_inv[leaving_idx, :]
            else:
                B_inv_new[i, :] = self.B_inv[i, :] / pivot
        
        self.B_inv = B_inv_new
    
    def solve(self, basis: Optional[List[int]] = None) -> Tuple[torch.Tensor, float, str]:
        """
        Solve the LP problem using the dual revised simplex method.
        
        Args:
            basis: Initial basic variable indices. If None, uses the last m variables.
        
        Returns:
            x: Optimal solution vector
            obj_val: Optimal objective value
            status: Solution status ('optimal', 'infeasible', 'unbounded', 'max_iter')
        """
        # Initialize basis if not provided
        if basis is None:
            basis = list(range(self.n - self.m, self.n))

        self.basis = basis
        self.non_basis = [i for i in range(self.n) if i not in basis]

        # Initialize solution
        self.x = torch.zeros(self.n, device=self.device)

        # Compute initial basis inverse
        try:
            self.B_inv = self._compute_basis_inverse()
        except ValueError:
            self.status = 'singular_basis'
            return self.x, float('inf'), self.status

        # Main dual revised simplex loop
        for iteration in range(self.max_iter):
            self.iterations = iteration + 1

            # Periodically refactor B_inv for numerical stability
            if iteration > 0 and iteration % self.refactor_frequency == 0:
                try:
                    self.B_inv = self._compute_basis_inverse()
                except ValueError:
                    self.status = 'singular_basis'
                    return self.x, float('inf'), self.status

            # Compute basic solution: x_B = B_inv @ b
            x_B = self.B_inv @ self.b
            self.x = torch.zeros(self.n, device=self.device)
            for i, idx in enumerate(self.basis):
                self.x[idx] = x_B[i]

            # Check for primal feasibility (all basic variables >= 0)
            if torch.all(x_B >= -self.tol):
                # Primal feasible, check dual feasibility
                c_B = self.c[self.basis]
                y = c_B @ self.B_inv  # Dual variables (simplex multipliers)

                # Compute reduced costs for non-basic variables
                dual_feasible = True
                for j in self.non_basis:
                    reduced_cost = self.c[j] - y @ self.A[:, j]
                    if reduced_cost < -self.tol:
                        dual_feasible = False
                        break

                if dual_feasible:
                    self.objective_value = (self.c @ self.x).item()
                    self.status = 'optimal'
                    return self.x, self.objective_value, self.status

            # Find leaving variable (most negative basic variable)
            leaving_idx = int(torch.argmin(x_B).item())
            if x_B[leaving_idx] >= -self.tol:
                # Should not happen if we reach here, but safety check
                self.objective_value = (self.c @ self.x).item()
                self.status = 'optimal'
                return self.x, self.objective_value, self.status

            leaving_var = self.basis[leaving_idx]

            # Compute pivot row (row of B_inv corresponding to leaving variable)
            pivot_row = self.B_inv[leaving_idx, :]
            
            # Find entering variable using dual simplex ratio test
            c_B = self.c[self.basis]
            y = c_B @ self.B_inv
            
            entering_var = None
            min_ratio = float('inf')
            
            for j in self.non_basis:
                a_j = self.A[:, j]
                alpha_j = pivot_row @ a_j
                
                if alpha_j < -self.tol:  # Valid candidate for entering
                    reduced_cost = self.c[j] - y @ a_j
                    ratio = -reduced_cost / alpha_j
                    
                    if ratio < min_ratio:
                        min_ratio = ratio
                        entering_var = j
            
            if entering_var is None:
                self.status = 'infeasible'
                return self.x, float('inf'), self.status
            
            # Update basis inverse (REVISED SIMPLEX KEY STEP)
            try:
                self._update_basis_inverse(leaving_idx, entering_var)
            except ValueError:
                # If update fails, refactor from scratch
                self.B_inv = self._compute_basis_inverse()
            
            # Update basis
            self.basis[leaving_idx] = entering_var
            self.non_basis.remove(entering_var)
            self.non_basis.append(leaving_var)
            self.non_basis.sort()
        
        # Max iterations reached
        self.status = 'max_iter'
        self.objective_value = (self.c @ self.x).item()
        return self.x, self.objective_value, self.status
    
    def get_info(self) -> dict:
        """
        Get information about the solution.
        
        Returns:
            Dictionary containing solution information
        """
        return {
            'x': self.x,
            'objective_value': self.objective_value,
            'status': self.status,
            'iterations': self.iterations,
            'basis': self.basis if hasattr(self, 'basis') else None
        }


def add_slack_variables(c: torch.Tensor, A: torch.Tensor, b: torch.Tensor,
                       inequality_types: Optional[List[str]] = None,
                       device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert inequality constraints to equality by adding slack variables.
    
    Args:
        c: Cost vector
        A: Constraint matrix
        b: Right-hand side vector
        inequality_types: List of inequality types ('<=', '>=', '=')
                         If None, assumes all are '='
        device: Device to place tensors on (if None, uses c's device)
    
    Returns:
        c_aug: Augmented cost vector
        A_aug: Augmented constraint matrix
        b_aug: Right-hand side vector (unchanged)
    """
    m, n = A.shape
    
    if device is None:
        device = c.device
    
    if inequality_types is None:
        return c, A, b
    
    # Count slack variables needed
    n_slack = sum(1 for ineq in inequality_types if ineq in ['<=', '>='])
    
    # Augment cost vector (slack variables have zero cost)
    c_aug = torch.cat([c, torch.zeros(n_slack, device=device)])
    
    # Augment constraint matrix
    slack_cols = []
    slack_idx = 0
    for i, ineq in enumerate(inequality_types):
        if ineq == '<=':
            col = torch.zeros(m, device=device)
            col[i] = 1.0
            slack_cols.append(col)
            slack_idx += 1
        elif ineq == '>=':
            col = torch.zeros(m, device=device)
            col[i] = -1.0
            slack_cols.append(col)
            slack_idx += 1
    
    if slack_cols:
        slack_matrix = torch.stack(slack_cols, dim=1)
        A_aug = torch.cat([A, slack_matrix], dim=1)
    else:
        A_aug = A
    
    return c_aug, A_aug, b
