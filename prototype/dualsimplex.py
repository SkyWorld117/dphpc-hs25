"""
Dual Simplex Module - PyTorch Version

PyTorch port of dualsimplex.py for GPU acceleration.
Solves the asymmetric dual of a problem:

    max C^T X
    s.t. A X <= B

where A is dense and the dual solution is unique.

Two functions are provided:
- dualsimplex: for solving an LP when the initial basis is known (Phase II)
- feasible_basis: for finding an initial dual feasible basis (Phase I)

Author: Tyler Chang (original Fortran)
PyTorch port: 2024
"""

import torch
from torch.profiler import record_function
from typing import Tuple, Optional

# Device configuration - will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Use double precision for numerical stability

torch.set_grad_enabled(False)  # Disable autograd for optimization routines

def set_device(device: str):
    """Set the computation device ('cuda', 'cpu', or 'mps')."""
    global DEVICE
    DEVICE = torch.device(device)


def get_device() -> torch.device:
    """Get the current computation device."""
    return DEVICE


class DualSimplexError(Exception):
    """Exception class for dual simplex errors."""
    
    ERROR_MESSAGES = {
        0: "Success: C^T X has been successfully maximized.",
        1: "The dual problem is unbounded, therefore the primal must be infeasible.",
        2: "The dual problem is infeasible, the primal may be unbounded or infeasible.",
        10: "Illegal problem dimensions: N < 1.",
        11: "Illegal problem dimensions: M < N.",
        12: "N does not match the first dimension of the constraint matrix AT.",
        13: "M does not match the second dimension of the constraint matrix AT.",
        14: "M does not match the length of the upper bounds B.",
        15: "N does not match the length of the cost vector C.",
        16: "N does not match the length of the initial basis IBASIS.",
        17: "N does not match the length of the primal solution vector X.",
        18: "M does not match the length of the dual solution vector Y.",
        20: "The optional argument EPS must be strictly positive.",
        21: "The optional argument IBUDGET must be nonnegative.",
        22: "The optional argument OBASIS must be length N.",
        30: "The provided initial basis IBASIS contains out of bounds indices.",
        31: "The provided initial basis IBASIS contains duplicate indices.",
        32: "The provided initial basis IBASIS produced a singularity.",
        33: "The provided initial basis IBASIS is not feasible.",
        40: "The pivot budget was exceeded before a solution could be found.",
        41: "A pivot has produced a singular basis.",
        50: "LAPACK error in LU factorization.",
        51: "LAPACK error in triangular solve.",
    }
    
    def __init__(self, code: int):
        self.code = code
        self.message = self.ERROR_MESSAGES.get(code, f"Unknown error code: {code}")
        super().__init__(self.message)


def _to_tensor(arr, device=None) -> torch.Tensor:
    """Convert input to tensor on the specified device."""
    if device is None:
        device = DEVICE
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=DTYPE)
    return torch.tensor(arr, device=device, dtype=DTYPE)


def _compute_basis_inverse(A: torch.Tensor) -> torch.Tensor:
    """Compute pseudo-inverse of basis matrix."""
    return torch.linalg.pinv(A)


def _apply_eta_update(B_inv: torch.Tensor, eta_col: torch.Tensor, pivot_row: int, 
                      pivot_row_buf: torch.Tensor = None) -> None:
    """
    Apply eta transformation to update the basis inverse (in-place).
    
    When we pivot, the new basis matrix B_new = B * E where E is an eta matrix.
    E is the identity matrix with column `pivot_row` replaced by `eta_col`.
    
    B_new^{-1} = E^{-1} * B^{-1}
    
    E^{-1} is also an eta matrix where:
    - E^{-1}[pivot_row, pivot_row] = 1 / eta_col[pivot_row]
    - E^{-1}[j, pivot_row] = -eta_col[j] / eta_col[pivot_row] for j != pivot_row
    
    Parameters
    ----------
    pivot_row_buf : torch.Tensor, optional
        Preallocated buffer for pivot row to avoid allocation
    """
    pivot_val = eta_col[pivot_row]
    inv_pivot = 1.0 / pivot_val
    
    # Compute eta inverse in-place
    eta_col.mul_(-inv_pivot)
    eta_col[pivot_row] = inv_pivot
    
    # Copy pivot row to buffer (reuse provided buffer to avoid allocation)
    if pivot_row_buf is None:
        pivot_row_buf = B_inv[pivot_row, :].clone()
    else:
        pivot_row_buf.copy_(B_inv[pivot_row, :])
    
    # Apply E^{-1} * B_inv using addmm-style update
    # B_inv += eta_col.unsqueeze(1) * pivot_row_buf.unsqueeze(0)
    torch.addr(B_inv, eta_col, pivot_row_buf, out=B_inv)
    # Correct the pivot row (it was added twice)
    B_inv[pivot_row, :].copy_(pivot_row_buf).mul_(inv_pivot)


def _solve_with_inverse(B_inv: torch.Tensor, b: torch.Tensor, trans: bool = False, 
                        out: torch.Tensor = None) -> torch.Tensor:
    """Solve a linear system using the basis inverse.
    
    Parameters
    ----------
    out : torch.Tensor, optional
        Preallocated output tensor to avoid allocation
    """
    if trans:
        # Solve A^T x = b => x = (A^{-1})^T b = B_inv^T @ b
        if out is not None:
            return torch.mv(B_inv.T, b, out=out)
        return B_inv.T @ b
    else:
        # Solve A x = b => x = A^{-1} b = B_inv @ b
        if out is not None:
            return torch.mv(B_inv, b, out=out)
        return B_inv @ b


def dualsimplex(
    n: int,
    m: int,
    AT: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ibasis: torch.Tensor,
    eps: Optional[float] = None,
    ibudget: int = 50000,
    return_basis: bool = False,
    refactor_interval: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
    """
    Solve a primal problem of the form:
    
        maximize     C^T X
        such that    A X <= B
    
    where A in R^{M x N}, C,X in R^N, and B in R^M.
    
    This is done by applying the revised simplex method on the asymmetric dual:
    
        minimize     B^T Y
        such that    A^T Y = C
        and          Y >= 0
    
    Parameters
    ----------
    n : int
        Number of variables in the primal problem
    m : int
        Number of constraints in the primal problem
    AT : torch.Tensor
        Transpose of the constraint matrix A, shape (n, m)
    B : torch.Tensor
        Upper bounds vector, shape (m,)
    C : torch.Tensor
        Cost vector, shape (n,)
    ibasis : torch.Tensor
        Initial dual feasible basis indices, shape (n,)
    eps : float, optional
        Working precision (default: sqrt of machine epsilon)
    ibudget : int, optional
        Maximum number of pivots allowed (default: 50000)
    return_basis : bool, optional
        Whether to return the final basis (default: False)
    refactor_interval : int, optional
        Number of iterations between full pseudo-inverse recomputation (default: 100).
        Lower values improve numerical stability but reduce performance.
    
    Returns
    -------
    X : torch.Tensor
        Primal solution, shape (n,)
    Y : torch.Tensor
        Dual solution, shape (m,)
    ierr : int
        Error code (0 = success)
    obasis : torch.Tensor or None
        Final basis indices if return_basis=True
    """
    # Set default eps
    if eps is None:
        eps = torch.finfo(DTYPE).eps
    
    # Convert inputs to tensors
    AT = _to_tensor(AT)
    B = _to_tensor(B)
    C = _to_tensor(C)
    
    # Input validation
    if n < 1:
        raise DualSimplexError(10)
    if m < n:
        raise DualSimplexError(11)
    if AT.shape[0] != n:
        raise DualSimplexError(12)
    if AT.shape[1] != m:
        raise DualSimplexError(13)
    if len(B) != m:
        raise DualSimplexError(14)
    if len(C) != n:
        raise DualSimplexError(15)
    if len(ibasis) != n:
        raise DualSimplexError(16)
    if eps <= 0:
        raise DualSimplexError(20)
    if ibudget < 0:
        raise DualSimplexError(21)
    
    # Convert ibasis to tensor if needed
    if isinstance(ibasis, torch.Tensor):
        ibasis = ibasis.clone().to(device=DEVICE, dtype=torch.long)
    else:
        ibasis = torch.tensor(ibasis, device=DEVICE, dtype=torch.long)
    
    # Check initial basis validity
    if torch.any(ibasis < 0) or torch.any(ibasis >= m):
        raise DualSimplexError(30)
    if len(torch.unique(ibasis)) != n:
        raise DualSimplexError(31)
    
    # Initialize pivot tracking
    jpiv = torch.arange(m, device=DEVICE, dtype=torch.long)
    
    # Initialize APIV and BPIV
    apiv = AT.clone()
    bpiv = B.clone()
    
    # Preallocate swap buffers to avoid repeated allocations
    _col_buf = torch.empty(n, device=DEVICE, dtype=DTYPE)
    
    # Pivot to match initial basis using index swaps (avoids fancy indexing overhead)
    # for i in range(n):
    #     j = (jpiv == ibasis[i]).nonzero(as_tuple=True)[0][0].item()
    #     if i != j:
    #         # Swap columns in APIV using buffer
    #         _col_buf.copy_(apiv[:, i])
    #         apiv[:, i].copy_(apiv[:, j])
    #         apiv[:, j].copy_(_col_buf)
    #         # Swap elements in BPIV (scalars)
    #         tmp_b = bpiv[i].item()
    #         bpiv[i] = bpiv[j]
    #         bpiv[j] = tmp_b
    #         # Swap tracking indices
    #         tmp_j = jpiv[i].item()
    #         jpiv[i] = jpiv[j]
    #         jpiv[j] = tmp_j

    # Vectorized permutation: place ibasis entries in the first n positions in one shot
    mask = torch.ones(m, dtype=torch.bool, device=DEVICE)
    mask[ibasis] = False
    remaining = torch.nonzero(mask, as_tuple=True)[0]
    order = torch.cat([ibasis, remaining])
    apiv = apiv[:, order]
    bpiv = bpiv[order]
    jpiv = jpiv[order]

    # Initialize solution arrays
    X = torch.zeros(n, device=DEVICE, dtype=DTYPE)
    Y = torch.zeros(m, device=DEVICE, dtype=DTYPE)
    
    # Get solution using pseudo-inverse
    try:
        B_inv = _compute_basis_inverse(apiv[:, :n])
    except RuntimeError:
        raise DualSimplexError(32)
    
    # Solve for first N elements of dual solution
    try:
        Y[:n] = _solve_with_inverse(B_inv, C)
    except RuntimeError:
        raise DualSimplexError(51)
    
    if torch.any(Y[:n] < -eps):
        raise DualSimplexError(33)
    
    Y[n:] = 0.0
    
    # Get primal solution
    try:
        X = _solve_with_inverse(B_inv, bpiv[:n], trans=True)
    except RuntimeError:
        raise DualSimplexError(51)
    
    # Track iterations since last refactorization
    iters_since_refactor = 0
    
    # Preallocate working buffers to avoid allocations in the main loop
    S = torch.empty(m - n, device=DEVICE, dtype=DTYPE)  # Slack variables
    eta_col = torch.empty(n, device=DEVICE, dtype=DTYPE)  # Eta column buffer
    pivot_row_buf = torch.empty(n, device=DEVICE, dtype=DTYPE)  # For eta update
    Y_n_buf = torch.empty(n, device=DEVICE, dtype=DTYPE)  # Buffer for Y[:n] solve
    
    # Compute slack variables: S = bpiv[n:] - apiv[:, n:].T @ X
    torch.mv(apiv[:, n:].T, X, out=S)
    S.neg_().add_(bpiv[n:])
    
    # Check KKT conditions
    if torch.all(S >= -eps):
        # Undo pivots in Y using scatter
        Y_out = torch.zeros(m, device=DEVICE, dtype=DTYPE)
        Y_out.scatter_(0, jpiv, Y)
        obasis = jpiv[:n].clone() if return_basis else None
        return X, Y_out, 0, obasis
    
    # Begin iteration
    newsol = torch.dot(bpiv[:n], Y[:n])
    oldsol = newsol + 1.0
    
    for iteration in range(ibudget):
        # Choose pivot rule based on improvement
        if oldsol - newsol > eps:
            # Use Dantzig's rule
            ienter, iexit = _pivot_dantzig(n, m, apiv, Y, S, B_inv, eps)
        else:
            # Use Bland's rule
            ienter, iexit = _pivot_bland(n, m, apiv, Y, S, B_inv, eps)
        # TODO consider using dual steepest edge
        
        if iexit is None:
            # Dual unbounded
            raise DualSimplexError(1)
        
        # Compute eta column before swapping (reuse buffer)
        _solve_with_inverse(B_inv, apiv[:, ienter], out=eta_col)
        
        # Perform pivot using explicit swaps (avoids fancy indexing temporaries)
        _col_buf.copy_(apiv[:, iexit])
        apiv[:, iexit].copy_(apiv[:, ienter])
        apiv[:, ienter].copy_(_col_buf)
        tmp_b = bpiv[iexit].item()
        bpiv[iexit] = bpiv[ienter]
        bpiv[ienter] = tmp_b
        tmp_j = jpiv[iexit].item()
        jpiv[iexit] = jpiv[ienter]
        jpiv[ienter] = tmp_j
        
        # Update basis inverse using eta transformation or recompute
        iters_since_refactor += 1
        if iters_since_refactor >= refactor_interval:
            # Full refactorization
            try:
                B_inv = _compute_basis_inverse(apiv[:, :n])
            except RuntimeError:
                raise DualSimplexError(41)
            iters_since_refactor = 0
        else:
            # Incremental update using eta transformation
            try:
                _apply_eta_update(B_inv, eta_col, iexit, pivot_row_buf)
            except RuntimeError:
                raise DualSimplexError(41)
        
        # Update dual solution (reuse buffer then copy)
        try:
            _solve_with_inverse(B_inv, C, out=Y_n_buf)
            Y[:n].copy_(Y_n_buf)
        except RuntimeError:
            raise DualSimplexError(51)
        
        # Update primal solution in-place
        try:
            _solve_with_inverse(B_inv, bpiv[:n], trans=True, out=X)
        except RuntimeError:
            raise DualSimplexError(51)
        
        # Update slack variables in-place: S = bpiv[n:] - apiv[:, n:].T @ X
        torch.mv(apiv[:, n:].T, X, out=S)
        S.neg_().add_(bpiv[n:])
        
        # Check KKT conditions
        if torch.all(S >= -eps):
            # Undo pivots in Y using scatter (vectorized)
            Y_out = torch.zeros(m, device=DEVICE, dtype=DTYPE)
            Y_out.scatter_(0, jpiv, Y)
            obasis = jpiv[:n].clone() if return_basis else None
            return X, Y_out, 0, obasis
        
        # Update solutions
        oldsol = newsol
        newsol = torch.dot(bpiv[:n], Y[:n])
    
    # Budget exceeded
    raise DualSimplexError(40)


# Preallocated buffers for pivot functions (module-level to avoid repeated allocation)
_pivot_W: torch.Tensor = None
_pivot_ratio: torch.Tensor = None


def _ensure_pivot_buffers(n: int, device: torch.device):
    """Ensure pivot buffers are allocated with correct size."""
    global _pivot_W, _pivot_ratio
    if _pivot_W is None or _pivot_W.shape[0] != n or _pivot_W.device != device:
        _pivot_W = torch.empty(n, device=device, dtype=DTYPE)
        _pivot_ratio = torch.empty(n, device=device, dtype=DTYPE)


def _pivot_dantzig(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor,
    B_inv: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Dantzig's minimum ratio method for fast convergence.
    """
    _ensure_pivot_buffers(n, apiv.device)
    
    # Entering index: most negative slack
    ienter = torch.argmin(S).item() + n
    
    # Build weight vector using basis inverse (reuse buffer)
    _solve_with_inverse(B_inv, apiv[:, ienter], out=_pivot_W)
    
    # Compute ratios and choose exiting index
    if not torch.any(_pivot_W > eps):
        return ienter, None

    # Compute ratio in-place
    torch.div(Y[:n], _pivot_W, out=_pivot_ratio)
    iexit = torch.argmin(torch.where(_pivot_W > eps, _pivot_ratio, torch.inf)).item()
    
    return ienter, iexit


def _pivot_bland(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor,
    B_inv: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Bland's anticycling rule for guaranteed convergence.
    """
    _ensure_pivot_buffers(n, apiv.device)
    
    # Entering index: first negative slack
    neg_mask = S < -eps
    if not torch.any(neg_mask):
        return n, None
    
    ienter = neg_mask.nonzero(as_tuple=True)[0][0].item() + n
    
    # Build weight vector using basis inverse (reuse buffer)
    _solve_with_inverse(B_inv, apiv[:, ienter], out=_pivot_W)
    
    # Compute ratios and choose exiting index
    if not torch.any(_pivot_W > eps):
        return ienter, None

    # Compute ratio in-place
    torch.div(Y[:n], _pivot_W, out=_pivot_ratio)
    iexit = torch.argmin(torch.where(_pivot_W > eps, _pivot_ratio, torch.inf)).item()
    
    return ienter, iexit


def feasible_basis(
    n: int,
    m: int,
    AT: torch.Tensor,
    C: torch.Tensor,
    eps: Optional[float] = None,
    ibudget: int = 50000,
    refactor_interval: int = 100
) -> Tuple[torch.Tensor, int]:
    """
    Find a dual feasible basis for a primal problem using the auxiliary method.
    
    Parameters
    ----------
    n : int
        Number of variables in the primal problem
    m : int
        Number of constraints in the primal problem
    AT : torch.Tensor
        Transpose of the constraint matrix A, shape (n, m)
    C : torch.Tensor
        Cost vector, shape (n,)
    eps : float, optional
        Working precision (default: sqrt of machine epsilon)
    ibudget : int, optional
        Maximum number of pivots (default: 50000)
    refactor_interval : int, optional
        Number of iterations between full pseudo-inverse recomputation (default: 100)
    
    Returns
    -------
    basis : torch.Tensor
        Indices of a dual feasible basis for AT
    ierr : int
        Error code (0 = success)
    """
    # Set default eps
    if eps is None:
        eps = torch.finfo(DTYPE).eps
    
    # Convert inputs to tensors
    AT = _to_tensor(AT)
    C = _to_tensor(C)
    
    # Input validation
    if n < 1:
        raise DualSimplexError(10)
    if m < n:
        raise DualSimplexError(11)
    if AT.shape[0] != n:
        raise DualSimplexError(12)
    if AT.shape[1] != m:
        raise DualSimplexError(13)
    if len(C) != n:
        raise DualSimplexError(15)
    if eps <= 0:
        raise DualSimplexError(20)
    if ibudget < 0:
        raise DualSimplexError(21)
    
    # Build auxiliary problem
    A_aux = torch.zeros((n, n + m), device=DEVICE, dtype=DTYPE)
    A_aux[:, n:n+m] = AT
    
    B_aux = torch.zeros(n + m, device=DEVICE, dtype=DTYPE)
    B_aux[:n] = 1.0
    
    # Create artificial variables
    # TODO: This gets slower huh?
    # for i in range(n):
    #     A_aux[i, i] = torch.sign(C[i]).item() if C[i].item() != 0 else 1.0
    A_aux[torch.arange(n), torch.arange(n)] = torch.where(C != 0, torch.sign(C), torch.tensor(1.0, device=DEVICE, dtype=DTYPE))
    
    # Initial basis
    ibasis = torch.arange(n, device=DEVICE, dtype=torch.long)
    
    # Solve auxiliary problem
    X, Y_aux, ierr, basis = dualsimplex(
        n, n + m, A_aux, B_aux, C, ibasis,
        eps=eps, ibudget=ibudget, return_basis=True, refactor_interval=refactor_interval
    )
    
    if ierr != 0:
        raise DualSimplexError(ierr)
    
    # Check for infeasible dual solution
    if torch.dot(Y_aux, B_aux).item() > eps:
        raise DualSimplexError(2)
    
    # Adjust basis indices
    basis = basis - n
    
    # Handle degeneracies - ensure all basis elements are legal
    for i in range(n):
        if basis[i].item() < 0:
            idx = 0
            while idx in basis.tolist():
                idx += 1
            basis[i] = idx
    
    return basis, 0


def solve_lp(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    eps: Optional[float] = None,
    ibudget: int = 50000,
    refactor_interval: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Convenience function to solve an LP from scratch.
    
    Solves:
        maximize     C^T X
        such that    A X <= B
    
    Parameters
    ----------
    A : torch.Tensor or np.ndarray
        Constraint matrix, shape (m, n)
    B : torch.Tensor or np.ndarray
        Upper bounds, shape (m,)
    C : torch.Tensor or np.ndarray
        Cost vector, shape (n,)
    eps : float, optional
        Working precision
    ibudget : int, optional
        Maximum pivots
    refactor_interval : int, optional
        Number of iterations between full pseudo-inverse recomputation (default: 100)
    
    Returns
    -------
    X : torch.Tensor
        Primal solution
    Y : torch.Tensor
        Dual solution
    basis : torch.Tensor
        Final basis indices
    ierr : int
        Error code
    """
    if eps is None:
        eps = float(torch.finfo(DTYPE).eps ** 0.5)
    
    # Convert to tensors
    A = _to_tensor(A)
    B = _to_tensor(B)
    C = _to_tensor(C)
    
    m, n = A.shape
    AT = A.T  # Transpose for our implementation
    
    # Find initial feasible basis
    basis, ierr = feasible_basis(n, m, AT, C, eps=eps, ibudget=ibudget, refactor_interval=refactor_interval)
    if ierr != 0:
        return None, None, None, ierr

    # Solve the LP
    X, Y, ierr, obasis = dualsimplex(
        n, m, AT, B, C, basis,
        eps=eps, ibudget=ibudget, return_basis=True, refactor_interval=refactor_interval
    )

    return X, Y, obasis, ierr


# Utility functions for numpy interoperability
def to_numpy(tensor: torch.Tensor):
    """Convert a tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def from_numpy(arr, device=None):
    """Convert a numpy array to tensor."""
    return _to_tensor(arr, device)
