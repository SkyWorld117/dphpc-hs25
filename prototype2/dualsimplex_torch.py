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
from typing import Tuple, Optional

# Device configuration - will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Use double precision for numerical stability


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


def _lu_factor(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LU factorization with partial pivoting."""
    LU, pivots = torch.linalg.lu_factor(A)
    return LU, pivots


def _lu_solve(LU: torch.Tensor, pivots: torch.Tensor, b: torch.Tensor, trans: bool = False) -> torch.Tensor:
    """Solve a linear system using LU factorization."""
    if trans:
        # For transposed solve, we need to solve A^T x = b
        # PyTorch's lu_solve doesn't have a transpose option, so we handle it differently
        # A^T x = b => x = (A^-1)^T b = (A^T)^-1 b
        # We can use: x = torch.linalg.lu_solve(LU, pivots, b, adjoint=True)
        return torch.linalg.lu_solve(LU, pivots, b.unsqueeze(-1), adjoint=True).squeeze(-1)
    else:
        return torch.linalg.lu_solve(LU, pivots, b.unsqueeze(-1)).squeeze(-1)


def dualsimplex(
    n: int,
    m: int,
    AT: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    ibasis: torch.Tensor,
    eps: Optional[float] = None,
    ibudget: int = 50000,
    return_basis: bool = False
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
    
    # Pivot to match initial basis
    for i in range(n):
        j = (jpiv == ibasis[i]).nonzero(as_tuple=True)[0][0].item()
        # Swap columns in APIV
        apiv[:, [i, j]] = apiv[:, [j, i]]
        # Swap elements in BPIV
        bpiv[i], bpiv[j] = bpiv[j].clone(), bpiv[i].clone()
        # Track changes
        jpiv[i], jpiv[j] = jpiv[j].clone(), jpiv[i].clone()
    
    # Initialize solution arrays
    X = torch.zeros(n, device=DEVICE, dtype=DTYPE)
    Y = torch.zeros(m, device=DEVICE, dtype=DTYPE)
    
    # Get solution using LU factorization
    try:
        lu, piv = _lu_factor(apiv[:, :n])
    except RuntimeError:
        raise DualSimplexError(32)
    
    # Solve for first N elements of dual solution
    try:
        Y[:n] = _lu_solve(lu, piv, C)
    except RuntimeError:
        raise DualSimplexError(51)
    
    if torch.any(Y[:n] < -eps):
        raise DualSimplexError(33)
    
    Y[n:] = 0.0
    
    # Get primal solution
    try:
        X = _lu_solve(lu, piv, bpiv[:n], trans=True)
    except RuntimeError:
        raise DualSimplexError(51)
    
    # Compute slack variables
    S = bpiv[n:] - apiv[:, n:].T @ X
    
    # Check KKT conditions
    if torch.all(S >= -eps):
        # Undo pivots in Y
        Y_out = torch.zeros(m, device=DEVICE, dtype=DTYPE)
        for i in range(m):
            Y_out[jpiv[i]] = Y[i]
        obasis = jpiv[:n].clone() if return_basis else None
        return X, Y_out, 0, obasis
    
    # Begin iteration
    newsol = torch.dot(bpiv[:n], Y[:n])
    oldsol = newsol + 1.0
    
    for iteration in range(ibudget):
        # Choose pivot rule based on improvement
        if oldsol - newsol > eps:
            # Use Dantzig's rule
            ienter, iexit = _pivot_dantzig(n, m, apiv, Y, S, lu, piv, eps)
        else:
            # Use Bland's rule
            ienter, iexit = _pivot_bland(n, m, apiv, Y, S, lu, piv, eps)
        
        if iexit is None:
            # Dual unbounded
            raise DualSimplexError(1)
        
        # Perform pivot
        apiv[:, [iexit, ienter]] = apiv[:, [ienter, iexit]]
        bpiv[iexit], bpiv[ienter] = bpiv[ienter].clone(), bpiv[iexit].clone()
        jpiv[iexit], jpiv[ienter] = jpiv[ienter].clone(), jpiv[iexit].clone()
        
        # Update LU factorization
        try:
            lu, piv = _lu_factor(apiv[:, :n])
        except RuntimeError:
            raise DualSimplexError(41)
        
        # Update dual solution
        try:
            Y[:n] = _lu_solve(lu, piv, C)
        except RuntimeError:
            raise DualSimplexError(51)
        
        # Update primal solution
        try:
            X = _lu_solve(lu, piv, bpiv[:n], trans=True)
        except RuntimeError:
            raise DualSimplexError(51)
        
        # Update slack variables
        S = bpiv[n:] - apiv[:, n:].T @ X
        
        # Check KKT conditions
        if torch.all(S >= -eps):
            # Undo pivots in Y
            Y_out = torch.zeros(m, device=DEVICE, dtype=DTYPE)
            for i in range(m):
                Y_out[jpiv[i]] = Y[i]
            obasis = jpiv[:n].clone() if return_basis else None
            return X, Y_out, 0, obasis
        
        # Update solutions
        oldsol = newsol
        newsol = torch.dot(bpiv[:n], Y[:n])
    
    # Budget exceeded
    raise DualSimplexError(40)


def _pivot_dantzig(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor,
    lu: torch.Tensor, piv: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Dantzig's minimum ratio method for fast convergence.
    """
    # Entering index: most negative slack
    ienter = torch.argmin(S).item() + n
    
    # Build weight vector
    W = _lu_solve(lu, piv, apiv[:, ienter])
    
    # Compute ratios and choose exiting index
    currmin = float('inf')
    iexit = None
    
    for j in range(n):
        if W[j].item() < eps:
            continue
        ratio = Y[j].item() / W[j].item()
        if ratio < currmin:
            currmin = ratio
            iexit = j
    
    return ienter, iexit


def _pivot_bland(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor,
    lu: torch.Tensor, piv: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Bland's anticycling rule for guaranteed convergence.
    """
    # Entering index: first negative slack
    ienter = None
    for j in range(m - n):
        if S[j].item() < -eps:
            ienter = j + n
            break
    
    if ienter is None:
        return n, None
    
    # Build weight vector
    W = _lu_solve(lu, piv, apiv[:, ienter])
    
    # Compute ratios and choose exiting index
    currmin = float('inf')
    iexit = None
    
    for j in range(n):
        if W[j].item() < eps:
            continue
        ratio = Y[j].item() / W[j].item()
        if ratio - currmin < -eps:
            currmin = ratio
            iexit = j
    
    return ienter, iexit


def feasible_basis(
    n: int,
    m: int,
    AT: torch.Tensor,
    C: torch.Tensor,
    eps: Optional[float] = None,
    ibudget: int = 50000
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
    for i in range(n):
        A_aux[i, i] = torch.sign(C[i]).item() if C[i].item() != 0 else 1.0
    
    # Initial basis
    ibasis = torch.arange(n, device=DEVICE, dtype=torch.long)
    
    # Solve auxiliary problem
    X, Y_aux, ierr, basis = dualsimplex(
        n, n + m, A_aux, B_aux, C, ibasis,
        eps=eps, ibudget=ibudget, return_basis=True
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
    ibudget: int = 50000
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
    basis, ierr = feasible_basis(n, m, AT, C, eps=eps, ibudget=ibudget)
    if ierr != 0:
        return None, None, None, ierr
    
    # Solve the LP
    X, Y, ierr, obasis = dualsimplex(
        n, m, AT, B, C, basis,
        eps=eps, ibudget=ibudget, return_basis=True
    )
    
    return X, Y, obasis, ierr


# Utility functions for numpy interoperability
def to_numpy(tensor: torch.Tensor):
    """Convert a tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def from_numpy(arr, device=None):
    """Convert a numpy array to tensor."""
    return _to_tensor(arr, device)
