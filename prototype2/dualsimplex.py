"""
Dual Simplex Module

Python version of dualsimplex.f90 - Fortran 90 module for solving the asymmetric 
dual of a problem:

    max C^T X
    s.t. A X <= B

where A is dense and the dual solution is unique.

Two functions are provided:
- dualsimplex: for solving an LP when the initial basis is known (Phase II)
- feasible_basis: for finding an initial dual feasible basis (Phase I)

Author: Tyler Chang (original Fortran)
Last Update: July, 2019
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional


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
    AT : np.ndarray
        Transpose of the constraint matrix A, shape (n, m)
    B : np.ndarray
        Upper bounds vector, shape (m,)
    C : np.ndarray
        Cost vector, shape (n,)
    ibasis : np.ndarray
        Initial dual feasible basis indices, shape (n,)
    eps : float, optional
        Working precision (default: sqrt of machine epsilon)
    ibudget : int, optional
        Maximum number of pivots allowed (default: 50000)
    return_basis : bool, optional
        Whether to return the final basis (default: False)
    
    Returns
    -------
    X : np.ndarray
        Primal solution, shape (n,)
    Y : np.ndarray
        Dual solution, shape (m,)
    ierr : int
        Error code (0 = success)
    obasis : np.ndarray or None
        Final basis indices if return_basis=True
    """
    # Set default eps
    if eps is None:
        eps = np.finfo(float).eps
    
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
    
    # Check initial basis validity
    ibasis = ibasis.copy()
    if np.any(ibasis < 0) or np.any(ibasis >= m):
        raise DualSimplexError(30)
    if len(np.unique(ibasis)) != n:
        raise DualSimplexError(31)
    
    # Initialize pivot tracking
    jpiv = np.arange(m)
    
    # Initialize APIV and BPIV
    apiv = AT.copy()
    bpiv = B.copy()
    
    # Pivot to match initial basis
    for i in range(n):
        j = np.where(jpiv == ibasis[i])[0][0]
        # Swap columns in APIV
        apiv[:, [i, j]] = apiv[:, [j, i]]
        # Swap elements in BPIV
        bpiv[i], bpiv[j] = bpiv[j], bpiv[i]
        # Track changes
        jpiv[i], jpiv[j] = jpiv[j], jpiv[i]
    
    # Initialize solution arrays
    X = np.zeros(n)
    Y = np.zeros(m)
    
    # Get solution using LU factorization
    try:
        lu, piv = linalg.lu_factor(apiv[:, :n])
    except linalg.LinAlgError:
        raise DualSimplexError(32)
    
    # Solve for first N elements of dual solution
    try:
        Y[:n] = linalg.lu_solve((lu, piv), C)
    except linalg.LinAlgError:
        raise DualSimplexError(51)
    
    if np.any(Y[:n] < -eps):
        raise DualSimplexError(33)
    
    Y[n:] = 0.0
    
    # Get primal solution
    try:
        X = linalg.lu_solve((lu, piv), bpiv[:n], trans=1)
    except linalg.LinAlgError:
        raise DualSimplexError(51)
    
    # Compute slack variables
    S = bpiv[n:] - apiv[:, n:].T @ X
    
    # Check KKT conditions
    if np.all(S >= -eps):
        # Undo pivots in Y
        Y_out = np.zeros(m)
        for i in range(m):
            Y_out[jpiv[i]] = Y[i]
        obasis = jpiv[:n].copy() if return_basis else None
        return X, Y_out, 0, obasis
    
    # Begin iteration
    newsol = np.dot(bpiv[:n], Y[:n])
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
        bpiv[iexit], bpiv[ienter] = bpiv[ienter], bpiv[iexit]
        jpiv[iexit], jpiv[ienter] = jpiv[ienter], jpiv[iexit]
        
        # Update LU factorization
        try:
            lu, piv = linalg.lu_factor(apiv[:, :n])
        except linalg.LinAlgError:
            raise DualSimplexError(41)
        
        # Update dual solution
        try:
            Y[:n] = linalg.lu_solve((lu, piv), C)
        except linalg.LinAlgError:
            raise DualSimplexError(51)
        
        # Update primal solution
        try:
            X = linalg.lu_solve((lu, piv), bpiv[:n], trans=1)
        except linalg.LinAlgError:
            raise DualSimplexError(51)
        
        # Update slack variables
        S = bpiv[n:] - apiv[:, n:].T @ X
        
        # Check KKT conditions
        if np.all(S >= -eps):
            # Undo pivots in Y
            Y_out = np.zeros(m)
            for i in range(m):
                Y_out[jpiv[i]] = Y[i]
            obasis = jpiv[:n].copy() if return_basis else None
            return X, Y_out, 0, obasis
        
        # Update solutions
        oldsol = newsol
        newsol = np.dot(bpiv[:n], Y[:n])
    
    # Budget exceeded
    raise DualSimplexError(40)


def _pivot_dantzig(
    n: int, m: int, apiv: np.ndarray, Y: np.ndarray, S: np.ndarray,
    lu: np.ndarray, piv: np.ndarray, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Dantzig's minimum ratio method for fast convergence.
    """
    # Entering index: most negative slack
    ienter = np.argmin(S) + n
    
    # Build weight vector
    W = linalg.lu_solve((lu, piv), apiv[:, ienter])
    
    # Compute ratios and choose exiting index
    currmin = np.inf
    iexit = None
    
    for j in range(n):
        if W[j] < eps:
            continue
        ratio = Y[j] / W[j]
        if ratio < currmin:
            currmin = ratio
            iexit = j
    
    return ienter, iexit


def _pivot_bland(
    n: int, m: int, apiv: np.ndarray, Y: np.ndarray, S: np.ndarray,
    lu: np.ndarray, piv: np.ndarray, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Bland's anticycling rule for guaranteed convergence.
    """
    # Entering index: first negative slack
    ienter = None
    for j in range(m - n):
        if S[j] < -eps:
            ienter = j + n
            break
    
    if ienter is None:
        return n, None
    
    # Build weight vector
    W = linalg.lu_solve((lu, piv), apiv[:, ienter])
    
    # Compute ratios and choose exiting index
    currmin = np.inf
    iexit = None
    
    for j in range(n):
        if W[j] < eps:
            continue
        ratio = Y[j] / W[j]
        if ratio - currmin < -eps:
            currmin = ratio
            iexit = j
    
    return ienter, iexit


def feasible_basis(
    n: int,
    m: int,
    AT: np.ndarray,
    C: np.ndarray,
    eps: Optional[float] = None,
    ibudget: int = 50000
) -> Tuple[np.ndarray, int]:
    """
    Find a dual feasible basis for a primal problem using the auxiliary method.
    
    Parameters
    ----------
    n : int
        Number of variables in the primal problem
    m : int
        Number of constraints in the primal problem
    AT : np.ndarray
        Transpose of the constraint matrix A, shape (n, m)
    C : np.ndarray
        Cost vector, shape (n,)
    eps : float, optional
        Working precision (default: sqrt of machine epsilon)
    ibudget : int, optional
        Maximum number of pivots (default: 50000)
    
    Returns
    -------
    basis : np.ndarray
        Indices of a dual feasible basis for AT
    ierr : int
        Error code (0 = success)
    """
    # Set default eps
    if eps is None:
        eps = np.finfo(float).eps
    
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
    A_aux = np.zeros((n, n + m))
    A_aux[:, n:n+m] = AT
    
    B_aux = np.zeros(n + m)
    B_aux[:n] = 1.0
    
    # Create artificial variables
    for i in range(n):
        A_aux[i, i] = np.sign(C[i]) if C[i] != 0 else 1.0
    
    # Initial basis
    ibasis = np.arange(n)
    
    # Solve auxiliary problem
    X, Y_aux, ierr, basis = dualsimplex(
        n, n + m, A_aux, B_aux, C, ibasis,
        eps=eps, ibudget=ibudget, return_basis=True
    )
    
    if ierr != 0:
        raise DualSimplexError(ierr)
    
    # Check for infeasible dual solution
    if np.dot(Y_aux, B_aux) > eps:
        raise DualSimplexError(2)
    
    # Adjust basis indices
    basis = basis - n
    
    # Handle degeneracies - ensure all basis elements are legal
    for i in range(n):
        if basis[i] < 0:
            idx = 0
            while idx in basis:
                idx += 1
            basis[i] = idx
    
    return basis, 0


def solve_lp(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    eps: Optional[float] = None,
    ibudget: int = 50000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Convenience function to solve an LP from scratch.
    
    Solves:
        maximize     C^T X
        such that    A X <= B
    
    Parameters
    ----------
    A : np.ndarray
        Constraint matrix, shape (m, n)
    B : np.ndarray
        Upper bounds, shape (m,)
    C : np.ndarray
        Cost vector, shape (n,)
    eps : float, optional
        Working precision
    ibudget : int, optional
        Maximum pivots
    
    Returns
    -------
    X : np.ndarray
        Primal solution
    Y : np.ndarray
        Dual solution
    basis : np.ndarray
        Final basis indices
    ierr : int
        Error code
    """
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)
    
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
