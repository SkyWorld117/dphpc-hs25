from typing import Tuple, Optional
import torch
from linsys_solvers import (
    _solve_with_conjugate_gradient,
    _solve_with_lu,
    _solve_with_inverse,
    _solve_with_conjugate_gradient_multi,
    _solve_with_lu_multi,
    _solve_with_inverse_multi
)

# Preallocated buffers for pivot functions (module-level to avoid repeated allocation)
_pivot_W: torch.Tensor = None
_pivot_ratio: torch.Tensor = None

_pivot_W_multi: torch.Tensor = None
_pivot_ratio_multi: torch.Tensor = None


def _ensure_pivot_buffers(n: int, device: torch.device, dtype: torch.dtype) -> None:
    """Ensure pivot buffers are allocated with correct size."""
    global _pivot_W, _pivot_ratio
    if _pivot_W is None or _pivot_W.shape[0] != n or _pivot_W.device != device:
        _pivot_W = torch.empty(n, device=device, dtype=dtype)
        _pivot_ratio = torch.empty(n, device=device, dtype=dtype)


def _ensure_pivot_buffers_multi(n: int, m: int, device: torch.device, dtype: torch.dtype) -> None:
    global _pivot_W_multi, _pivot_ratio_multi
    need_alloc = False
    if _pivot_W_multi is None:
        need_alloc = True
    else:
        # Ensure both dimensions and device/dtype match expected sizes
        if _pivot_W_multi.shape[0] != n or _pivot_W_multi.shape[1] != (m - n):
            need_alloc = True
        if _pivot_W_multi.device != device or _pivot_W_multi.dtype != dtype:
            need_alloc = True

    if need_alloc:
        _pivot_W_multi = torch.empty((n, m - n), device=device, dtype=dtype)
        _pivot_ratio_multi = torch.empty(m - n, device=device, dtype=dtype)


def _pivot_dantzig(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, solver:str,
    B_basis:torch.Tensor, B_inv: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Dantzig's minimum ratio method for fast convergence.
    """
    _ensure_pivot_buffers(n, apiv.device, apiv.dtype)
    
    # Entering index: most negative slack
    ienter = int(torch.argmin(S).item() + n)
    
    if (solver == "pinv"):
        # Build weight vector using basis inverse (reuse buffer)
        _solve_with_inverse(B_inv, apiv[:, ienter], out=_pivot_W)
    elif (solver == "lu"):
        # Build weight vector using LU factorization (reuse buffer)
        _solve_with_lu(LU, pivots, apiv[:, ienter], out=_pivot_W)
    elif (solver == "cg"):
        # Use conjugate gradient to solve linear systems
        _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=_pivot_W)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Compute ratios and choose exiting index
    if not torch.any(_pivot_W > eps):
        return ienter, None

    # Compute ratio in-place
    torch.div(Y[:n], _pivot_W, out=_pivot_ratio)
    iexit = int(torch.argmin(torch.where(_pivot_W > eps, _pivot_ratio, torch.inf)).item())
    
    return ienter, iexit


def _pivot_bland(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, solver:str,
    B_basis:torch.Tensor, B_inv: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using Bland's anticycling rule for guaranteed convergence.
    """
    _ensure_pivot_buffers(n, apiv.device, apiv.dtype)
    
    # Entering index: first negative slack
    neg_mask = S < -eps
    if not torch.any(neg_mask):
        return n, None
    
    ienter = int(neg_mask.nonzero(as_tuple=True)[0][0].item() + n)
    
    if (solver == "pinv"):
        # Build weight vector using basis inverse (reuse buffer)
        _solve_with_inverse(B_inv, apiv[:, ienter], out=_pivot_W)
    elif (solver == "lu"):
        # Build weight vector using LU factorization (reuse buffer)
        _solve_with_lu(LU, pivots, apiv[:, ienter], out=_pivot_W)
    elif (solver == "cg"):
        # Use conjugate gradient to solve linear systems
        _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=_pivot_W)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Compute ratios and choose exiting index
    if not torch.any(_pivot_W > eps):
        return ienter, None

    # Compute ratio in-place
    torch.div(Y[:n], _pivot_W, out=_pivot_ratio)
    iexit = int(torch.argmin(torch.where(_pivot_W > eps, _pivot_ratio, torch.inf)).item())
    
    return ienter, iexit


def _pivot_dual_steepest_edge(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, solver:str,
    B_basis:torch.Tensor, B_inv: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    """
    Pivot using steepest edge rule for faster convergence.
    Like Dantzig, but uses steepest edge criterion to select entering variable.
    The per iteration cost can be improved by batching the solves for all candidates. (Not implemented here.)
    """
    _ensure_pivot_buffers(n, apiv.device, apiv.dtype)
    _ensure_pivot_buffers_multi(n, m, apiv.device, apiv.dtype)

    # Find all negative slacks (candidates for entering)
    neg_mask = S < -eps
    if not torch.any(neg_mask):
        return n, None  # No negative slack, optimal solution

    # Steepest edge: select entering variable that maximizes |S[j]| / ||B^{-1} * a_j||
    # This approximates the rate of objective improvement per unit step
    ienter = None
    # Vectorized steepest edge selection
    if solver == "pinv":
        _solve_with_inverse_multi(B_inv, apiv[:, n:], out=_pivot_W_multi)
    elif solver == "lu":
        _solve_with_lu_multi(LU, pivots, apiv[:, n:], out=_pivot_W_multi)
    elif solver == "cg":
        _solve_with_conjugate_gradient_multi(B_basis, apiv[:, n:], out=_pivot_W_multi)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    norms_W = torch.norm(_pivot_W_multi, dim=0)
    # valid_indices = torch.where(neg_mask & (norms_W > eps))[0]
    # if valid_indices.numel() > 0:
    #     ratios = torch.abs(S[valid_indices]) / norms_W[valid_indices]
    #     best_idx = torch.argmax(ratios).item()
    #     ienter = int(valid_indices[best_idx].item() + n)
    torch.div(torch.abs(S), torch.where((S < -eps) & (norms_W > eps), norms_W, torch.inf), out=_pivot_ratio_multi)
    ienter = int(torch.argmax(_pivot_ratio_multi).item() + n)

    if ienter is None:
        return n, None  # Should not happen if neg_mask is True

    # Now compute exiting variable using minimum ratio test
    # Recompute weight vector for the selected entering variable
    if solver == "pinv":
        _solve_with_inverse(B_inv, apiv[:, ienter], out=_pivot_W)
    elif solver == "lu":
        _solve_with_lu(LU, pivots, apiv[:, ienter], out=_pivot_W)
    elif solver == "cg":
        _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=_pivot_W)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Check for unboundedness
    if not torch.any(_pivot_W > eps):
        return ienter, None  # Unbounded

    # Compute ratios: Y[:n] / W and select minimum
    torch.div(Y[:n], _pivot_W, out=_pivot_ratio)
    iexit = int(torch.argmin(torch.where(_pivot_W > eps, _pivot_ratio, torch.inf)).item())

    return ienter, iexit