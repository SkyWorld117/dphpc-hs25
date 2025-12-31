from typing import Tuple, Optional
import torch
import threading
from linsys_solvers import (
    _solve_with_conjugate_gradient,
    _solve_with_lu,
    _solve_with_inverse,
    _solve_with_conjugate_gradient_multi,
    _solve_with_lu_multi,
    _solve_with_inverse_multi
)

# Thread-local buffers so pivoting can run in parallel safely
_tls = threading.local()

def _tls_get(name: str, default):
    if not hasattr(_tls, name):
        setattr(_tls, name, default)
    return getattr(_tls, name)

def _tls_set(name: str, value):
    setattr(_tls, name, value)

def _ensure_pivot_buffers(n: int, device: torch.device, dtype: torch.dtype) -> None:
    """Ensure per-thread pivot buffers are allocated with correct size."""
    W = _tls_get("pivot_W", None)
    ratio = _tls_get("pivot_ratio", None)

    if W is None or W.shape[0] != n or W.device != device or W.dtype != dtype:
        W = torch.empty(n, device=device, dtype=dtype)
        ratio = torch.empty(n, device=device, dtype=dtype)
        _tls_set("pivot_W", W)
        _tls_set("pivot_ratio", ratio)

def _ensure_pivot_buffers_multi(n: int, m: int, device: torch.device, dtype: torch.dtype) -> None:
    Wm = _tls_get("pivot_W_multi", None)
    ratiom = _tls_get("pivot_ratio_multi", None)

    need_alloc = (
        Wm is None
        or Wm.shape[0] != n
        or Wm.shape[1] != (m - n)
        or Wm.device != device
        or Wm.dtype != dtype
    )
    if need_alloc:
        Wm = torch.empty((n, m - n), device=device, dtype=dtype)
        ratiom = torch.empty(m - n, device=device, dtype=dtype)
        _tls_set("pivot_W_multi", Wm)
        _tls_set("pivot_ratio_multi", ratiom)


def _pivot_dantzig(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, solver: str,
    B_basis: torch.Tensor, B_inv: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    _ensure_pivot_buffers(n, apiv.device, apiv.dtype)
    W = _tls_get("pivot_W", None)
    ratio = _tls_get("pivot_ratio", None)

    ienter = int(torch.argmin(S).item() + n)

    if solver == "pinv":
        _solve_with_inverse(B_inv, apiv[:, ienter], out=W)
    elif solver == "lu":
        _solve_with_lu(LU, pivots, apiv[:, ienter], out=W)
    elif solver == "cg":
        _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=W)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if not torch.any(W > eps):
        return ienter, None

    torch.div(Y[:n], W, out=ratio)
    iexit = int(torch.argmin(torch.where(W > eps, ratio, torch.inf)).item())
    return ienter, iexit


def _pivot_bland(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, solver: str,
    B_basis: torch.Tensor, B_inv: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    _ensure_pivot_buffers(n, apiv.device, apiv.dtype)
    W = _tls_get("pivot_W", None)
    ratio = _tls_get("pivot_ratio", None)

    neg_mask = S < -eps
    if not torch.any(neg_mask):
        return n, None

    ienter = int(neg_mask.nonzero(as_tuple=True)[0][0].item() + n)

    if solver == "pinv":
        _solve_with_inverse(B_inv, apiv[:, ienter], out=W)
    elif solver == "lu":
        _solve_with_lu(LU, pivots, apiv[:, ienter], out=W)
    elif solver == "cg":
        _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=W)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if not torch.any(W > eps):
        return ienter, None

    torch.div(Y[:n], W, out=ratio)
    iexit = int(torch.argmin(torch.where(W > eps, ratio, torch.inf)).item())
    return ienter, iexit


def _pivot_dual_steepest_edge(
    n: int, m: int, apiv: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, solver: str,
    B_basis: torch.Tensor, B_inv: torch.Tensor, LU: torch.Tensor, pivots: torch.Tensor, eps: float
) -> Tuple[int, Optional[int]]:
    _ensure_pivot_buffers(n, apiv.device, apiv.dtype)
    _ensure_pivot_buffers_multi(n, m, apiv.device, apiv.dtype)

    W = _tls_get("pivot_W", None)
    ratio = _tls_get("pivot_ratio", None)
    Wm = _tls_get("pivot_W_multi", None)
    ratiom = _tls_get("pivot_ratio_multi", None)

    neg_mask = S < -eps
    if not torch.any(neg_mask):
        return n, None

    if solver == "pinv":
        _solve_with_inverse_multi(B_inv, apiv[:, n:], out=Wm)
    elif solver == "lu":
        _solve_with_lu_multi(LU, pivots, apiv[:, n:], out=Wm)
    elif solver == "cg":
        _solve_with_conjugate_gradient_multi(B_basis, apiv[:, n:], out=Wm)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    norms_W = torch.norm(Wm, dim=0)
    torch.div(torch.abs(S), torch.where((S < -eps) & (norms_W > eps), norms_W, torch.inf), out=ratiom)
    ienter = int(torch.argmax(ratiom).item() + n)

    if solver == "pinv":
        _solve_with_inverse(B_inv, apiv[:, ienter], out=W)
    elif solver == "lu":
        _solve_with_lu(LU, pivots, apiv[:, ienter], out=W)
    elif solver == "cg":
        _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=W)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    if not torch.any(W > eps):
        return ienter, None

    torch.div(Y[:n], W, out=ratio)
    iexit = int(torch.argmin(torch.where(W > eps, ratio, torch.inf)).item())
    return ienter, iexit