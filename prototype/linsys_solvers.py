from typing import Tuple, Optional

import torch
import torch_linalg
import scipy
import numpy as np

CG_IMPL = "torch"  # Implementation of CG ['torch', 'scipy']

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


def _solve_with_inverse_multi(B_inv: torch.Tensor, b: torch.Tensor, trans: bool = False,
                              out: torch.Tensor = None) -> torch.Tensor:
    """Solve multiple linear systems using the basis inverse.
    
    Solves A X = B or A^T X = B for multiple right-hand sides (columns of B).
    
    Parameters
    ----------
    out : torch.Tensor, optional
        Preallocated output tensor to avoid allocation
    """
    if trans:
        # Solve A^T X = B => X = (A^{-1})^T B = B_inv^T @ B
        if out is not None:
            return torch.mm(B_inv.T, b, out=out)
        return B_inv.T @ b
    else:
        # Solve A X = B => X = A^{-1} B = B_inv @ B
        if out is not None:
            return torch.mm(B_inv, b, out=out)
        return B_inv @ b


def _solve_with_conjugate_gradient(B: torch.Tensor, b: torch.Tensor, trans: bool = False,
                                  out: torch.Tensor = None, maxiter: int = 1000) -> torch.Tensor:
    """Solve a linear system using conjugate gradient method.
    Parameters
    ----------
    out : torch.Tensor, optional
        Preallocated output tensor to avoid allocation
    """
    # We use CG on a symmetric positive (semi-)definite matrix formed
    # from the normal equations. To solve B x = b (non-transposed case)
    # we solve (B^T B) x = B^T b. To solve B^T x = b (transposed case)
    # we solve (B B^T) x = B b.
    if CG_IMPL == "scipy":
        if trans:
            A = (B @ B.T).cpu().numpy()
            rhs = (B @ b).cpu().numpy()
        else:
            A = (B.T @ B).cpu().numpy()
            rhs = (B.T @ b).cpu().numpy()

        x, info = scipy.sparse.linalg.cg(A, rhs, maxiter=maxiter)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient did not converge, info={info}")

        x_t = torch.tensor(x, device=B.device, dtype=B.dtype)
        if out is not None:
            out.copy_(x_t)
            return out
        return x_t
    elif CG_IMPL == "torch":
        if trans:
            A = (B @ B.T)
            rhs = (B @ b)
        else:
            A = (B.T @ B)
            rhs = (B.T @ b)

        x, info = torch_linalg.CG(A, rhs, max_iter=maxiter)
        if len(info[1]) > 0 or info[0] == maxiter:
            raise RuntimeError(f"Conjugate gradient did not converge, info={info}")

        if out is not None:
            out.copy_(x)
            return out
        return x
    else:
        raise ValueError(f"Unknown CG implementation: {CG_IMPL}")


def _solve_with_conjugate_gradient_multi(B: torch.Tensor, b: torch.Tensor, trans: bool = False,
                                        out: torch.Tensor = None, maxiter: int = 1000) -> torch.Tensor:
    """Solve multiple linear systems using conjugate gradient method.
    
    Solves A X = B or A^T X = B for multiple right-hand sides (columns of B).
    
    Parameters
    ----------
    out : torch.Tensor, optional
        Preallocated output tensor to avoid allocation
    """
    # We use CG on a symmetric positive (semi-)definite matrix formed
    # from the normal equations. To solve B X = C (non-transposed case)
    # we solve (B^T B) X = B^T C. To solve B^T X = C (transposed case)
    # we solve (B B^T) X = B C.
    if CG_IMPL == "scipy":
        if trans:
            A = (B @ B.T).cpu().numpy()
            rhs = (B @ b).cpu().numpy()
        else:
            A = (B.T @ B).cpu().numpy()
            rhs = (B.T @ b).cpu().numpy()

        x_list = []
        for i in range(rhs.shape[1]):
            x_col, info = scipy.sparse.linalg.cg(A, rhs[:, i], maxiter=maxiter)
            if info != 0:
                raise RuntimeError(f"Conjugate gradient did not converge for column {i}, info={info}")
            x_list.append(x_col)

        x = np.column_stack(x_list)
        x_t = torch.tensor(x, device=B.device, dtype=B.dtype)
        if out is not None:
            out.copy_(x_t)
            return out
        return x_t
    elif CG_IMPL == "torch":
        if trans:
            A = (B @ B.T)
            rhs = (B @ b)
        else:
            A = (B.T @ B)
            rhs = (B.T @ b)
        x, info = torch_linalg.CG(A, rhs, max_iter=maxiter)
        if len(info[1]) > 0 or info[0] == maxiter:
            raise RuntimeError(f"Conjugate gradient did not converge, info={info}")
        if out is not None:
            out.copy_(x)
            return out
        return x
    else:
        raise ValueError(f"Unknown CG implementation: {CG_IMPL}")


def _compute_lu_factorization(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute LU factorization of matrix A using torch.lu."""
    LU, pivots = torch.linalg.lu_factor(A)
    return LU, pivots


def _apply_forrest_tomlin_update(LU: torch.Tensor, pivots: torch.Tensor, eta_col: torch.Tensor, pivot_row: int) -> None:
    """
    Apply a Forrest--Tomlin style update to an LU factorization when the basis
    is updated by a single eta column replacement.

    This implementation reconstructs the current basis matrix from the packed
    `LU` and `pivots`, forms the new basis by replacing column `pivot_row`
    with `A @ eta_col` (since entering_col = B * eta_col), and then
    recomputes the LU factorization in-place. This is a robust (but not
    fully incremental) fallback that preserves the function contract.

    Parameters
    ----------
    LU : torch.Tensor
        Packed LU matrix returned by `torch.linalg.lu_factor` (shape (n,n)).
    pivots : torch.Tensor
        Pivot indices returned by `torch.linalg.lu_factor`.
    eta_col : torch.Tensor
        Eta column (length n) describing the column replacement in the basis
        such that entering_col = B @ eta_col.
    pivot_row : int
        Index of the column in the basis being replaced.
    """
    device = LU.device
    dtype = LU.dtype

    P, L, U = torch.lu_unpack(LU, pivots)
    B = P @ (L @ U)

    # Compute entering column: entering_col = B @ eta_col
    entering_col = B @ eta_col

    # Form the updated basis matrix by replacing the pivot_row column
    B_new = B.clone()
    B_new[:, pivot_row] = entering_col

    # Recompute LU factorization for the new basis and update in-place
    LU_new, piv_new = torch.linalg.lu_factor(B_new)

    LU.copy_(LU_new)
    # piv_new may be int tensor; ensure same dtype and device
    piv_new_t = piv_new.to(device=device)
    if pivots.shape != piv_new_t.shape:
        pivots.resize_(piv_new_t.shape)
    pivots.copy_(piv_new_t)


def _solve_with_lu(LU: torch.Tensor, pivots: torch.Tensor, b: torch.Tensor, trans: bool = False,
                  out: torch.Tensor = None) -> torch.Tensor:
    """Solve linear system using LU factorization."""
    if trans:
        # Solve A^T x = b where A = PLU
        # A^T = U^T L^T P^T
        P, L, U = torch.lu_unpack(LU, pivots)
        # Step 1: solve U^T w = b (U^T is lower triangular, non-unit diagonal)
        w = torch.linalg.solve_triangular(U.T, b.unsqueeze(-1), upper=False, unitriangular=False)
        # Step 2: solve L^T z = w (L^T is upper triangular, unit diagonal)
        z = torch.linalg.solve_triangular(L.T, w, upper=True, unitriangular=True)
        # Step 3: x = P z
        x = P @ z.squeeze(-1)           
    else:
        # Solve A x = b
        x = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(-1)).squeeze(-1)

    if out is not None:
        out.copy_(x)
        return out
    return x


def _solve_with_lu_multi(LU: torch.Tensor, pivots: torch.Tensor, b: torch.Tensor, trans: bool = False,
                        out: torch.Tensor = None) -> torch.Tensor:
    """Solve multiple linear systems using LU factorization."""
    if trans:
        # Solve A^T X = B where A = PLU
        # A^T = U^T L^T P^T
        P, L, U = torch.lu_unpack(LU, pivots)
        # Step 1: solve U^T W = B (U^T is lower triangular, non-unit diagonal)
        W = torch.linalg.solve_triangular(U.T, b, upper=False, unitriangular=False)
        # Step 2: solve L^T Z = W (L^T is upper triangular, unit diagonal)
        Z = torch.linalg.solve_triangular(L.T, W, upper=True, unitriangular=True)
        # Step 3: X = P Z
        X = P @ Z
    else:
        # Solve A X = B
        X = torch.linalg.lu_solve(LU, pivots, b)

    if out is not None:
        out.copy_(X)
        return out
    return X