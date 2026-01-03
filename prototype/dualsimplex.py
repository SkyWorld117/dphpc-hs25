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
import time
import concurrent.futures
from typing import Tuple, Optional

from linsys_solvers import (
    _compute_basis_inverse,
    _apply_eta_update,
    _solve_with_inverse,
    _compute_lu_factorization,
    _apply_forrest_tomlin_update,
    _solve_with_lu,
    _solve_with_conjugate_gradient
)

from pivot_methods import _pivot_dantzig, _pivot_bland, _pivot_dual_steepest_edge

# Device configuration - will use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # Use double precision for numerical stability
SOLVER = "pinv"  # Solver for linear systems ['cg', 'pinv', 'lu']
PIVOTING = "dual_steepest_edge"  # Pivoting rule ['dantzig', 'bland', 'dual_steepest_edge']

LOG_FREQ = 200 # Frequency of logging during iterations

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
    refactor_interval: int = 100,
    pivoting_strategies: Optional[Tuple[str, str]] = ("dual_steepest_edge","dantzig"),
    reconvene_interval: int = 100,
    maximize: bool = False,
    # --- new: race scheduling ---
    race_mode: str = "time",                     # "time" or "iterations"
    reconvene_time_s: float = 0.4,              # used when race_mode=="time"
    race_time_weights: Tuple[float, float] = (1.0, 1.0),
    race_iter_weights: Tuple[float, float] = (1.0, 1.0),  # used when race_mode=="iterations"
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
    pivoting_strategies : tuple of str, optional
        Two pivoting rules to race in parallel (default: None)
    reconvene_interval : int, optional
        Number of iterations between reconvene points when racing pivoting strategies (default: 100)
    maximize : bool, optional
        Whether to maximize or minimize the objective (default: True)
    race_mode : str, optional
        Mode for racing pivoting strategies: "time" or "iterations" (default: "time")
    reconvene_time_s : float, optional
        Time in seconds to allocate to each pivoting strategy before reconvening (default: 0.05)
    race_time_weights : tuple of float, optional
        Weights for allocating time budget to each pivoting strategy (default: (1.0, 1.0))
    race_iter_weights : tuple of float, optional
        Weights for allocating iteration budget to each pivoting strategy (default: (1.0, 1.0))
    
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

    B_basis = torch.empty((n, n), device=DEVICE, dtype=DTYPE)  # Basis matrix placeholder
    B_inv = torch.empty((n, n), device=DEVICE, dtype=DTYPE)  # Basis inverse placeholder
    LU = torch.empty((n, n), device=DEVICE, dtype=DTYPE)  # LU factorization placeholder
    pivots = torch.empty(n, device=DEVICE, dtype=torch.long)  # LU pivot indices placeholder
    
    if SOLVER == "pinv":
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

    elif SOLVER == "lu":
        # Get solution using LU factorization
        try:
            LU, pivots = _compute_lu_factorization(apiv[:, :n])
        except RuntimeError:
            raise DualSimplexError(32)
    
        # Solve for first N elements of dual solution
        try:
            Y[:n] = _solve_with_lu(LU, pivots, C)
        except RuntimeError:
            raise DualSimplexError(51)

    elif SOLVER == "cg":
        # Use conjugate gradient to solve linear systems
        B_basis = apiv[:, :n]
        try:
            Y[:n] = _solve_with_conjugate_gradient(B_basis, C)
        except RuntimeError:
            raise DualSimplexError(51)

    else:
        raise ValueError(f"Unknown solver: {SOLVER}")
    
    if torch.any(Y[:n] < -eps):
        raise DualSimplexError(33)
    
    Y[n:] = 0.0
    
    if SOLVER == "pinv":
        # Get primal solution
        try:
            X = _solve_with_inverse(B_inv, bpiv[:n], trans=True)
        except RuntimeError:
            raise DualSimplexError(51)
    elif SOLVER == "lu":
        # Get primal solution
        try:
            X = _solve_with_lu(LU, pivots, bpiv[:n], trans=True)
        except RuntimeError:
            raise DualSimplexError(51)
    elif SOLVER == "cg":
        # Use conjugate gradient to solve linear systems
        try:
            X = _solve_with_conjugate_gradient(B_basis, bpiv[:n], trans=True)
        except RuntimeError:
            raise DualSimplexError(51)
    else:
        raise ValueError(f"Unknown solver: {SOLVER}")
    
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

    def _compute_slack_inplace(apiv_local: torch.Tensor, bpiv_local: torch.Tensor, X_local: torch.Tensor, S_local: torch.Tensor) -> None:
        # S = bpiv[n:] - apiv[:, n:].T @ X
        torch.mv(apiv_local[:, n:].T, X_local, out=S_local)
        S_local.neg_().add_(bpiv_local[n:])

    def _refactor_and_resolve(
        apiv_local: torch.Tensor,
        bpiv_local: torch.Tensor,
        C_local: torch.Tensor,
        X_local: torch.Tensor,
        Y_local: torch.Tensor,
    ):
        """
        Full recompute of basis factorization and consistent X/Y for the current basis.
        Returns (B_inv_new, LU_new, pivots_new, B_basis_new).
        """
        if SOLVER == "pinv":
            B_inv_new = _compute_basis_inverse(apiv_local[:, :n])
            _solve_with_inverse(B_inv_new, C_local, out=Y_local[:n])
            _solve_with_inverse(B_inv_new, bpiv_local[:n], trans=True, out=X_local)
            return B_inv_new, None, None, None

        if SOLVER == "lu":
            LU_new, piv_new = _compute_lu_factorization(apiv_local[:, :n])
            _solve_with_lu(LU_new, piv_new, C_local, out=Y_local[:n])
            _solve_with_lu(LU_new, piv_new, bpiv_local[:n], trans=True, out=X_local)
            return None, LU_new, piv_new, None

        if SOLVER == "cg":
            B_basis_new = apiv_local[:, :n].clone()
            _solve_with_conjugate_gradient(B_basis_new, C_local, out=Y_local[:n])
            _solve_with_conjugate_gradient(B_basis_new, bpiv_local[:n], trans=True, out=X_local)
            return None, None, None, B_basis_new

        raise ValueError(f"Unknown solver: {SOLVER}")

    def _now() -> float:
        return time.perf_counter()

    def _run_block(rule: str, state: dict, *, max_iters: int, time_budget_s: Optional[float] = None, time_check_every: int = 8):
        """
        Run simplex iterations starting from `state`.
        Stops when:
          - reaches `max_iters`, OR
          - exceeds `time_budget_s` (if provided), OR
          - reaches optimal/unbounded.
        Returns: (status, state, obj, iters_done, elapsed_s)
          status: "continue" | "optimal" | "unbounded"
        """
        stream = None
        if state["apiv"].is_cuda:
            torch.cuda.set_device(state["apiv"].device.index or 0)
            stream = torch.cuda.Stream()

        def _body():
            apiv_l = state["apiv"]
            bpiv_l = state["bpiv"]
            jpiv_l = state["jpiv"]
            X_l = state["X"]
            Y_l = state["Y"]
            S_l = state["S"]

            B_inv_l = state.get("B_inv", None)
            LU_l = state.get("LU", None)
            piv_l = state.get("pivots", None)
            B_basis_l = state.get("B_basis", None)
            iters_since_ref = int(state.get("iters_since_refactor", 0))

            col_buf = torch.empty(n, device=apiv_l.device, dtype=apiv_l.dtype)
            eta_buf = torch.empty(n, device=apiv_l.device, dtype=apiv_l.dtype)
            pivot_row_buf_l = torch.empty(n, device=apiv_l.device, dtype=apiv_l.dtype)
            Y_n_buf_l = torch.empty(n, device=apiv_l.device, dtype=apiv_l.dtype)

            if rule == "dantzig":
                pivot_fn = _pivot_dantzig
            elif rule == "bland":
                pivot_fn = _pivot_bland
            elif rule == "dual_steepest_edge":
                pivot_fn = _pivot_dual_steepest_edge
            else:
                raise ValueError(f"Unknown pivoting rule: {rule}")

            iters_done = 0
            t0 = _now()

            for k in range(max_iters):
                # time budget check (avoid checking every iteration to reduce overhead)
                if time_budget_s is not None and (k % max(1, time_check_every) == 0):
                    if (_now() - t0) >= time_budget_s:
                        break

                ienter, iexit = pivot_fn(n, m, apiv_l, Y_l, S_l, SOLVER, B_basis_l, B_inv_l, LU_l, piv_l, eps)
                if iexit is None:
                    state["B_inv"] = B_inv_l
                    state["LU"] = LU_l
                    state["pivots"] = piv_l
                    state["B_basis"] = B_basis_l
                    state["iters_since_refactor"] = iters_since_ref
                    return "unbounded", iters_done

                # eta column BEFORE swap
                if SOLVER == "pinv":
                    _solve_with_inverse(B_inv_l, apiv_l[:, ienter], out=eta_buf)
                elif SOLVER == "lu":
                    _solve_with_lu(LU_l, piv_l, apiv_l[:, ienter], out=eta_buf)
                elif SOLVER == "cg":
                    _solve_with_conjugate_gradient(B_basis_l, apiv_l[:, ienter], out=eta_buf)
                else:
                    raise ValueError(f"Unknown solver: {SOLVER}")

                # swap columns
                col_buf.copy_(apiv_l[:, iexit])
                apiv_l[:, iexit].copy_(apiv_l[:, ienter])
                apiv_l[:, ienter].copy_(col_buf)

                # swap bpiv
                tmp_b = bpiv_l[iexit].item()
                bpiv_l[iexit] = bpiv_l[ienter]
                bpiv_l[ienter] = tmp_b

                # swap jpiv
                tmp_j = jpiv_l[iexit].item()
                jpiv_l[iexit] = jpiv_l[ienter]
                jpiv_l[ienter] = tmp_j

                # update factorization and resolve
                if SOLVER == "pinv":
                    iters_since_ref += 1
                    if iters_since_ref >= refactor_interval:
                        B_inv_l = _compute_basis_inverse(apiv_l[:, :n])
                        iters_since_ref = 0
                    else:
                        _apply_eta_update(B_inv_l, eta_buf, iexit, pivot_row_buf_l)

                    _solve_with_inverse(B_inv_l, C, out=Y_n_buf_l)
                    Y_l[:n].copy_(Y_n_buf_l)
                    _solve_with_inverse(B_inv_l, bpiv_l[:n], trans=True, out=X_l)

                elif SOLVER == "lu":
                    iters_since_ref += 1
                    if iters_since_ref >= refactor_interval:
                        LU_l, piv_l = _compute_lu_factorization(apiv_l[:, :n])
                        iters_since_ref = 0
                    else:
                        _apply_forrest_tomlin_update(LU_l, piv_l, eta_buf, iexit)

                    _solve_with_lu(LU_l, piv_l, C, out=Y_n_buf_l)
                    Y_l[:n].copy_(Y_n_buf_l)
                    _solve_with_lu(LU_l, piv_l, bpiv_l[:n], trans=True, out=X_l)

                elif SOLVER == "cg":
                    if B_basis_l is None or B_basis_l.shape != (n, n):
                        B_basis_l = apiv_l[:, :n].clone()
                    B_basis_l[:, iexit].copy_(apiv_l[:, iexit])
                    _solve_with_conjugate_gradient(B_basis_l, C, out=Y_n_buf_l)
                    Y_l[:n].copy_(Y_n_buf_l)
                    _solve_with_conjugate_gradient(B_basis_l, bpiv_l[:n], trans=True, out=X_l)

                _compute_slack_inplace(apiv_l, bpiv_l, X_l, S_l)
                iters_done += 1

                if torch.all(S_l >= -eps):
                    state["B_inv"] = B_inv_l
                    state["LU"] = LU_l
                    state["pivots"] = piv_l
                    state["B_basis"] = B_basis_l
                    state["iters_since_refactor"] = iters_since_ref
                    return "optimal", iters_done

            state["B_inv"] = B_inv_l
            state["LU"] = LU_l
            state["pivots"] = piv_l
            state["B_basis"] = B_basis_l
            state["iters_since_refactor"] = iters_since_ref
            return "continue", iters_done

        t_start = _now()
        if stream is not None:
            with torch.cuda.stream(stream):
                status, iters_done = _body()
            stream.synchronize()
        else:
            status, iters_done = _body()
        elapsed = _now() - t_start

        obj = float(torch.dot(C, state["X"]).item())
        return status, state, obj, iters_done, elapsed

    print("Starting dual simplex iterations...")
    start_time = time.time()

    if pivoting_strategies is not None:
        if race_mode not in ("time", "iterations"):
            raise ValueError("race_mode must be 'time' or 'iterations'")

        if race_mode == "iterations" and reconvene_interval <= 0:
            raise ValueError("reconvene_interval must be > 0 when race_mode=='iterations'")

        if race_mode == "time" and reconvene_time_s <= 0:
            raise ValueError("reconvene_time_s must be > 0 when race_mode=='time'")

        base_state = {
            "apiv": apiv,
            "bpiv": bpiv,
            "jpiv": jpiv,
            "X": X,
            "Y": Y,
            "S": S,
            "B_inv": B_inv if SOLVER == "pinv" else None,
            "LU": LU if SOLVER == "lu" else None,
            "pivots": pivots if SOLVER == "lu" else None,
            "B_basis": B_basis if SOLVER == "cg" else None,
            "iters_since_refactor": iters_since_refactor,
        }

        def _clone_state(st: dict) -> dict:
            out = {}
            for k, v in st.items():
                out[k] = v.clone() if isinstance(v, torch.Tensor) else v
            return out

        done_iters = 0
        r0, r1 = pivoting_strategies

        while done_iters < ibudget:
            s0 = _clone_state(base_state)
            s1 = _clone_state(base_state)

            # allocate budgets
            if race_mode == "time":
                w0, w1 = race_time_weights
                denom = max(1e-12, float(w0 + w1))
                t0 = reconvene_time_s * float(w0) / denom
                t1 = reconvene_time_s * float(w1) / denom
                max_iters0 = ibudget - done_iters
                max_iters1 = ibudget - done_iters
                kwargs0 = dict(max_iters=max_iters0, time_budget_s=t0)
                kwargs1 = dict(max_iters=max_iters1, time_budget_s=t1)
            else:
                w0, w1 = race_iter_weights
                # scale reconvene_interval by weights, but cap to remaining budget
                i0 = int(max(1, round(reconvene_interval * float(w0))))
                i1 = int(max(1, round(reconvene_interval * float(w1))))
                max_iters0 = min(i0, ibudget - done_iters)
                max_iters1 = min(i1, ibudget - done_iters)
                kwargs0 = dict(max_iters=max_iters0, time_budget_s=None)
                kwargs1 = dict(max_iters=max_iters1, time_budget_s=None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(_run_block, r0, s0, **kwargs0)
                f1 = ex.submit(_run_block, r1, s1, **kwargs1)
                status0, st0, obj0, it0, dt0 = f0.result()
                status1, st1, obj1, it1, dt1 = f1.result()

            # choose candidate
            candidates = []
            if status0 != "unbounded":
                candidates.append((status0, st0, obj0, r0, it0, dt0))
            if status1 != "unbounded":
                candidates.append((status1, st1, obj1, r1, it1, dt1))
            if not candidates:
                raise DualSimplexError(1)

            optimal = [c for c in candidates if c[0] == "optimal"]
            if optimal:
                pick = max(optimal, key=lambda t: t[2]) if maximize else min(optimal, key=lambda t: t[2])
            else:
                # TODO wtf is it doing here?
                pick = max(candidates, key=lambda t: t[2]) if maximize else min(candidates, key=lambda t: t[2])

            status_pick, base_state, obj_pick, picked_rule, it_pick, dt_pick = pick

            # full refactor at reconvene
            try:
                B_inv_new, LU_new, piv_new, B_basis_new = _refactor_and_resolve(
                    base_state["apiv"], base_state["bpiv"], C, base_state["X"], base_state["Y"]
                )
            except RuntimeError:
                raise DualSimplexError(41)

            base_state["B_inv"] = B_inv_new
            base_state["LU"] = LU_new
            base_state["pivots"] = piv_new
            base_state["B_basis"] = B_basis_new
            base_state["iters_since_refactor"] = 0
            _compute_slack_inplace(base_state["apiv"], base_state["bpiv"], base_state["X"], base_state["S"])

            done_iters += int(it_pick)

            if LOG_FREQ > 0:
                rate0 = (it0 / dt0) if dt0 > 0 else float("inf")
                rate1 = (it1 / dt1) if dt1 > 0 else float("inf")
                print(
                    f"{status_pick} pivots={done_iters} picked={picked_rule} obj={obj_pick:.6f} "
                    f"| {r0}: it={it0} dt={dt0*1e3:.1f}ms obj={obj0:.6f} rate={rate0:.1f} it/s "
                    f"| {r1}: it={it1} dt={dt1*1e3:.1f}ms obj={obj1:.6f} rate={rate1:.1f} it/s "
                    f"| elap={time.time() - start_time:.2f}s"
                )

            if torch.all(base_state["S"] >= -eps):
                Y_out = torch.zeros(m, device=DEVICE, dtype=DTYPE)
                Y_out.scatter_(0, base_state["jpiv"], base_state["Y"])
                obasis = base_state["jpiv"][:n].clone() if return_basis else None
                print(f"Dual simplex completed in {done_iters} pivots, time elapsed = {time.time() - start_time:.2f} seconds")
                return base_state["X"], Y_out, 0, obasis

        raise DualSimplexError(40)

    # ...existing single-strategy loop remains unchanged below...
    for iteration in range(ibudget):
        # Choose pivot rule based on improvement

        # if oldsol - newsol > eps:
        #     # Use Dantzig's rule
        #     ienter, iexit = _pivot_dantzig(n, m, apiv, Y, S, SOLVER, B_basis, B_inv, LU, pivots, eps)
        # else:
        #     # Use Bland's rule
        #     ienter, iexit = _pivot_bland(n, m, apiv, Y, S, SOLVER, B_basis, B_inv, LU, pivots, eps)

        ienter, iexit = None, None

        if PIVOTING == "dantzig":
            ienter, iexit = _pivot_dantzig(n, m, apiv, Y, S, SOLVER, B_basis, B_inv, LU, pivots, eps)
        elif PIVOTING == "bland":
            ienter, iexit = _pivot_bland(n, m, apiv, Y, S, SOLVER, B_basis, B_inv, LU, pivots, eps)
        elif PIVOTING == "dual_steepest_edge":
            ienter, iexit = _pivot_dual_steepest_edge(n, m, apiv, Y, S, SOLVER, B_basis, B_inv, LU, pivots, eps)
        else:
            raise ValueError(f"Unknown pivoting rule: {PIVOTING}")
        
        if iexit is None:
            # Dual unbounded
            raise DualSimplexError(1)
        
        if SOLVER == "pinv":
            # Compute eta column before swapping (reuse buffer)
            _solve_with_inverse(B_inv, apiv[:, ienter], out=eta_col)
        elif SOLVER == "lu":
            # Compute eta column before swapping (reuse buffer)
            _solve_with_lu(LU, pivots, apiv[:, ienter], out=eta_col)
        elif SOLVER == "cg":
            # Use conjugate gradient to solve linear systems
            _solve_with_conjugate_gradient(B_basis, apiv[:, ienter], out=eta_col)
        else:
            raise ValueError(f"Unknown solver: {SOLVER}")
        
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
        if SOLVER == "pinv":
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

        elif SOLVER == "lu":
            iters_since_refactor += 1
            if iters_since_refactor >= refactor_interval:
                # Full refactorization
                try:
                    LU, pivots = _compute_lu_factorization(apiv[:, :n])
                except RuntimeError:
                    raise DualSimplexError(41)
                iters_since_refactor = 0
            else:
                # Incremental update using Forrest--Tomlin style update
                try:
                    _apply_forrest_tomlin_update(LU, pivots, eta_col, iexit)
                except RuntimeError:
                    raise DualSimplexError(41)
        
            # Update dual solution (reuse buffer then copy)
            try:
                _solve_with_lu(LU, pivots, C, out=Y_n_buf)
                Y[:n].copy_(Y_n_buf)
            except RuntimeError:
                raise DualSimplexError(51)
            
            # Update primal solution in-place
            try:
                _solve_with_lu(LU, pivots, bpiv[:n], trans=True, out=X)
            except RuntimeError:
                raise DualSimplexError(51)

        elif SOLVER == "cg":
            # Use conjugate gradient to solve linear systems
            B_basis[:, iexit].copy_(apiv[:, iexit])
            try:
                _solve_with_conjugate_gradient(B_basis, C, out=Y_n_buf)
                Y[:n].copy_(Y_n_buf)
            except RuntimeError:
                raise DualSimplexError(51)
            try:
                _solve_with_conjugate_gradient(B_basis, bpiv[:n], trans=True, out=X)
            except RuntimeError:
                raise DualSimplexError(51)
        else:
            raise ValueError(f"Unknown solver: {SOLVER}")
        
        # Update slack variables in-place: S = bpiv[n:] - apiv[:, n:].T @ X
        torch.mv(apiv[:, n:].T, X, out=S)
        S.neg_().add_(bpiv[n:])
        
        # Check KKT conditions
        if torch.all(S >= -eps):
            # Undo pivots in Y using scatter (vectorized)
            Y_out = torch.zeros(m, device=DEVICE, dtype=DTYPE)
            Y_out.scatter_(0, jpiv, Y)
            obasis = jpiv[:n].clone() if return_basis else None
            print(f"Dual simplex completed in {iteration+1} iterations, time elapsed = {time.time() - start_time:.2f} seconds")
            return X, Y_out, 0, obasis

        if LOG_FREQ > 0 and iteration % LOG_FREQ == 0:
            print(f"Iteration {iteration+1}: Current objective value = {torch.dot(bpiv[:n], Y[:n]).item():.6f}, Time elapsed = {time.time() - start_time:.2f} seconds")
        
        # Update solutions
        oldsol = newsol
        newsol = torch.dot(bpiv[:n], Y[:n])
    
    # Budget exceeded
    raise DualSimplexError(40)


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
