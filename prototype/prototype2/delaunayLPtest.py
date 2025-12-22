"""
Delaunay LP Test

Python version of delaunayLPtest.f90 - Locate the Delaunay simplex containing
a point Q using linear programming.

Consider a set of points PTS = {P_i} in R^d, i = 1, ..., n.
Let A be a (d+1) x (d+1) matrix whose rows are given by: A_i = [p_i, 1];
let B be a n-vector B_i = p_i . p_i; and
let C be a (d+1)-vector C_i = -Q_i for i = 1, ..., d, C_{d+1} = -1.

If the problem
    max C^T X
    s.t. AX <= B
has a unique solution, then the vertices of the simplex S in DT(PTS) that
contains Q are given by the solution basis and the affine interpolation 
weights are given by the dual solution for the corresponding basis.

Author: Tyler Chang (original Fortran)
"""

import sys
import time
import numpy as np
from dualsimplex import dualsimplex, feasible_basis, DualSimplexError


def load_data(filepath: str):
    """
    Load data from input file.
    
    Parameters
    ----------
    filepath : str
        Path to the input data file
    
    Returns
    -------
    d : int
        Dimension of points
    n : int
        Number of training points
    m : int
        Number of interpolation points
    A : np.ndarray
        Constraint matrix (d+1, n)
    B : np.ndarray
        Upper bounds (n,)
    C : np.ndarray
        Interpolation points matrix (d+1, m)
    """
    with open(filepath, 'r') as f:
        # Read metadata from first line
        first_line = f.readline().split()
        d = int(first_line[0])
        n = int(first_line[1])
        m = int(first_line[2])
        # dummy = int(first_line[3])
        
        if d <= 0 or n <= 0 or m <= 0:
            raise ValueError("Illegal input dimensions in input file, line 1.")
        
        # Read training points
        A = np.zeros((d + 1, n))
        B = np.zeros(n)
        
        for i in range(n):
            line = f.readline().split()
            point = np.array([float(x) for x in line[:d]])
            A[:d, i] = point
            A[d, i] = 1.0
            B[i] = np.dot(point, point)
        
        # Negate A as in Fortran code
        A = -A
        
        # Read interpolation points
        C = np.zeros((d + 1, m))
        for i in range(m):
            line = f.readline().split()
            point = np.array([float(x) for x in line[:d]])
            C[:d, i] = point
            C[d, i] = 1.0
        
        # Negate C as in Fortran code
        C = -C
    
    return d, n, m, A, B, C


def interpolate_delaunay(filepath: str, verbose: bool = True):
    """
    Perform Delaunay interpolation using linear programming.
    
    Parameters
    ----------
    filepath : str
        Path to the input data file
    verbose : bool
        Whether to print results
    
    Returns
    -------
    simplices : list
        List of simplex vertex indices for each interpolation point
    weights : list
        List of interpolation weights for each point
    errors : list
        Error codes for each interpolation point
    elapsed_time : float
        Total computation time
    """
    # Load data
    d, n, m, A, B, C = load_data(filepath)
    
    # Set precision
    eps = np.sqrt(np.finfo(float).eps)
    
    # Allocate result arrays
    simplices = []
    weights = []
    errors = []
    
    # Time the interpolation
    start = time.time()
    
    for i in range(m):
        c_i = C[:, i]
        
        try:
            # Find initial feasible basis (Phase I)
            basis, ierr = feasible_basis(d + 1, n, A, c_i, eps=eps)
            
            if ierr == 0:
                # Solve the LP (Phase II)
                X = np.zeros(d + 1)
                Y = np.zeros(n)
                X, Y, ierr, simplex = dualsimplex(
                    d + 1, n, A, B, c_i, basis,
                    eps=eps, return_basis=True
                )
                
                if ierr == 0:
                    # Extract weights from dual solution
                    w = Y[simplex]
                    simplices.append(simplex)
                    weights.append(w)
                    errors.append(0)
                else:
                    simplices.append(None)
                    weights.append(None)
                    errors.append(ierr)
            else:
                simplices.append(None)
                weights.append(None)
                errors.append(ierr)
                
        except DualSimplexError as e:
            simplices.append(None)
            weights.append(None)
            errors.append(e.code)
    
    finish = time.time()
    elapsed_time = finish - start
    
    # Print results
    if verbose:
        for i in range(m):
            if errors[i] == 2:
                print(f"Extrapolation at point {i + 1}. No solution computed.")
            elif errors[i] != 0:
                print(f"Error at point {i + 1}. IERR = {errors[i]}")
            else:
                print(f"Interpolation point: {-C[:d, i]}")
                print(f"Simplex: {simplices[i] + 1}")  # 1-indexed for Fortran compatibility
                print(f"Weights: {weights[i]}")
                print()
        
        print(f"\n{m} points interpolated in {elapsed_time:.8e} seconds.\n")
    
    return simplices, weights, errors, elapsed_time


def main():
    """Main function to run the Delaunay LP test."""
    if len(sys.argv) < 2:
        print("Usage: python delaunayLPtest.py <datafile>")
        print("Example: python delaunayLPtest.py deldata.txt")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        interpolate_delaunay(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
