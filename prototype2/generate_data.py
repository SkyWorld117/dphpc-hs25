"""
Generate test data to be interpolated by Delaunay.

Python version of generate_data.f90
Author: Tyler Chang (original Fortran)
"""

import numpy as np


def generate_delaunay_data(d: int = 20, n: int = 500, filename: str = "deldata.txt"):
    """
    Generate test data for Delaunay interpolation.
    
    Parameters
    ----------
    d : int
        Dimension of the data points (default: 20)
    n : int
        Number of data points to generate (default: 500)
    filename : str
        Output filename (default: 'deldata.txt')
    """
    # Generate random data points
    points = np.random.rand(n, d)
    
    # Compute the centroid as an interior point for interpolation
    centroid = np.mean(points, axis=0)
    
    # Write output to file
    with open(filename, 'w') as f:
        # Write metadata: D, N, M (number of interpolation points), dummy
        f.write(f"{d} {n} 1 0\n")
        
        # Write the data points
        for i in range(n):
            f.write(" ".join(f"{x:.15e}" for x in points[i]) + "\n")
        
        # Write the interpolation point (centroid)
        f.write(" ".join(f"{x:.15e}" for x in centroid) + "\n")
    
    print(f"Generated {n} points in {d} dimensions")
    print(f"Output written to {filename}")
    
    return points, centroid


if __name__ == "__main__":
    # Default problem dimensions - adjust these to generate problems of varying size
    D = 20  # Dimension
    N = 500  # Number of points
    
    generate_delaunay_data(D, N)
