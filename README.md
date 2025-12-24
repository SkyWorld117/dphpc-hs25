# Minimal Dual Simplex Solver

A standalone, minimal dual-simplex LP solver extracted from cuOpt with integrated MPS parser, presolve, and postsolve capabilities.

## Features

- **MPS Parser**: Read standard MPS format linear programming problem files
- **Dual Simplex Algorithm**: Robust implementation with:
  - Presolve and postsolve for problem reduction
  - Phase I (finding initial feasible basis) and Phase II (optimization)
  - Basis factorization and updates
  - Ratio tests and pivot selection
  - Crossover from barrier solutions
- **Standalone**: No Python dependencies, direct C++/CUDA binary

## Directory Structure

```
.
├── main.cpp                          # CLI entry point
├── CMakeLists.txt                    # Build configuration
├── libmps_parser/                    # MPS file parser
│   ├── include/mps_parser/
│   └── src/
├── src/
│   ├── dual_simplex/                 # Core dual simplex solver
│   │   ├── solve.cpp/.hpp            # Main solve interface
│   │   ├── presolve.cpp/.hpp         # Presolve/postsolve
│   │   ├── phase1.cpp/.hpp           # Phase I (feasibility)
│   │   ├── phase2.cpp/.hpp           # Phase II (optimization)
│   │   ├── basis_solves.cpp/.hpp     # Basis factorization
│   │   ├── basis_updates.cpp/.hpp    # Basis updates
│   │   └── ... (other components)
│   ├── linear_programming/           # Glue layer
│   │   ├── optimization_problem.cu
│   │   └── solve.cu
│   └── utilities/                    # Logging and helpers
└── include/cuopt/                    # Public headers
```

## Build Requirements

- CMake 3.20 or higher
- CUDA Toolkit 11.0+ (with nvcc, cuBLAS, cuSPARSE)
- C++17 compatible compiler (g++ 9+, clang 10+)
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)

## Building

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# The binary will be at: build/dual_simplex_solver
```

## Usage

```bash
# Basic usage
./build/dual_simplex_solver problem.mps

# Disable presolve
./build/dual_simplex_solver problem.mps --no-presolve

# Log to file
./build/dual_simplex_solver problem.mps --log-file solver.log

# Show help
./build/dual_simplex_solver --help
```

## Testing

Example MPS files for testing are available in the original cuOpt repository:
- `cuopt/datasets/mip/` - MIP problems (can solve LP relaxation)
- `cuopt/datasets/linear_programming/` - Pure LP problems

```bash
# Test with an example problem
./build/dual_simplex_solver cuopt/datasets/mip/example.mps
```

## Algorithm Overview

The dual simplex method works as follows:

1. **MPS Parsing**: Load problem in standard MPS format
2. **Presolve** (optional): Reduce problem size by:
   - Removing fixed variables
   - Eliminating singleton rows/columns
   - Tightening bounds
   - Detecting redundant constraints
3. **Phase I**: Find an initial dual-feasible basis (if not already provided)
4. **Phase II**: Iterate to optimality:
   - Select leaving variable (most infeasible primal variable)
   - Compute dual ray and reduced costs
   - Select entering variable (ratio test)
   - Update basis factorization
   - Check termination conditions
5. **Postsolve**: Map solution back to original problem space
6. **Output**: Objective value, solution vector, and statistics

## Performance Notes

- The solver uses CUDA for sparse linear algebra operations (cuBLAS, cuSPARSE)
- Presolve is highly recommended for real-world problems (can reduce problem size by 50%+)
- For very large problems, consider adjusting CUDA architecture targets in CMakeLists.txt

## Relevant Files from cuOpt

This minimal solver extracts the following key components:

### MPS Parser (cpp/libmps_parser/)
- `parser.cpp/hpp` - MPS file parsing
- `mps_data_model.cpp/hpp` - Internal data representation

### Dual Simplex Core (cpp/src/dual_simplex/)
- `solve.cpp/hpp` - Main solver entry point
- `presolve.cpp/hpp` - Presolve and postsolve logic
- `phase1.cpp/hpp` - Phase I (finding feasible basis)
- `phase2.cpp/hpp` - Phase II (optimization)
- `basis_solves.cpp/hpp` - LU factorization and solves
- `basis_updates.cpp/hpp` - Basis update operations
- `initial_basis.cpp/hpp` - Initial basis construction
- `crossover.cpp/hpp` - Crossover from IPM solutions
- `sparse_matrix.cpp/hpp` - Sparse matrix operations
- `vector_math.cpp/hpp` - Vector operations

### Utilities
- `logger.cpp/hpp` - Logging infrastructure
- `copy_helpers.hpp` - Memory copy utilities

## License

This code is extracted from NVIDIA cuOpt. See the original LICENSE file in the cuOpt repository for licensing terms (Apache 2.0).

## Troubleshooting

### CUDA Architecture Mismatch
If you see CUDA errors about unsupported architecture:
```bash
# Edit CMakeLists.txt and adjust CUDA_ARCHITECTURES to match your GPU
# Common values: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 3090), 89 (RTX 4090), 90 (H100)
```

### Missing Dependencies
```bash
# Install CUDA toolkit (Ubuntu/Debian)
sudo apt-get install nvidia-cuda-toolkit

# Verify CUDA is available
nvcc --version
```

### Build Errors
Check that all source files have correct include paths. The solver expects:
- `#include <dual_simplex/...>` to resolve to `src/dual_simplex/`
- `#include <mps_parser/...>` to resolve to `libmps_parser/include/mps_parser/`
- `#include <cuopt/...>` to resolve to `include/cuopt/`

## Further Development

This minimal solver provides the core dual-simplex algorithm. For additional features, refer to the full cuOpt repository:
- MIP solving (branch-and-bound)
- PDLP (First-order method)
- Advanced presolve techniques
- Warm-starting
- Solution callbacks
