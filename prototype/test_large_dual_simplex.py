"""
Large-scale test cases for the Dual Simplex Solver

This module contains large test cases designed to benchmark GPU performance.
"""

import torch
import time
from dual_revised_simplex_solver import DualRevisedSimplexSolver, add_slack_variables


def test_large_transportation_problem(n_supply=100, n_demand=100, device='cpu'):
    """
    Test Case: Large Transportation Problem
    
    Transportation problem: minimize cost of shipping goods from supply points to demand points.
    
    Variables: x[i,j] = amount shipped from supply point i to demand point j
    
    minimize    sum(c[i,j] * x[i,j])
    subject to  sum_j x[i,j] = supply[i]  for all i
                sum_i x[i,j] = demand[j]  for all j
                x[i,j] >= 0
    
    Args:
        n_supply: Number of supply points
        n_demand: Number of demand points
        device: Device to run on ('cpu' or 'cuda')
    """
    print("=" * 80)
    print(f"Large Transportation Problem: {n_supply} supply x {n_demand} demand points")
    print(f"Device: {device}")
    print("=" * 80)
    
    dev = torch.device(device)
    
    # Problem dimensions
    n_vars = n_supply * n_demand
    n_constraints = n_supply + n_demand - 1  # One constraint is redundant
    
    print(f"\nProblem size:")
    print(f"  Variables: {n_vars:,}")
    print(f"  Constraints: {n_constraints:,}")
    
    # Generate random cost matrix
    torch.manual_seed(42)
    costs = torch.rand(n_supply, n_demand, device=dev) * 10 + 1
    c = costs.flatten()
    
    # Generate random supply and demand (balanced)
    supply = torch.rand(n_supply, device=dev) * 100 + 50
    demand = torch.rand(n_demand, device=dev) * 100 + 50
    
    # Balance supply and demand
    total_supply = supply.sum()
    demand = demand * (total_supply / demand.sum())
    
    # Build constraint matrix
    # Supply constraints: sum_j x[i,j] = supply[i]
    # Demand constraints: sum_i x[i,j] = demand[j]
    # We drop the last demand constraint (redundant)
    
    A_rows = []
    b_vec = []
    
    # Supply constraints
    for i in range(n_supply):
        row = torch.zeros(n_vars, device=dev)
        for j in range(n_demand):
            idx = i * n_demand + j
            row[idx] = 1.0
        A_rows.append(row)
        b_vec.append(supply[i])
    
    # Demand constraints (except last one)
    for j in range(n_demand - 1):
        row = torch.zeros(n_vars, device=dev)
        for i in range(n_supply):
            idx = i * n_demand + j
            row[idx] = 1.0
        A_rows.append(row)
        b_vec.append(demand[j])
    
    A = torch.stack(A_rows, dim=0)
    b = torch.stack(b_vec)
    
    print(f"\nConstraint matrix shape: {A.shape}")
    print(f"Memory usage: ~{(A.numel() + c.numel() + b.numel()) * 4 / 1024 / 1024:.2f} MB")
    
    # Create solver
    print("\nInitializing solver...")
    start_time = time.time()
    solver = DualRevisedSimplexSolver(c, A, b, max_iter=5000, device=device)
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f} seconds")
    
    # Create initial basis using a simple heuristic
    # Use the first n_constraints variables
    initial_basis = list(range(n_constraints))
    
    print("\nSolving...")
    start_time = time.time()
    x, obj_val, status = solver.solve(basis=initial_basis)
    solve_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Status: {status}")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Solve time: {solve_time:.4f} seconds")
    print(f"  Objective value: {obj_val:.6f}")
    
    # Verify solution
    if status == 'optimal':
        x_matrix = x.reshape(n_supply, n_demand)
        
        # Check supply constraints
        supply_satisfied = torch.allclose(x_matrix.sum(dim=1), supply, atol=1e-4)
        demand_satisfied = torch.allclose(x_matrix.sum(dim=0)[:n_demand-1], 
                                         demand[:n_demand-1], atol=1e-4)
        non_negative = torch.all(x >= -1e-6)
        
        print(f"\nConstraint verification:")
        print(f"  Supply constraints: {'✓' if supply_satisfied else '✗'}")
        print(f"  Demand constraints: {'✓' if demand_satisfied else '✗'}")
        print(f"  Non-negativity: {'✓' if non_negative else '✗'}")
    
    return status == 'optimal', solve_time


def test_large_production_planning(n_products=200, n_resources=100, device='cpu'):
    """
    Test Case: Large Production Planning Problem
    
    Production planning: maximize profit subject to resource constraints.
    
    maximize    sum(profit[i] * x[i])
    subject to  sum(resource_usage[j,i] * x[i]) <= resource_available[j]  for all j
                x[i] >= 0
    
    Args:
        n_products: Number of products
        n_resources: Number of resources
        device: Device to run on ('cpu' or 'cuda')
    """
    print("=" * 80)
    print(f"Large Production Planning: {n_products} products, {n_resources} resources")
    print(f"Device: {device}")
    print("=" * 80)
    
    dev = torch.device(device)
    
    # Generate problem data
    torch.manual_seed(123)
    
    # Profits (negative because we minimize)
    profits = torch.rand(n_products, device=dev) * 100 + 10
    c_orig = -profits
    
    # Resource usage matrix (sparse-ish)
    A_orig = torch.rand(n_resources, n_products, device=dev) * 5
    A_orig = torch.where(torch.rand_like(A_orig) > 0.7, A_orig, torch.zeros_like(A_orig))
    
    # Resource availability
    b = torch.rand(n_resources, device=dev) * 1000 + 500
    
    print(f"\nProblem size:")
    print(f"  Variables: {n_products:,}")
    print(f"  Constraints: {n_resources:,}")
    print(f"  Sparsity: {(A_orig == 0).sum().item() / A_orig.numel() * 100:.1f}% zeros")
    
    # Convert to standard form
    inequality_types = ['<='] * n_resources
    c, A, b = add_slack_variables(c_orig, A_orig, b, inequality_types, device=dev)
    
    print(f"\nStandard form:")
    print(f"  Variables (with slack): {c.shape[0]:,}")
    print(f"  Constraint matrix shape: {A.shape}")
    print(f"  Memory usage: ~{(A.numel() + c.numel() + b.numel()) * 4 / 1024 / 1024:.2f} MB")
    
    # Create solver
    print("\nInitializing solver...")
    start_time = time.time()
    solver = DualRevisedSimplexSolver(c, A, b, max_iter=5000, device=device)
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f} seconds")
    
    # Initial basis: slack variables
    initial_basis = list(range(n_products, n_products + n_resources))
    
    print("\nSolving...")
    start_time = time.time()
    x, obj_val, status = solver.solve(basis=initial_basis)
    solve_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Status: {status}")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Solve time: {solve_time:.4f} seconds")
    print(f"  Objective value (minimize): {obj_val:.6f}")
    print(f"  Profit (maximize): {-obj_val:.6f}")
    
    # Verify solution
    if status == 'optimal':
        x_prod = x[:n_products]
        
        # Check resource constraints
        resource_usage = A_orig @ x_prod
        constraints_satisfied = torch.all(resource_usage <= b + 1e-4)
        non_negative = torch.all(x >= -1e-6)
        
        print(f"\nConstraint verification:")
        print(f"  Resource constraints: {'✓' if constraints_satisfied else '✗'}")
        print(f"  Non-negativity: {'✓' if non_negative else '✗'}")
        
        # Statistics
        active_products = (x_prod > 1e-6).sum().item()
        print(f"  Active products: {active_products}/{n_products}")
    
    return status == 'optimal', solve_time


def test_large_network_flow(n_nodes=150, edge_density=0.1, device='cpu'):
    """
    Test Case: Large Network Flow Problem
    
    Minimum cost flow problem on a network.
    
    minimize    sum(cost[e] * flow[e])
    subject to  flow conservation at each node
                capacity constraints
                flow[e] >= 0
    
    Args:
        n_nodes: Number of nodes in the network
        edge_density: Probability of edge existence
        device: Device to run on ('cpu' or 'cuda')
    """
    print("=" * 80)
    print(f"Large Network Flow: {n_nodes} nodes, edge density {edge_density}")
    print(f"Device: {device}")
    print("=" * 80)
    
    dev = torch.device(device)
    torch.manual_seed(456)
    
    # Generate random network
    edges = []
    costs = []
    capacities = []
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and torch.rand(1).item() < edge_density:
                edges.append((i, j))
                costs.append(torch.rand(1, device=dev).item() * 10 + 1)
                capacities.append(torch.rand(1, device=dev).item() * 100 + 50)
    
    n_edges = len(edges)
    
    # Generate supply/demand (must sum to 0)
    supply = torch.randn(n_nodes, device=dev) * 50
    supply = supply - supply.mean()  # Ensure sum is 0
    
    print(f"\nProblem size:")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Variables: {n_edges:,}")
    print(f"  Constraints: {n_nodes - 1:,} (flow conservation, one redundant)")
    
    # Build constraint matrix (flow conservation)
    # For each node (except last): sum(in_flow) - sum(out_flow) = supply
    c = torch.tensor(costs, device=dev)
    
    A_rows = []
    b_vec = []
    
    for node in range(n_nodes - 1):  # Skip last node (redundant)
        row = torch.zeros(n_edges, device=dev)
        for edge_idx, (src, dst) in enumerate(edges):
            if dst == node:
                row[edge_idx] = 1.0  # Incoming flow
            elif src == node:
                row[edge_idx] = -1.0  # Outgoing flow
        A_rows.append(row)
        b_vec.append(supply[node])
    
    A = torch.stack(A_rows, dim=0)
    b = torch.stack(b_vec)
    
    print(f"\nConstraint matrix shape: {A.shape}")
    print(f"Memory usage: ~{(A.numel() + c.numel() + b.numel()) * 4 / 1024 / 1024:.2f} MB")
    
    # Create solver
    print("\nInitializing solver...")
    start_time = time.time()
    solver = DualRevisedSimplexSolver(c, A, b, max_iter=5000, device=device)
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f} seconds")
    
    # Initial basis
    initial_basis = list(range(min(n_nodes - 1, n_edges)))
    
    print("\nSolving...")
    start_time = time.time()
    x, obj_val, status = solver.solve(basis=initial_basis)
    solve_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Status: {status}")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Solve time: {solve_time:.4f} seconds")
    print(f"  Objective value: {obj_val:.6f}")
    
    # Verify solution
    if status == 'optimal':
        # Check flow conservation
        flow_conservation = torch.allclose(A @ x, b, atol=1e-4)
        non_negative = torch.all(x >= -1e-6)
        
        print(f"\nConstraint verification:")
        print(f"  Flow conservation: {'✓' if flow_conservation else '✗'}")
        print(f"  Non-negativity: {'✓' if non_negative else '✗'}")
        
        # Statistics
        active_edges = (x > 1e-6).sum().item()
        print(f"  Active edges: {active_edges}/{n_edges}")
    
    return status == 'optimal', solve_time


def benchmark_cpu_vs_gpu():
    """
    Benchmark CPU vs GPU performance on large problems.
    """
    print("\n" + "=" * 80)
    print("CPU vs GPU BENCHMARK")
    print("=" * 80)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("\nCUDA not available. Skipping GPU benchmarks.")
        print("Running CPU-only tests...\n")
        devices = ['cpu']
    else:
        print(f"\nCUDA available: {cuda_available}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
        devices = ['cpu', 'cuda']
    
    results = []
    
    # Test 1: Production Planning
    for device in devices:
        try:
            success, solve_time = test_large_production_planning(
                n_products=200, n_resources=100, device=device
            )
            results.append((f"Production Planning ({device})", success, solve_time))
            print()
        except Exception as e:
            print(f"\nProduction Planning on {device} failed: {e}\n")
            results.append((f"Production Planning ({device})", False, 0))
    
    # Test 2: Network Flow
    for device in devices:
        try:
            success, solve_time = test_large_network_flow(
                n_nodes=150, edge_density=0.1, device=device
            )
            results.append((f"Network Flow ({device})", success, solve_time))
            print()
        except Exception as e:
            print(f"\nNetwork Flow on {device} failed: {e}\n")
            results.append((f"Network Flow ({device})", False, 0))
    
    # Test 3: Transportation (smaller for feasibility)
    for device in devices:
        try:
            success, solve_time = test_large_transportation_problem(
                n_supply=50, n_demand=50, device=device
            )
            results.append((f"Transportation ({device})", success, solve_time))
            print()
        except Exception as e:
            print(f"\nTransportation on {device} failed: {e}\n")
            results.append((f"Transportation ({device})", False, 0))
    
    # Summary
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    for name, success, solve_time in results:
        status = "✓" if success else "✗"
        time_str = f"{solve_time:.4f}s" if solve_time > 0 else "N/A"
        print(f"{status} {name}: {time_str}")
    
    # Speedup comparison
    if len(devices) > 1:
        print("\n" + "=" * 80)
        print("SPEEDUP (CPU vs GPU)")
        print("=" * 80)
        
        test_names = ["Production Planning", "Network Flow", "Transportation"]
        for i, test_name in enumerate(test_names):
            cpu_time = results[i * 2][2]
            gpu_time = results[i * 2 + 1][2]
            
            if cpu_time > 0 and gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"{test_name}: {speedup:.2f}x")
            else:
                print(f"{test_name}: N/A")


def main():
    """Run all large test cases."""
    print("\n" + "=" * 80)
    print("LARGE-SCALE DUAL SIMPLEX SOLVER TEST SUITE")
    print("=" * 80)
    
    benchmark_cpu_vs_gpu()


if __name__ == "__main__":
    main()
