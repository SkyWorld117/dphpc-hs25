import os
import torch
import matplotlib.pyplot as plt


torch.manual_seed(42)
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP = 2          # number of warmup runs
NUM_REPEATS = 10    # number of benchmark runs
TOP = 5             # number of problems to run

BLOCK_SIZE = 128


def get_sparse_matrix(sparsity:float, rows:int, cols:int):
    density = sparsity
    
    # Calculate approximate number of non-zero elements
    total_elements = rows * cols
    nnz = max(rows + cols, int(total_elements * density))  # At least one per row and column
    
    # Generate random indices for non-zero elements
    # First ensure at least one element per row and column
    row_indices = []
    col_indices = []
    values = []
    
    # Add one element per row (ensure coverage)
    for i in range(rows):
        j = torch.randint(0, cols, (1,), device=device).item()
        row_indices.append(i)
        col_indices.append(j)
        values.append(torch.rand(1, device=device).item())
    
    # Add one element per column (ensure coverage)
    for j in range(cols):
        i = torch.randint(0, rows, (1,), device=device).item()
        row_indices.append(i)
        col_indices.append(j)
        values.append(torch.rand(1, device=device).item())
    
    # Add remaining random elements
    remaining = nnz - len(row_indices)
    if remaining > 0:
        random_rows = torch.randint(0, rows, (remaining,), device=device)
        random_cols = torch.randint(0, cols, (remaining,), device=device)
        random_vals = torch.rand(remaining, device=device)
        
        row_indices.extend(random_rows.tolist())
        col_indices.extend(random_cols.tolist())
        values.extend(random_vals.tolist())
    
    # Create COO sparse tensor
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (rows, cols), device=device)
    
    # Coalesce to remove duplicate indices (sum their values)
    sparse_matrix = sparse_matrix.coalesce()
    vector = torch.rand(rows, device=device)

    return sparse_matrix, vector


def convert_format(sparse_matrix: torch.Tensor, format: str) -> torch.Tensor:
    if format == 'csr':
        return sparse_matrix.to_sparse_csr()
    elif format == 'csc':
        return sparse_matrix.to_sparse_csc()
    elif format == 'coo':
        return sparse_matrix.coalesce()
    elif format == 'bsc':
        return sparse_matrix.to_sparse_bsc(blocksize=(BLOCK_SIZE, BLOCK_SIZE))
    else:
        raise ValueError(f"Unknown format: {format}")

 
problems = '''
problem           rows    columns     nonzeros
==============================================
L1_sixm250obs   986069     428032      4280320
Linf_520c        93326      69004       566193
a2864            22117     200787     20078717
bdry2           376500     250998      1500003
cont1           160793      40398       399991
cont11          160793      80396       439989
datt256          11077     262144      1503732
dlr1           1735470    9121907     18365107
ex10             69609      17680      1179680
fhnw-bin1       772872    1141653      8611326
fome13           48569      97840       334984
graph40-40      360900     102600      1260900
irish-e         104260      61728       538809
neos            479120      36786      1084461
neos3           512209       6624      1542816
neos-3025225     91572      69846      9357951
neos5052403      38269      32868      4898304
neos-5251015    486531     136971      1955388
ns1687037        50622      43749      1406739
ns1688926        32768      16587      1712128
nug08-3rd        19728      20448       139008
pds-100         156244     505360      1390539
psched3-3       266228      79555      1062480
qap15             6331      22275       110700
rail02           95791     270869       756228
rail4284          4284    1092610     12372358
rmine15         358395      42438       879732
s82              87878    1690631      7022608
s100             14734     364417      2127672
s250r10          10963     273142      1572104
savsched1       295990     328575      1846351
scpm1             5000     500000      6250000
shs1023         133944     444625      1044725
square41         40161      62234     13628623
stat96v2         29089     957432      2852184
stormG2_1000    528186    1259121      4228817
stp3d           159488     204880       662128
support10       165685      14770       551152
tpl-tub-ws16   1154615     747691      4720567
woodlands09     194599     382147      2646003
Dual2_5000    30000600   33050602     93001800
Primal2_1000   1299380    2559380      5498140
thk_48         6366377    8609262     27802878
thk_63         5694387    7701112     21592414
L1_sixm1000obs 3082940    1426256     14262560
L2CTA3D         210000   10000000     30000000
degme           185501     659415      8127528
dlr2           7132926   38868107     78091589
set-cover        10000    1102008     20442268
'''

def benchmark():
    formats = ['csr', 'csc', 'coo', 'bsc']

    # Parse problem definitions
    matrix_sizes = []
    nnzs = []
    problem_names = []
    for line in problems.strip().split('\n')[2:]:
        parts = line.split()
        if len(parts) != 4:
            continue
        name, rows, cols, nnz = parts

        # Rows and cols need to be multiples of BLOCK_SIZE for bsc format
        rows = (int(rows) + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        cols = (int(cols) + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

        problem_names.append(name)
        matrix_sizes.append((int(rows), int(cols)))
        nnzs.append(int(nnz))

    assert len(matrix_sizes) == len(nnzs) == len(problem_names)

    matrix_sizes = matrix_sizes[:TOP]
    nnzs = nnzs[:TOP]
    problem_names = problem_names[:TOP]

    sparsities = [nnz / (rows * cols) for (rows, cols), nnz in zip(matrix_sizes, nnzs)]

    results = {}

    for sparsity, (rows, cols), problem_name in zip(sparsities, matrix_sizes, problem_names):
        print(f"\nBenchmarking problem: {problem_name}, sparsity: {sparsity}, size: {rows}x{cols}")

        sparse_matrix, vector = get_sparse_matrix(sparsity, rows, cols)

        if problem_name not in results:
            results[problem_name] = {}

        for fmt in formats:
            local_sparse = convert_format(sparse_matrix, fmt)

            # Warm-up
            for _ in range(WARMUP):
                torch.matmul(vector, local_sparse)

            # Benchmark
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(NUM_REPEATS):
                torch.matmul(vector, local_sparse)
            end_event.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
            avg_time = elapsed_time / NUM_REPEATS

            results[problem_name][fmt] = avg_time
            print(f"Format: {fmt}, Average Time: {avg_time:.4f} ms")

    return results


def plot_results(results, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # Get list of problems and formats
    problems = list(results.keys())
    if not problems:
        print("No results to plot")
        return

    formats = list(results[problems[0]].keys())

    # Create a single comparison chart with problems on x-axis
    plt.figure(figsize=(max(12, len(problems) * 2), 6))
    x = range(len(problems))
    width = 0.2

    for i, fmt in enumerate(formats):
        times = [results[problem][fmt] for problem in problems]
        offset = width * (i - len(formats)/2 + 0.5)
        plt.bar([xi + offset for xi in x], times, width, label=fmt)

    plt.xlabel('Problem')
    plt.ylabel('Average Time (ms)')
    plt.title('Benchmark Comparison Across All Problems')
    plt.xticks(x, problems, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'benchmark_results.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


results = benchmark()
plot_results(results)