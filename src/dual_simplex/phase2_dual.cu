#include <cstdio>
#include <dual_simplex/phase2_dual.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>


namespace cuopt::linear_programming::dual_simplex {

namespace phase2_cu {

template <typename i_t>
__global__ void count_basis_rows_kernel(i_t m, const i_t *__restrict__ A_col_start,
                                        const i_t *__restrict__ A_row_ind,
                                        const i_t *__restrict__ basic_list,
                                        i_t *__restrict__ row_counts) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in A
    i_t start = A_col_start[col_idx];
    i_t end = A_col_start[col_idx + 1];

    for (i_t k = start; k < end; ++k) {
        i_t row = A_row_ind[k];
        atomicAdd(&row_counts[row], 1);
    }
}

template <typename i_t, typename f_t>
__global__ void
fill_basis_csr_kernel(i_t m, const i_t *__restrict__ A_col_start, const i_t *__restrict__ A_row_ind,
                      const f_t *__restrict__ A_values, const i_t *__restrict__ basic_list,
                      i_t *__restrict__ write_offsets, i_t *__restrict__ B_col_ind,
                      f_t *__restrict__ B_values) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in A
    // The column index in B is 'idx' because B = A[:, basic_list]

    i_t start = A_col_start[col_idx];
    i_t end = A_col_start[col_idx + 1];

    for (i_t k = start; k < end; ++k) {
        i_t row = A_row_ind[k];
        f_t val = A_values[k];

        // Get position to write using atomic increment on the row offset
        // This ensures thread-safe writing when multiple columns contribute to the
        // same row
        i_t pos = atomicAdd(&write_offsets[row], 1);

        B_col_ind[pos] = idx; // Column index in B
        B_values[pos] = val;
    }
}

// Helper function to orchestrate the basis construction on device
template <typename i_t, typename f_t>
void build_basis_on_device(i_t m, const i_t *d_A_col_start, const i_t *d_A_row_ind,
                           const f_t *d_A_values, const i_t *d_basic_list, i_t *&d_B_row_ptr,
                           i_t *&d_B_col_ind, f_t *&d_B_values, i_t *nz_B) {
    // 1. Allocate and compute row counts
    // Initialize row counts to 0
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMemset(d_B_row_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_B_row_ptr");
    assert(d_B_row_ptr != nullptr);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    count_basis_rows_kernel<<<grid_size, block_size>>>(m, d_A_col_start, d_A_row_ind, d_basic_list,
                                                       d_B_row_ptr);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_basis_rows_kernel");

    // 2. Prefix sum to get row pointers
    // We use thrust for exclusive scan. d_B_row_ptr currently holds counts.
    // exclusive_scan on [0, m+1) will transform counts to offsets.
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_B_row_ptr);
    thrust::exclusive_scan(dev_ptr, dev_ptr + m + 1, dev_ptr);

    // Get total NNZ (stored in the last element after scan)
    i_t total_nz;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&total_nz, d_B_row_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy total_nz");
    *nz_B = total_nz;

    // 3. Allocate B column indices and values
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ind, total_nz * sizeof(i_t)), "cudaMalloc d_B_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_values, total_nz * sizeof(f_t)), "cudaMalloc d_B_values");

    // 4. Fill CSR structure
    // We need a working copy of row pointers to use as write offsets because
    // atomicAdd will modify them.
    i_t *d_write_offsets;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_write_offsets, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_write_offsets");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_write_offsets, d_B_row_ptr, (m + 1) * sizeof(i_t), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_write_offsets");

    fill_basis_csr_kernel<<<grid_size, block_size>>>(m, d_A_col_start, d_A_row_ind, d_A_values,
                                                     d_basic_list, d_write_offsets, d_B_col_ind,
                                                     d_B_values);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_basis_csr_kernel");

    CUDA_CALL_AND_CHECK(cudaFree(d_write_offsets), "cudaFree d_write_offsets");
}

template <typename i_t>
__global__ void count_basis_transpose_rows_kernel(
    i_t m, const i_t *__restrict__ A_col_start,
    const i_t *__restrict__ A_row_ind, const i_t *__restrict__ basic_list,
    i_t *__restrict__ row_counts
) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in A
    i_t start = A_col_start[col_idx];
    i_t end = A_col_start[col_idx + 1];

    // Each non-zero in column 'col_idx' of A contributes to a row in B_T
    row_counts[idx] = end - start;
}

template <typename i_t, typename f_t>
__global__ void fill_basis_transpose_csr_kernel(
    i_t m, const i_t *__restrict__ A_col_start,
    const i_t *__restrict__ A_row_ind, const f_t *__restrict__ A_values,
    const i_t *__restrict__ basic_list, i_t *__restrict__ Bt_row_ptr,
    i_t *__restrict__ Bt_col_ind, f_t *__restrict__ Bt_values
) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in A
    i_t start = A_col_start[col_idx];
    i_t end = A_col_start[col_idx + 1];

    for (i_t k = start; k < end; ++k) {
        i_t row = A_row_ind[k];
        f_t val = A_values[k];

        // Position to write in B_T
        i_t pos = Bt_row_ptr[idx] + (k - start);

        Bt_col_ind[pos] = row; // Column index in B_T
        Bt_values[pos] = val;
    }
}

template <typename i_t, typename f_t>
void build_basis_transpose_on_device(
    i_t m, const i_t *d_A_col_start,
    const i_t *d_A_row_ind, const f_t *d_A_values, const i_t *d_basic_list,
    i_t *&d_Bt_row_ptr, i_t *&d_Bt_col_ind, f_t *&d_Bt_values
) {
    // 1. Allocate and compute column counts for B_T
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_Bt_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMemset(d_Bt_row_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_Bt_row_ptr");
    assert(d_Bt_row_ptr != nullptr);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    count_basis_transpose_rows_kernel<<<grid_size, block_size>>>(
        m, d_A_col_start, d_A_row_ind, d_basic_list, d_Bt_row_ptr
    );
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_basis_transpose_rows_kernel");

    // 2. Prefix sum to get row pointers for B_T
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_Bt_row_ptr);
    thrust::exclusive_scan(dev_ptr, dev_ptr + m + 1, dev_ptr);

    // Get total NNZ for B_T
    i_t total_nz;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&total_nz, d_Bt_row_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy total_nz for B_T");

    // 3. Allocate B_T column indices and values
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_col_ind, total_nz * sizeof(i_t)), "cudaMalloc d_Bt_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_values, total_nz * sizeof(f_t)), "cudaMalloc d_Bt_values");

    // 4. Fill CSR structure for B_T
    fill_basis_transpose_csr_kernel<<<grid_size, block_size>>>(
        m, d_A_col_start, d_A_row_ind, d_A_values, d_basic_list,
        d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values
    );
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_basis_transpose_csr_kernel");
}

template <typename i_t, typename f_t>
__global__ void fill_basis_transpose_dense_kernel(
    i_t m, const i_t *__restrict__ A_col_start,
    const i_t *__restrict__ A_row_ind, const f_t *__restrict__ A_values,
    const i_t *__restrict__ basic_list, f_t *__restrict__ Bt_dense
) {
    // A (CSC) => B_T (dense, col-major)
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in A
    i_t start = A_col_start[col_idx];
    i_t end = A_col_start[col_idx + 1];

    for (i_t k = start; k < end; ++k) {
        i_t row = A_row_ind[k];
        f_t val = A_values[k];

        // In dense B_T, position is (idx, row)
        Bt_dense[row * m + idx] = val; // Column-major storage
    }
}

template<typename i_t, typename f_t>
void build_basis_transpose_on_device_dense(
    i_t m, const i_t *d_A_col_start,
    const i_t *d_A_row_ind, const f_t *d_A_values, const i_t *d_basic_list,
    f_t *&d_Bt_dense
) {
    // Initialize d_Bt_dense to zero
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_dense, m * m * sizeof(f_t)), "cudaMalloc d_Bt_dense");
    CUDA_CALL_AND_CHECK(cudaMemset(d_Bt_dense, 0, m * m * sizeof(f_t)), "cudaMemset d_Bt_dense");

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    // Kernel to fill dense B_T
    fill_basis_transpose_dense_kernel<<<grid_size, block_size>>>(
        m, d_A_col_start, d_A_row_ind, d_A_values, d_basic_list, d_Bt_dense
    );
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_basis_transpose_dense_kernel");
}

/* Arguments:
 * [out] d_A_matrix: pointer to cudssMatrix_t where A will be stored on
 * device [in] A: host-side CSC matrix to be moved to device [in]
 * dss_handle: cuDSS handle
 */
template <typename i_t, typename f_t>
void move_A_to_device(const csc_matrix_t<i_t, f_t> &A, i_t *&d_A_col_ptr, i_t *&d_A_row_ind,
                      f_t *&d_A_values) {
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_A_col_ptr, (A.n + 1) * sizeof(i_t)),
                        "cudaMalloc d_A_row_ptr");
    i_t nnz = A.col_start[A.n];
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_A_row_ind, nnz * sizeof(i_t)), "cudaMalloc d_A_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_A_values, nnz * sizeof(f_t)), "cudaMalloc d_A_values");

    // Copy data to device
    // Column pointers
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_A_col_ptr, A.col_start.data(), (A.n + 1) * sizeof(i_t),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy d_A_col_ptr");
    // Row indices
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_A_row_ind, A.i.data(), nnz * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_A_row_ind");
    // Non-zero values
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_A_values, A.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_A_values");
}

template <typename i_t, typename f_t>
i_t compute_inverse(
    i_t m,
    i_t n,
    const csc_matrix_t<i_t, f_t>& A,
    const std::vector<i_t>& basic_list,
    f_t*& d_X
) {
    // Assemble B in CSR format
    csr_matrix_t<i_t, f_t> B_csr;
    B_csr.m = m;
    B_csr.n = m;
    B_csr.nz_max = 0;
    B_csr.row_start.resize(m + 1, 0);

    std::vector<i_t> elements_per_row(m);
    for (i_t col_idx = 0; col_idx < m; ++col_idx) {
        i_t j = basic_list[col_idx];
        for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
            i_t row = A.i[idx];
            elements_per_row[row]++;
        }
    }
    for (i_t row = 0; row < m; ++row) {
        B_csr.row_start[row + 1] = B_csr.row_start[row] + elements_per_row[row];
        B_csr.nz_max += elements_per_row[row];
    }
    B_csr.j.resize(B_csr.nz_max);
    B_csr.x.resize(B_csr.nz_max);
    std::vector<i_t> current_position_in_row(m);
    for (i_t col_idx = 0; col_idx < m; ++col_idx) {
        i_t j = basic_list[col_idx];
        for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
            i_t row = A.i[idx];
            i_t dest_pos = B_csr.row_start[row] + current_position_in_row[row];
            B_csr.j[dest_pos] = col_idx;
            B_csr.x[dest_pos] = A.x[idx];
            current_position_in_row[row]++;
        }
    }

    // Assemble B_T in CSR format
    // We construct B_T directly by considering the columns of A as rows of B_T
    csr_matrix_t<i_t, f_t> B_t_csr;
    B_t_csr.m = m;
    B_t_csr.n = m;
    B_t_csr.nz_max = 0;
    B_t_csr.row_start.resize(m + 1, 0);

    std::vector<i_t> elements_per_col(m);
    for (i_t col_idx = 0; col_idx < m; ++col_idx) {
        i_t j = basic_list[col_idx];
        elements_per_col[col_idx] = A.col_start[j + 1] - A.col_start[j];
    }
    for (i_t row = 0; row < m; ++row) {
        B_t_csr.row_start[row + 1] = B_t_csr.row_start[row] + elements_per_col[row];
        B_t_csr.nz_max += elements_per_col[row];
    }
    B_t_csr.j.resize(B_t_csr.nz_max);
    B_t_csr.x.resize(B_t_csr.nz_max);
    for (i_t col_idx = 0; col_idx < m; ++col_idx) {
        i_t j = basic_list[col_idx];
        for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
            i_t row = col_idx;
            i_t dest_pos = B_t_csr.row_start[row] + (idx - A.col_start[j]);
            B_t_csr.j[dest_pos] = A.i[idx];
            B_t_csr.x[dest_pos] = A.x[idx];
        }
    }

    // Move B to device
    i_t* d_B_row_ptr;
    i_t* d_B_col_ind;
    f_t* d_B_values;

    // // Debug comparison starts here
    // i_t *d_A_col_ptr = nullptr;
    // i_t *d_A_row_ind = nullptr;
    // f_t *d_A_values = nullptr;
    // phase2_cu::move_A_to_device<i_t, f_t>(A, d_A_col_ptr, d_A_row_ind, d_A_values);

    // i_t *d_basic_list;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
    //     "cudaMemcpy d_basic_list");

    // i_t nz_B;
    // phase2_cu::build_basis_on_device<i_t, f_t>(m, d_A_col_ptr, d_A_row_ind, d_A_values,
    //                                            d_basic_list, d_B_row_ptr, d_B_col_ind, d_B_values,
    //                                            &nz_B);

    // // Debug: Move B back to host and verify correctness
    // std::vector<i_t> B_row_ptr_host(m + 1);
    // std::vector<i_t> B_col_ind_host(nz_B);
    // std::vector<f_t> B_values_host(nz_B);
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(B_row_ptr_host.data(), d_B_row_ptr, (m + 1) * sizeof(i_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy B_row_ptr_host");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(B_col_ind_host.data(), d_B_col_ind, nz_B * sizeof(i_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy B_col_ind_host");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(B_values_host.data(), d_B_values, nz_B * sizeof(f_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy B_values_host");
    // // Create dense B
    // std::vector<std::vector<f_t>> B_dense_host(m, std::vector<f_t>(m, 0.0));
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t idx = B_row_ptr_host[row]; idx < B_row_ptr_host[row + 1]; ++idx) {
    //         i_t col = B_col_ind_host[idx];
    //         f_t val = B_values_host[idx];
    //         B_dense_host[row][col] = val;
    //     }
    // }
    // // Create cpu dense B from B_csr
    // std::vector<std::vector<f_t>> B_dense_cpu(m, std::vector<f_t>(m, 0.0));
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t idx = B_csr.row_start[row]; idx < B_csr.row_start[row + 1]; ++idx) {
    //         i_t col = B_csr.j[idx];
    //         f_t val = B_csr.x[idx];
    //         B_dense_cpu[row][col] = val;
    //     }
    // }
    // // Compare B_dense_host and B_dense_cpu
    // i_t mismatch_count = 0;
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t col = 0; col < m; ++col) {
    //         f_t val_host = B_dense_host[row][col];
    //         f_t val_cpu = B_dense_cpu[row][col];
    //         if (std::abs(val_host - val_cpu) > 1e-8) {
    //             printf("Mismatch in B at (%d, %d): host %f vs cpu %f\n", row, col, val_host, val_cpu);
    //             mismatch_count++;
    //         }
    //     }
    // }
    // // Debug comparison ends here

    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ind, B_csr.nz_max * sizeof(i_t)), "cudaMalloc d_B_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_values, B_csr.nz_max * sizeof(f_t)), "cudaMalloc d_B_values");

    CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_row_ptr, B_csr.row_start.data(), (m + 1) * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy d_B_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_col_ind, B_csr.j.data(), B_csr.nz_max * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy d_B_col_ind");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_values, B_csr.x.data(), B_csr.nz_max * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy d_B_values");

    // Move B_T to device
    i_t* d_Bt_row_ptr;
    i_t* d_Bt_col_ind;
    f_t* d_Bt_values;

    // // Debug comparison starts here
    // i_t *d_A_col_ptr = nullptr;
    // i_t *d_A_row_ind = nullptr;
    // f_t *d_A_values = nullptr;
    // phase2_cu::move_A_to_device<i_t, f_t>(A, d_A_col_ptr, d_A_row_ind, d_A_values);

    // i_t *d_basic_list;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
    //     "cudaMemcpy d_basic_list");
    // phase2_cu::build_basis_transpose_on_device<i_t, f_t>(
    //     m, d_A_col_ptr, d_A_row_ind, d_A_values,
    //     d_basic_list, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values
    // );

    // // Debug: Move B_T back to host and verify correctness
    // std::vector<i_t> Bt_row_ptr_host(m + 1);
    // std::vector<i_t> Bt_col_ind_host(B_t_csr.nz_max);
    // std::vector<f_t> Bt_values_host(B_t_csr.nz_max);
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(Bt_row_ptr_host.data(), d_Bt_row_ptr, (m + 1) * sizeof(i_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy Bt_row_ptr_host");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(Bt_col_ind_host.data(), d_Bt_col_ind, B_t_csr.nz_max * sizeof(i_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy Bt_col_ind_host");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(Bt_values_host.data(), d_Bt_values, B_t_csr.nz_max * sizeof(f_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy Bt_values_host");
    // // Create dense B_T
    // std::vector<std::vector<f_t>> Bt_dense_host_(m, std::vector<f_t>(m, 0.0));
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t idx = Bt_row_ptr_host[row]; idx < Bt_row_ptr_host[row + 1]; ++idx) {
    //         i_t col = Bt_col_ind_host[idx];
    //         f_t val = Bt_values_host[idx];
    //         Bt_dense_host_[row][col] = val;
    //     }
    // }
    // // Create cpu dense B_T from B_t_csr
    // std::vector<std::vector<f_t>> Bt_dense_cpu(m, std::vector<f_t>(m, 0.0));
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t idx = B_t_csr.row_start[row]; idx < B_t_csr.row_start[row + 1]; ++idx) {
    //         i_t col = B_t_csr.j[idx];
    //         f_t val = B_t_csr.x[idx];
    //         Bt_dense_cpu[row][col] = val;
    //     }
    // }
    // // Compare Bt_dense_host and Bt_dense_cpu
    // i_t mismatch_count = 0;
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t col = 0; col < m; ++col) {
    //         f_t val_host = Bt_dense_host_[row][col];
    //         f_t val_cpu = Bt_dense_cpu[row][col];
    //         if (std::abs(val_host - val_cpu) > 1e-8) {
    //             printf("Mismatch in B_T at (%d, %d): host %f vs cpu %f\n", row, col, val_host, val_cpu);
    //             mismatch_count++;
    //         }
    //     }
    // }
    // // Debug comparison ends here

    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_Bt_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_col_ind, B_t_csr.nz_max * sizeof(i_t)), "cudaMalloc d_Bt_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_values, B_t_csr.nz_max * sizeof(f_t)), "cudaMalloc d_Bt_values");

    CUDA_CALL_AND_CHECK(cudaMemcpy(d_Bt_row_ptr, B_t_csr.row_start.data(), (m + 1) * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy d_Bt_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_Bt_col_ind, B_t_csr.j.data(), B_t_csr.nz_max * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy d_Bt_col_ind");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_Bt_values, B_t_csr.x.data(), B_t_csr.nz_max * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy d_Bt_values");

    // Compute (B^T B) with cuSPARSE
    cusparseHandle_t cusparse_handle;
    // Here we assume (B^T B) is also sparse
    i_t* d_BtB_row_ptr = nullptr;
    i_t* d_BtB_col_ind = nullptr;
    f_t* d_BtB_values = nullptr;
    cusparseSpMatDescr_t d_B_matrix, d_Bt_matrix, d_BtB_matrix;
    CUSPARSE_CALL_AND_CHECK(cusparseCreate(&cusparse_handle), "cusparseCreate");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(
        &d_B_matrix,
        m,
        m,
        B_csr.nz_max,
        d_B_row_ptr,
        d_B_col_ind,
        d_B_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ), "cusparseCreateCsr for B");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(
        &d_Bt_matrix,
        m,
        m,
        B_t_csr.nz_max,
        d_Bt_row_ptr,
        d_Bt_col_ind,
        d_Bt_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ), "cusparseCreateCsr for B_T");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(
        &d_BtB_matrix,
        m,
        m,
        0,  // to be computed
        nullptr,
        nullptr,
        nullptr,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ), "cusparseCreateCsr for B_T B");

    void* d_buffer1 = nullptr;
    void* d_buffer2 = nullptr;
    size_t buffer_size1 = 0;
    size_t buffer_size2 = 0;
    f_t alpha = 1.0;
    f_t beta = 0.0;
    cusparseSpGEMMDescr_t spgemm_desc;
    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_createDescr(&spgemm_desc), "cusparseSpGEMM_createDescr");

    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_workEstimation(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        d_Bt_matrix,
        d_B_matrix,
        &beta,
        d_BtB_matrix,
        CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_desc,
        &buffer_size1,
        nullptr
    ), "cusparseSpGEMM_workEstimation for B_T B");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_buffer1, buffer_size1), "cudaMalloc d_buffer1 for B_T B");

    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_workEstimation(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        d_Bt_matrix,
        d_B_matrix,
        &beta,
        d_BtB_matrix,
        CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_desc,
        &buffer_size1,
        d_buffer1
    ), "cusparseSpGEMM_workEstimation for B_T B");

    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_compute(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        d_Bt_matrix,
        d_B_matrix,
        &beta,
        d_BtB_matrix,
        CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_desc,
        &buffer_size2,
        nullptr
    ), "cusparseSpGEMM_compute for B_T B");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_buffer2, buffer_size2), "cudaMalloc d_buffer2 for B_T B");

    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_compute(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        d_Bt_matrix,
        d_B_matrix,
        &beta,
        d_BtB_matrix,
        CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_desc,
        &buffer_size2,
        d_buffer2
    ), "cusparseSpGEMM_compute for B_T B");
    
    // Get the size of the result matrix AFTER compute
    int64_t rows_BtB, cols_BtB, nnz_BtB;
    CUSPARSE_CALL_AND_CHECK(cusparseSpMatGetSize(d_BtB_matrix, &rows_BtB, &cols_BtB, &nnz_BtB), "cusparseSpMatGetSize for B_T B");
    assert(rows_BtB == m);
    assert(cols_BtB == m);
    
    // Allocate memory for the result matrix
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_BtB_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_col_ind, nnz_BtB * sizeof(i_t)), "cudaMalloc d_BtB_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_values, nnz_BtB * sizeof(f_t)), "cudaMalloc d_BtB_values");
    
    // Update the matrix descriptor with the allocated pointers BEFORE copy
    CUSPARSE_CALL_AND_CHECK(cusparseCsrSetPointers(d_BtB_matrix, d_BtB_row_ptr, d_BtB_col_ind, d_BtB_values), "cusparseCsrSetPointers for B_T B");
    
    // Now copy the result into the allocated memory
    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_copy(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        d_Bt_matrix,
        d_B_matrix,
        &beta,
        d_BtB_matrix,
        CUDA_R_64F,
        CUSPARSE_SPGEMM_DEFAULT,
        spgemm_desc
    ), "cusparseSpGEMM_copy for B_T B");

    // cuSPARSE cleanup
    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_destroyDescr(spgemm_desc), "cusparseSpGEMM_destroyDescr");
    CUDA_CALL_AND_CHECK(cudaFree(d_buffer1), "cudaFree d_buffer1");
    CUDA_CALL_AND_CHECK(cudaFree(d_buffer2), "cudaFree d_buffer2");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(d_B_matrix), "cusparseDestroySpMat B");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(d_Bt_matrix), "cusparseDestroySpMat B_T");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(d_BtB_matrix), "cusparseDestroySpMat B_T B");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroy(cusparse_handle), "cusparseDestroy");

    // Factor (B^T B) = L*U using cuDSS and solve for (B^T B) X = B^T
    // => X = (B^T B)^(-1) B^T is the pseudo-inverse of B
    cudssStatus_t dss_status = CUDSS_STATUS_SUCCESS;

    cudssHandle_t dss_handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&dss_handle), dss_status, "cudssCreateHandle B");

    cudssConfig_t solverConfig;
    cudssData_t solverData;
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), dss_status, "cudssCreateConfig B");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(dss_handle, &solverData), dss_status, "cudssCreateData B");

    cudssMatrix_t d_BtB_matrix_cudss;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(
        &d_BtB_matrix_cudss,
        m,
        m,
        nnz_BtB,
        d_BtB_row_ptr,
        NULL,
        d_BtB_col_ind,
        d_BtB_values,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_SPD,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ), dss_status, "cudssMatrixCreateCsr for B");

    // Convert B^T from sparse CSR to dense format for RHS (column-major)
    f_t* d_Bt_dense;

    // // Debug starts here
    // i_t *d_A_col_ptr = nullptr;
    // i_t *d_A_row_ind = nullptr;
    // f_t *d_A_values = nullptr;
    // phase2_cu::move_A_to_device<i_t, f_t>(A, d_A_col_ptr, d_A_row_ind, d_A_values);

    // i_t *d_basic_list;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
    //     "cudaMemcpy d_basic_list");

    // phase2_cu::build_basis_transpose_on_device_dense(m, d_A_col_ptr, d_A_row_ind, d_A_values, d_basic_list, d_Bt_dense);

    // // Debug: Move B_T back to host and verify correctness
    // std::vector<f_t> Bt_dense_host_debug(m * m);
    // CUDA_CALL_AND_CHECK(
    //     cudaMemcpy(Bt_dense_host_debug.data(), d_Bt_dense, m * m * sizeof(f_t), cudaMemcpyDeviceToHost),
    //     "cudaMemcpy Bt_dense_host_debug");

    // Convert CSR to dense in column-major format
    std::vector<f_t> Bt_dense_host(m * m, 0.0);
    for (i_t row = 0; row < m; ++row) {
        for (i_t idx = B_t_csr.row_start[row]; idx < B_t_csr.row_start[row + 1]; ++idx) {
            i_t col = B_t_csr.j[idx];
            // Column-major: element (row, col) is at index col * m + row
            Bt_dense_host[col * m + row] = B_t_csr.x[idx];
        }
    }

    // // Debug comparison
    // i_t mismatch_count = 0;
    // for (i_t row = 0; row < m; ++row) {
    //     for (i_t col = 0; col < m; ++col) {
    //         f_t val_host = Bt_dense_host[col * m + row];
    //         f_t val_debug = Bt_dense_host_debug[col * m + row];
    //         if (std::abs(val_host - val_debug) > 1e-8) {
    //             printf("Mismatch in dense B_T at (%d, %d): host %f vs debug %f\n", row, col, val_host, val_debug);
    //             mismatch_count++;
    //         }
    //     }
    // }
    // // Debug ends here

    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_dense, m * m * sizeof(f_t)), "cudaMalloc d_Bt_dense");
    CUDA_CALL_AND_CHECK(cudaMemset(d_Bt_dense, 0, m * m * sizeof(f_t)), "cudaMemset d_Bt_dense");

    CUDA_CALL_AND_CHECK(cudaMemcpy(d_Bt_dense, Bt_dense_host.data(), m * m * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy d_Bt_dense");
    
    cudssMatrix_t d_Bt_matrix_cudss;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(
        &d_Bt_matrix_cudss,
        m,
        m,
        m,
        d_Bt_dense,
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    ), dss_status, "cudssMatrixCreateDn for B_T");

    cudssMatrix_t d_X_matrix_cudss; // This is the pseudo-inverse of B
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(
        &d_X_matrix_cudss,
        m,
        m,
        m,
        d_X,
        CUDA_R_64F,
        CUDSS_LAYOUT_COL_MAJOR
    ), dss_status, "cudssMatrixCreateDn for X");

    CUDSS_CALL_AND_CHECK(cudssExecute(
        dss_handle,
        CUDSS_PHASE_ANALYSIS,
        solverConfig,
        solverData,
        d_BtB_matrix_cudss,
        d_X_matrix_cudss,
        d_Bt_matrix_cudss
    ), dss_status, "cudssExecute Analysis for B");
    
    CUDSS_CALL_AND_CHECK(cudssExecute(
        dss_handle,
        CUDSS_PHASE_FACTORIZATION,
        solverConfig,
        solverData,
        d_BtB_matrix_cudss,
        d_X_matrix_cudss,
        d_Bt_matrix_cudss
    ), dss_status, "cudssExecute Factorization for B");

    CUDSS_CALL_AND_CHECK(cudssExecute(
        dss_handle,
        CUDSS_PHASE_SOLVE,
        solverConfig,
        solverData,
        d_BtB_matrix_cudss,
        d_X_matrix_cudss,
        d_Bt_matrix_cudss
    ), dss_status, "cudssExecute Solve for B");

    // cuDSS cleanup
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_BtB_matrix_cudss), dss_status, "cudssMatrixDestroy B_T B");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_Bt_matrix_cudss), dss_status, "cudssMatrixDestroy B_T");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_X_matrix_cudss), dss_status, "cudssMatrixDestroy X");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(dss_handle, solverData), dss_status, "cudssDataDestroy B");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), dss_status, "cudssConfigDestroy B");
    CUDSS_CALL_AND_CHECK(cudssDestroy(dss_handle), dss_status, "cudssDestroyHandle B");

    CUDA_CALL_AND_CHECK(cudaFree(d_Bt_dense), "cudaFree d_Bt_dense");
    // Note: d_X is not freed here - caller will free it after use

    return 0;
}

} // namespace phase2_cu

template <typename i_t, typename f_t>
dual::status_t dual_phase2_cu(i_t phase, i_t slack_basis, f_t start_time,
                              const lp_problem_t<i_t, f_t> &lp,
                              const simplex_solver_settings_t<i_t, f_t> &settings,
                              std::vector<variable_status_t> &vstatus, lp_solution_t<i_t, f_t> &sol,
                              i_t &iter, std::vector<f_t> &delta_y_steepest_edge) {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;
    assert(m <= n);
    assert(vstatus.size() == n);
    assert(lp.A.m == m);
    assert(lp.A.n == n);
    assert(lp.objective.size() == n);
    assert(lp.lower.size() == n);
    assert(lp.upper.size() == n);
    assert(lp.rhs.size() == m);
    std::vector<i_t> basic_list(m);
    std::vector<i_t> nonbasic_list;
    std::vector<i_t> superbasic_list;

    std::vector<f_t> &x = sol.x;
    std::vector<f_t> &y = sol.y;
    std::vector<f_t> &z = sol.z;

    dual::status_t status = dual::status_t::UNSET;

    // Perturbed objective
    std::vector<f_t> objective = lp.objective;

    settings.log.printf("Dual Simplex Phase %d\n", phase);
    std::vector<variable_status_t> vstatus_old = vstatus;
    std::vector<f_t> z_old = z;

    phase2::bound_info(lp, settings);
    get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
    assert(superbasic_list.size() == 0);
    assert(nonbasic_list.size() == n - m);

    // Compute Moore-Penrose pseudo-inverse of B
    f_t* d_B_pinv;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv, m * m * sizeof(f_t)), "cudaMalloc d_D_pinv");
    phase2_cu::compute_inverse<i_t, f_t>(
        m,
        n,
        lp.A,
        basic_list,
        d_B_pinv
    );

    if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
    std::vector<f_t> c_basic(m);
    for (i_t k = 0; k < m; ++k) {
        const i_t j = basic_list[k];
        c_basic[k]  = objective[j];
    }

    // Solve B'*y = cB
    cublasHandle_t cublas_handle;
    CUBLAS_CALL_AND_CHECK(cublasCreate(&cublas_handle), "cublasCreate");
    phase2::pinv_solve(
        cublas_handle,
        d_B_pinv,
        c_basic,
        y,
        m,
        true
    );

    if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

    phase2::compute_reduced_costs(objective, lp.A, y, basic_list, nonbasic_list, z);

    phase2::set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

    const f_t init_dual_inf =
        phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
    if (init_dual_inf > settings.dual_tol) {
        settings.log.printf("Initial dual infeasibility %e\n", init_dual_inf);
    }

    for (i_t j = 0; j < n; ++j) {
        if (lp.lower[j] == -inf && lp.upper[j] == inf && vstatus[j] != variable_status_t::BASIC) {
        settings.log.printf("Free variable %d vstatus %d\n", j, vstatus[j]);
        }
    }

    phase2::compute_primal_variables(
        cublas_handle, d_B_pinv, lp.rhs, lp.A, basic_list, nonbasic_list, settings.tight_tol, x);

    if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

    if (delta_y_steepest_edge.size() == 0) {
        delta_y_steepest_edge.resize(n);
        if (slack_basis) {
        phase2::initialize_steepest_edge_norms_from_slack_basis(
            basic_list, nonbasic_list, delta_y_steepest_edge);
        } else {
            std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
            if (phase2::initialize_steepest_edge_norms(
                    lp, settings, start_time, basic_list, cublas_handle, d_B_pinv, delta_y_steepest_edge) == -1) {
                return dual::status_t::TIME_LIMIT;
            }
        }
    } else {
        settings.log.printf("using exisiting steepest edge %e\n",
                            vector_norm2<i_t, f_t>(delta_y_steepest_edge));
    }

    if (phase == 2) {
        settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
    }

    const i_t iter_limit = settings.iteration_limit;
    std::vector<f_t> delta_y(m, 0.0);
    std::vector<f_t> delta_z(n, 0.0);
    std::vector<f_t> delta_x(n, 0.0);
    std::vector<f_t> delta_x_flip(n, 0.0);
    std::vector<f_t> atilde(m, 0.0);
    std::vector<i_t> atilde_mark(m, 0);
    std::vector<i_t> atilde_index;
    std::vector<i_t> nonbasic_mark(n);
    std::vector<i_t> basic_mark(n);
    std::vector<i_t> delta_z_mark(n, 0);
    std::vector<i_t> delta_z_indices;
    std::vector<f_t> v(m, 0.0);
    std::vector<f_t> squared_infeasibilities;
    std::vector<i_t> infeasibility_indices;

    delta_z_indices.reserve(n);

    phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);

    std::vector<uint8_t> bounded_variables(n, 0);
    phase2::compute_bounded_info(lp.lower, lp.upper, bounded_variables);

    f_t primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
        lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

    csc_matrix_t<i_t, f_t> A_transpose(1, 1, 0);
    lp.A.transpose(A_transpose);

    f_t obj              = compute_objective(lp, x);
    const i_t start_iter = iter;

    i_t sparse_delta_z = 0;
    i_t dense_delta_z  = 0;
    phase2::phase2_timers_t<i_t, f_t> timers(settings.profile && phase == 2);

    while (iter < iter_limit) {
        // Pricing
        i_t direction           = 0;
        i_t basic_leaving_index = -1;
        i_t leaving_index       = -1;
        f_t max_val;
        timers.start_timer();
        if (settings.use_steepest_edge_pricing) {
        leaving_index = phase2::steepest_edge_pricing_with_infeasibilities(lp,
                                                                            settings,
                                                                            x,
                                                                            delta_y_steepest_edge,
                                                                            basic_mark,
                                                                            squared_infeasibilities,
                                                                            infeasibility_indices,
                                                                            direction,
                                                                            basic_leaving_index,
                                                                            max_val);
        } else {
        // Max infeasibility pricing
        leaving_index = phase2::phase2_pricing(
            lp, settings, x, basic_list, direction, basic_leaving_index, primal_infeasibility);
        }
        timers.pricing_time += timers.stop_timer();
        if (leaving_index == -1) {
        phase2::prepare_optimality(lp,
                                    settings,
                                    cublas_handle,
                                    d_B_pinv,
                                    objective,
                                    basic_list,
                                    nonbasic_list,
                                    vstatus,
                                    phase,
                                    start_time,
                                    max_val,
                                    iter,
                                    x,
                                    y,
                                    z,
                                    sol);
        status = dual::status_t::OPTIMAL;
        break;
        }

        // BTran
        // BT*delta_y = -delta_zB = -sigma*ei
        timers.start_timer();
        sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
        phase2::compute_delta_y(cublas_handle, d_B_pinv, basic_leaving_index, direction, delta_y_sparse);
        timers.btran_time += timers.stop_timer();

        const f_t steepest_edge_norm_check = delta_y_sparse.norm2_squared();
        if (delta_y_steepest_edge[leaving_index] <
            settings.steepest_edge_ratio * steepest_edge_norm_check) {
        constexpr bool verbose = false;
        if constexpr (verbose) {
            settings.log.printf(
            "iteration restart due to steepest edge. Leaving %d. Actual %.2e "
            "from update %.2e\n",
            leaving_index,
            steepest_edge_norm_check,
            delta_y_steepest_edge[leaving_index]);
        }
        delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
        continue;
        }

        timers.start_timer();
        i_t delta_y_nz0      = 0;
        const i_t nz_delta_y = delta_y_sparse.i.size();
        for (i_t k = 0; k < nz_delta_y; k++) {
        if (std::abs(delta_y_sparse.x[k]) > 1e-12) { delta_y_nz0++; }
        }
        const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
        const bool use_transpose        = delta_y_nz_percentage <= 30.0;
        if (use_transpose) {
        sparse_delta_z++;
        phase2::compute_delta_z(A_transpose,
                                delta_y_sparse,
                                leaving_index,
                                direction,
                                nonbasic_mark,
                                delta_z_mark,
                                delta_z_indices,
                                delta_z);
        } else {
        dense_delta_z++;
        // delta_zB = sigma*ei
        delta_y_sparse.to_dense(delta_y);
        phase2::compute_reduced_cost_update(lp,
                                            basic_list,
                                            nonbasic_list,
                                            delta_y,
                                            leaving_index,
                                            direction,
                                            delta_z_mark,
                                            delta_z_indices,
                                            delta_z);
        }
        timers.delta_z_time += timers.stop_timer();

        // Ratio test
        f_t step_length;
        i_t entering_index          = -1;
        i_t nonbasic_entering_index = -1;
        const bool harris_ratio     = settings.use_harris_ratio;
        const bool bound_flip_ratio = settings.use_bound_flip_ratio;
        if (harris_ratio) {
            f_t max_step_length = phase2::first_stage_harris(lp, vstatus, nonbasic_list, z, delta_z);
            entering_index      = phase2::second_stage_harris(lp,
                                                        vstatus,
                                                        nonbasic_list,
                                                        z,
                                                        delta_z,
                                                        max_step_length,
                                                        step_length,
                                                        nonbasic_entering_index);
        } else if (bound_flip_ratio) {
            timers.start_timer();
            f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                                        : (x[leaving_index] - lp.upper[leaving_index]);
            bound_flipping_ratio_test_t<i_t, f_t> bfrt(settings,
                                                        start_time,
                                                        m,
                                                        n,
                                                        slope,
                                                        lp.lower,
                                                        lp.upper,
                                                        bounded_variables,
                                                        vstatus,
                                                        nonbasic_list,
                                                        z,
                                                        delta_z,
                                                        delta_z_indices,
                                                        nonbasic_mark);
            entering_index = bfrt.compute_step_length(step_length, nonbasic_entering_index);
            timers.bfrt_time += timers.stop_timer();
        } else {
            entering_index = phase2::phase2_ratio_test(
                lp, settings, vstatus, nonbasic_list, z, delta_z, step_length, nonbasic_entering_index);
        }
        if (entering_index == -2) { return dual::status_t::TIME_LIMIT; }
        if (entering_index == -3) { return dual::status_t::CONCURRENT_LIMIT; }
        if (entering_index == -1) {
            settings.log.printf("No entering variable found. Iter %d\n", iter);
            settings.log.printf("Scaled infeasibility %e\n", max_val);
            f_t perturbation = phase2::amount_of_perturbation(lp, objective);

            if (perturbation > 0.0 && phase == 2) {
                // Try to remove perturbation
                std::vector<f_t> unperturbed_y(m);
                std::vector<f_t> unperturbed_z(n);
                phase2::compute_dual_solution_from_basis(
                lp, cublas_handle, d_B_pinv, basic_list, nonbasic_list, unperturbed_y, unperturbed_z);
                {
                    const f_t dual_infeas = phase2::dual_infeasibility(
                        lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
                    settings.log.printf("Dual infeasibility after removing perturbation %e\n", dual_infeas);
                    if (dual_infeas <= settings.dual_tol) {
                        settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
                        z            = unperturbed_z;
                        y            = unperturbed_y;
                        perturbation = 0.0;

                        std::vector<f_t> unperturbed_x(n);
                        phase2::compute_primal_solution_from_basis(
                        lp, cublas_handle, d_B_pinv, basic_list, nonbasic_list, vstatus, unperturbed_x);
                        x                    = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                        lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
                        settings.log.printf("Updated primal infeasibility: %e\n", primal_infeasibility);

                        objective = lp.objective;
                        // Need to reset the objective value, since we have recomputed x
                        obj = phase2::compute_perturbed_objective(objective, x);
                        if (dual_infeas <= settings.dual_tol && primal_infeasibility <= settings.primal_tol) {
                            phase2::prepare_optimality(lp,
                                                        settings,
                                                        cublas_handle,
                                                        d_B_pinv,
                                                        objective,
                                                        basic_list,
                                                        nonbasic_list,
                                                        vstatus,
                                                        phase,
                                                        start_time,
                                                        max_val,
                                                        iter,
                                                        x,
                                                        y,
                                                        z,
                                                        sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf(
                            "Continuing with perturbation removed and steepest edge norms reset\n");
                        // Clear delta_z before restarting the iteration
                        phase2::clear_delta_z(
                            entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);
                        continue;
                    } else {
                        std::vector<f_t> unperturbed_x(n);
                        phase2::compute_primal_solution_from_basis(
                            lp, cublas_handle, d_B_pinv, basic_list, nonbasic_list, vstatus, unperturbed_x);
                        x                    = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                        lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

                        const f_t orig_dual_infeas = phase2::dual_infeasibility(
                            lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);

                        if (primal_infeasibility <= settings.primal_tol &&
                            orig_dual_infeas <= settings.dual_tol) {
                            phase2::prepare_optimality(lp,
                                                        settings,
                                                        cublas_handle,
                                                        d_B_pinv,
                                                        objective,
                                                        basic_list,
                                                        nonbasic_list,
                                                        vstatus,
                                                        phase,
                                                        start_time,
                                                        max_val,
                                                        iter,
                                                        x,
                                                        y,
                                                        z,
                                                        sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf("Failed to remove perturbation of %.2e.\n", perturbation);
                    }
                }
            }

            const f_t dual_infeas =
                phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
            settings.log.printf("Dual infeasibility %e\n", dual_infeas);
            const f_t primal_inf = phase2::primal_infeasibility(lp, settings, vstatus, x);
            settings.log.printf("Primal infeasibility %e\n", primal_inf);
            settings.log.printf("Steepest edge %e\n", max_val);
            if (dual_infeas > settings.dual_tol) {
                settings.log.printf(
                    "Numerical issues encountered. No entering variable found with large infeasibility.\n");
                return dual::status_t::NUMERICAL;
            }
            return dual::status_t::DUAL_UNBOUNDED;
        }

        timers.start_timer();
        // Update dual variables
        // y <- y + steplength * delta_y
        // z <- z + steplength * delta_z
        phase2::update_dual_variables(
            delta_y_sparse, delta_z_indices, delta_z, step_length, leaving_index, y, z);
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // Update primal variable
        const i_t num_flipped = phase2::flip_bounds(lp,
                                                    settings,
                                                    bounded_variables,
                                                    objective,
                                                    z,
                                                    delta_z_indices,
                                                    nonbasic_list,
                                                    entering_index,
                                                    vstatus,
                                                    delta_x_flip,
                                                    atilde_mark,
                                                    atilde,
                                                    atilde_index);

        timers.flip_time += timers.stop_timer();

        sparse_vector_t<i_t, f_t> delta_xB_0_sparse(m, 0);
        if (num_flipped > 0) {
            timers.start_timer();
            phase2::adjust_for_flips(cublas_handle,
                                    d_B_pinv,
                                    basic_list,
                                    delta_z_indices,
                                    atilde_index,
                                    atilde,
                                    atilde_mark,
                                    delta_xB_0_sparse,
                                    delta_x_flip,
                                    x);
            timers.ftran_time += timers.stop_timer();
        }

        timers.start_timer();
        sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
        sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
        if (phase2::compute_delta_x(lp,
                                    cublas_handle,
                                    d_B_pinv,
                                    entering_index,
                                    leaving_index,
                                    basic_leaving_index,
                                    direction,
                                    basic_list,
                                    delta_x_flip,
                                    rhs_sparse,
                                    x,
                                    scaled_delta_xB_sparse,
                                    delta_x) == -1) {
            settings.log.printf("Failed to compute delta_x. Iter %d\n", iter);
            return dual::status_t::NUMERICAL;
        }

        timers.ftran_time += timers.stop_timer();

        timers.start_timer();
        const i_t steepest_edge_status = phase2::update_steepest_edge_norms(settings,
                                                                            basic_list,
                                                                            cublas_handle,
                                                                            d_B_pinv,
                                                                            direction,
                                                                            delta_y_sparse,
                                                                            steepest_edge_norm_check,
                                                                            scaled_delta_xB_sparse,
                                                                            basic_leaving_index,
                                                                            entering_index,
                                                                            v,
                                                                            delta_y_steepest_edge);
        assert(steepest_edge_status == 0);
        timers.se_norms_time += timers.stop_timer();

        timers.start_timer();
        // x <- x + delta_x
        phase2::update_primal_variables(scaled_delta_xB_sparse, basic_list, delta_x, entering_index, x);
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // TODO(CMM): Do I also need to update the objective due to the bound flips?
        // TODO(CMM): I'm using the unperturbed objective here, should this be the perturbed objective?
        phase2::update_objective(
        basic_list, scaled_delta_xB_sparse.i, lp.objective, delta_x, entering_index, obj);
        timers.objective_time += timers.stop_timer();

        timers.start_timer();
        // Update primal infeasibilities due to changes in basic variables
        // from flipping bounds
        phase2::update_primal_infeasibilities(lp,
                                            settings,
                                            basic_list,
                                            x,
                                            entering_index,
                                            leaving_index,
                                            delta_xB_0_sparse.i,
                                            squared_infeasibilities,
                                            infeasibility_indices,
                                            primal_infeasibility);
        // Update primal infeasibilities due to changes in basic variables
        // from the leaving and entering variables
        phase2::update_primal_infeasibilities(lp,
                                            settings,
                                            basic_list,
                                            x,
                                            entering_index,
                                            leaving_index,
                                            scaled_delta_xB_sparse.i,
                                            squared_infeasibilities,
                                            infeasibility_indices,
                                            primal_infeasibility);
        // Update the entering variable
        phase2::update_single_primal_infeasibility(lp.lower,
                                                lp.upper,
                                                x,
                                                settings.primal_tol,
                                                squared_infeasibilities,
                                                infeasibility_indices,
                                                entering_index,
                                                primal_infeasibility);

        phase2::clean_up_infeasibilities(squared_infeasibilities, infeasibility_indices);

        timers.update_infeasibility_time += timers.stop_timer();

        // Clear delta_x
        phase2::clear_delta_x(basic_list, entering_index, scaled_delta_xB_sparse, delta_x);

        timers.start_timer();
        f_t sum_perturb = 0.0;
        phase2::compute_perturbation(lp, settings, delta_z_indices, z, objective, sum_perturb);
        timers.perturb_time += timers.stop_timer();

        // Update basis information
        vstatus[entering_index] = variable_status_t::BASIC;
        if (lp.lower[leaving_index] != lp.upper[leaving_index]) {
            vstatus[leaving_index] = static_cast<variable_status_t>(-direction);
        } else {
            vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
        }
        basic_list[basic_leaving_index]        = entering_index;
        nonbasic_list[nonbasic_entering_index] = leaving_index;
        nonbasic_mark[entering_index]          = -1;
        nonbasic_mark[leaving_index]           = nonbasic_entering_index;
        basic_mark[leaving_index]              = -1;
        basic_mark[entering_index]             = basic_leaving_index;

        timers.start_timer();
        // Refactor or update the basis factorization
        bool should_refactor = true;
        if (!should_refactor) {
            // i_t recommend_refactor = ft.update(utilde_sparse, UTsol_sparse, basic_leaving_index);
            // should_refactor = recommend_refactor == 1;
            // TODO: eta updates
        }

        if (should_refactor) {
            // Recompute d_B_pinv
            phase2_cu::compute_inverse<i_t, f_t>(
                m,
                n,
                lp.A,
                basic_list,
                d_B_pinv
            );

            phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);
            phase2::compute_initial_primal_infeasibilities(
                lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);
        }
        timers.lu_update_time += timers.stop_timer();

        timers.start_timer();
        phase2::compute_steepest_edge_norm_entering(
            settings, m, cublas_handle, d_B_pinv, basic_leaving_index, entering_index, delta_y_steepest_edge);
        timers.se_entering_time += timers.stop_timer();

        iter++;

        // Clear delta_z
        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices, delta_z);

        f_t now = toc(start_time);
        if ((iter - start_iter) < settings.first_iteration_log ||
            (iter % settings.iteration_log_frequency) == 0) {
        if (phase == 1 && iter == 1) {
            settings.log.printf(" Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
        }
        settings.log.printf("%5d %+.16e %7d %.8e %.2e %.2f\n",
                            iter,
                            compute_user_objective(lp, obj),
                            infeasibility_indices.size(),
                            primal_infeasibility,
                            sum_perturb,
                            now);
        }

        if (obj >= settings.cut_off) {
            settings.log.printf("Solve cutoff. Current objective %e. Cutoff %e\n", obj, settings.cut_off);
            return dual::status_t::CUTOFF;
        }

        if (now > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

        if (settings.concurrent_halt != nullptr &&
            settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
            return dual::status_t::CONCURRENT_LIMIT;
        }
    }
    if (iter >= iter_limit) { status = dual::status_t::ITERATION_LIMIT; }

    if (phase == 2) {
        timers.print_timers(settings);
    }

    // Cleanup GPU resources
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv), "cudaFree d_D_pinv");
    CUBLAS_CALL_AND_CHECK(cublasDestroy(cublas_handle), "cublasDestroy");

    return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template dual::status_t dual_phase2_cu<int, double>(
    int phase, int slack_basis, double start_time, const lp_problem_t<int, double> &lp,
    const simplex_solver_settings_t<int, double> &settings, std::vector<variable_status_t> &vstatus,
    lp_solution_t<int, double> &sol, int &iter, std::vector<double> &steepest_edge_norms);

#endif

} // namespace cuopt::linear_programming::dual_simplex
