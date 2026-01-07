#include "cudss.h"
#include "cusparse.h"
#include "dual_simplex/sparse_vector.hpp"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <driver_types.h>
#include <dual_simplex/phase2_dual.cuh>
#include <dual_simplex/problem_analysis.hpp>

#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace phase2_cu {

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

    row_counts[idx] = end - start;
}

template <typename i_t, typename f_t>
__global__ void fill_basis_kernel(i_t m, const i_t *__restrict__ A_col_start,
                                  const i_t *__restrict__ A_row_ind,
                                  const f_t *__restrict__ A_values,
                                  const i_t *__restrict__ basic_list, i_t *__restrict__ B_col_ptr,
                                  i_t *__restrict__ B_row_ind, f_t *__restrict__ B_values) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in A
    i_t start = A_col_start[col_idx];
    i_t end = A_col_start[col_idx + 1];

    for (i_t k = start; k < end; ++k) {
        i_t row = A_row_ind[k];
        f_t val = A_values[k];
        i_t pos = B_col_ptr[idx] + (k - start);
        B_row_ind[pos] = row;
        B_values[pos] = val;
    }
}

// Helper function to orchestrate the basis
// construction on device
template <typename i_t, typename f_t>
void build_basis_on_device(i_t m, const i_t *d_A_col_start, const i_t *d_A_row_ind,
                           const f_t *d_A_values, const i_t *d_basic_list, i_t *&d_B_col_ptr,
                           i_t *&d_B_row_ind, f_t *&d_B_values, i_t &nz_B) {
    // 1. Allocate and compute column counts for B_T
    CUDA_CALL_AND_CHECK(cudaMemset(d_B_col_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_B_col_ptr");
    assert(d_B_col_ptr != nullptr);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    count_basis_rows_kernel<<<grid_size, block_size>>>(m, d_A_col_start, d_A_row_ind, d_basic_list,
                                                       d_B_col_ptr);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_basis_rows_kernel");

    // 2. Prefix sum to get row pointers for B_T
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_B_col_ptr);
    thrust::exclusive_scan(thrust::cuda::par, dev_ptr, dev_ptr + m + 1, dev_ptr, i_t(0));

    // Get total NNZ for B
    i_t total_nz;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&total_nz, d_B_col_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy total_nz for B");
    nz_B = total_nz;

    // 3. Allocate B column indices and values
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ind, total_nz * sizeof(i_t)), "cudaMalloc d_B_row_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_values, total_nz * sizeof(f_t)), "cudaMalloc d_B_values");

    // 4. Fill CSR structure for B
    fill_basis_kernel<<<grid_size, block_size>>>(m, d_A_col_start, d_A_row_ind, d_A_values,
                                                 d_basic_list, d_B_col_ptr, d_B_row_ind,
                                                 d_B_values);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_basis_kernel");
}

template <typename i_t, typename f_t>
__global__ void BtB_preprocess(i_t m, const i_t *__restrict__ d_B_col_ptr,
                               const i_t *__restrict__ d_B_row_ind,
                               const f_t *__restrict__ d_B_values, i_t *d_BtB_row_ptr) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return; // Parallelization over row

    for (i_t col = 0; col < m; ++col) {
        // Fix the index for symmetric matrix (ensuring the same floating point order)
        i_t vec_a = idx <= col ? idx : col; // Lower index
        i_t vec_b = idx <= col ? col : idx; // Higher index

        // Compute dot product of column 'vec_a' and column 'vec_b' of B
        i_t vec_a_start = d_B_col_ptr[vec_a];
        i_t vec_a_end = d_B_col_ptr[vec_a + 1];
        i_t vec_b_start = d_B_col_ptr[vec_b];
        i_t vec_b_end = d_B_col_ptr[vec_b + 1];
        // Merge-like traversal (assuming both columns are sorted by row indices)
        f_t dot_product = 0.0;
        i_t pa = vec_a_start;
        i_t pb = vec_b_start;
        while (pa < vec_a_end && pb < vec_b_end) {
            i_t row_a = d_B_row_ind[pa];
            i_t row_b = d_B_row_ind[pb];
            if (row_a == row_b) {
                dot_product += d_B_values[pa] * d_B_values[pb];
                ++pa;
                ++pb;
            } else if (row_a < row_b) {
                ++pa;
            } else {
                ++pb;
            }
        }
        if (abs(dot_product) > 1e-12) {
            d_BtB_row_ptr[idx] += 1;
        }
    }
}

template <typename i_t, typename f_t>
__global__ void BtB_compute(i_t m, const i_t *__restrict__ d_B_col_ptr,
                            const i_t *__restrict__ d_B_row_ind, const f_t *__restrict__ d_B_values,
                            i_t *d_BtB_row_ptr, i_t *d_BtB_col_ind, f_t *d_BtB_values,
                            i_t *write_offsets) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return; // Parallelization over row

    for (i_t col = 0; col < m; ++col) {
        // Fix the index for symmetric matrix (ensuring the same floating point order)
        i_t vec_a = idx <= col ? idx : col; // Lower index
        i_t vec_b = idx <= col ? col : idx; // Higher index

        // Compute dot product of column 'vec_a' and column 'vec_b' of B
        i_t vec_a_start = d_B_col_ptr[vec_a];
        i_t vec_a_end = d_B_col_ptr[vec_a + 1];
        i_t vec_b_start = d_B_col_ptr[vec_b];
        i_t vec_b_end = d_B_col_ptr[vec_b + 1];
        // Merge-like traversal (assuming both columns are sorted by row indices)
        f_t dot_product = 0.0;
        i_t pa = vec_a_start;
        i_t pb = vec_b_start;
        while (pa < vec_a_end && pb < vec_b_end) {
            i_t row_a = d_B_row_ind[pa];
            i_t row_b = d_B_row_ind[pb];
            if (row_a == row_b) {
                dot_product += d_B_values[pa] * d_B_values[pb];
                ++pa;
                ++pb;
            } else if (row_a < row_b) {
                ++pa;
            } else {
                ++pb;
            }
        }
        if (abs(dot_product) > 1e-12) {
            // Write to BtB
            i_t pos = write_offsets[idx]++;
            d_BtB_col_ind[pos] = col;
            d_BtB_values[pos] = dot_product;
        }
    }
}

template <typename i_t, typename f_t>
__global__ void densify_Bt_cols(i_t m, i_t num_cols, i_t col_start,
                                const i_t *__restrict__ B_col_ptr,
                                const i_t *__restrict__ B_row_ind, const f_t *__restrict__ B_values,
                                f_t *__restrict__ Bt_dense_slice) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    // Columns in B_T correspond to rows in B
    const i_t row_start = col_start;
    const i_t row_end = col_start + num_cols;

    for (i_t i = B_col_ptr[idx]; i < B_col_ptr[idx + 1]; ++i) {
        i_t row_in_B = B_row_ind[i];
        if (row_in_B >= row_start && row_in_B < row_end) {
            i_t col_in_slice = row_in_B - row_start;
            f_t val = B_values[i];
            Bt_dense_slice[col_in_slice * m + idx] = val; // Column-major
        }
    }
}

template <typename i_t, typename f_t>
__global__ void count_nnz_cols(i_t m, i_t num_cols, i_t col_start, const f_t *__restrict__ X_slice,
                               i_t *__restrict__ nnz_per_col) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cols)
        return;

    i_t col_idx = idx;
    for (i_t row_idx = 0; row_idx < m; ++row_idx) {
        f_t val = X_slice[col_idx * m + row_idx]; // Column-major
        if (abs(val) > 1e-12) {
            nnz_per_col[col_idx] += 1;
        }
    }
}

template <typename i_t, typename f_t>
__global__ void fill_csc_cols(i_t m, i_t num_cols, i_t col_start, const f_t *__restrict__ X_slice,
                              i_t *__restrict__ write_offsets, i_t *__restrict__ X_row_ind,
                              f_t *__restrict__ X_values) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cols)
        return;

    i_t col_idx = idx;
    for (i_t row_idx = 0; row_idx < m; ++row_idx) {
        f_t val = X_slice[col_idx * m + row_idx]; // Column-major
        if (abs(val) > 1e-12) {
            // Get position to write using atomic increment on the column offset
            i_t pos = write_offsets[col_idx]++;
            X_row_ind[pos] = row_idx;
            X_values[pos] = val;
        }
    }
}

template <typename i_t>
__global__ void shift_col_ptrs(i_t num_elements, i_t *col_ptrs, i_t offset) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements)
        return;
    col_ptrs[idx] += offset;
}

template <typename i_t, typename f_t>
void compute_inverse(cusparseHandle_t &cusparse_handle, cudssHandle_t &cudss_handle,
                     cudssConfig_t &cudss_config, i_t m, i_t n, const i_t *d_A_col_ptr,
                     const i_t *d_A_row_ind, const f_t *d_A_values, i_t *&d_B_col_ptr,
                     i_t *&d_B_row_ind, f_t *&d_B_values, const std::vector<i_t> &basic_list,
                     i_t *&d_X_col_ptr, i_t *&d_X_row_ind, f_t *&d_X_values, i_t &nz_B, i_t &nz_X, i_t &max_nz_X,
                     const simplex_solver_settings_t<i_t, f_t> &settings) {
    // Move basic list to device
    // TODO: Consider to keep basic list on device
    i_t *d_basic_list;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_basic_list");

    // Assemble B in CSC
    build_basis_on_device<i_t, f_t>(m, d_A_col_ptr, d_A_row_ind, d_A_values, d_basic_list,
                                    d_B_col_ptr, d_B_row_ind, d_B_values, nz_B);

    // Here we assume (B^T B) is also sparse
    i_t *d_BtB_row_ptr = nullptr;
    i_t *d_BtB_col_ind = nullptr;
    f_t *d_BtB_values = nullptr;

    // 1. Preprocess to get column counts for B^T B
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_row_ptr, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_BtB_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMemset(d_BtB_row_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_BtB_row_ptr");
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    BtB_preprocess<<<grid_size, block_size>>>(m, d_B_col_ptr, d_B_row_ind, d_B_values,
                                              d_BtB_row_ptr);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "BtB_preprocess kernel");
    // Prefix sum to get row pointers for B^T B
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_BtB_row_ptr);
    thrust::exclusive_scan(thrust::cuda::par, dev_ptr, dev_ptr + m + 1, dev_ptr, i_t(0));
    // Get total NNZ for B^T B
    i_t nnz_BtB;
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(&nnz_BtB, d_BtB_row_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy nnz_BtB");
    // 2. Allocate B^T B column indices and values
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_col_ind, nnz_BtB * sizeof(i_t)),
                        "cudaMalloc d_BtB_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_values, nnz_BtB * sizeof(f_t)),
                        "cudaMalloc d_BtB_values");
    // 3. Compute B^T B
    i_t *d_write_offsets;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_write_offsets, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_write_offsets for BtB");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_write_offsets, d_BtB_row_ptr, (m + 1) * sizeof(i_t), cudaMemcpyDeviceToDevice),
        "cudaMemcpy d_write_offsets for BtB");
    BtB_compute<<<grid_size, block_size>>>(m, d_B_col_ptr, d_B_row_ind, d_B_values, d_BtB_row_ptr,
                                           d_BtB_col_ind, d_BtB_values, d_write_offsets);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "BtB_compute kernel");
    CUDA_CALL_AND_CHECK(cudaFree(d_write_offsets), "cudaFree d_write_offsets for BtB");

    // Factor (B^T B) = L*U using cuDSS and solve
    // for (B^T B) X = B^T
    // => X = (B^T B)^(-1) B^T is the
    // pseudo-inverse of B

    // We use slices to create X in CSC
    // Surprisingly, X is quite sparse for many LP
    // problems
    cudssData_t solverData;
    CUDSS_CALL_AND_CHECK(cudssDataCreate(cudss_handle, &solverData), "cudssCreateData B");

    cudssMatrix_t d_BtB_matrix_cudss;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&d_BtB_matrix_cudss, m, m, nnz_BtB, d_BtB_row_ptr,
                                              NULL, d_BtB_col_ind, d_BtB_values, CUDA_R_32I,
                                              CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL,
                                              CUDSS_BASE_ZERO),
                         "cudssMatrixCreateCsr for BTB");

    CUDSS_CALL_AND_CHECK(
        cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION, cudss_config,
                     solverData, d_BtB_matrix_cudss, nullptr, nullptr),
        "cudssExecute Analysis for B");

    std::vector<i_t> X_nnz_per_slice(settings.pinv_slices, 0);
    std::vector<f_t *> d_X_values_slices(settings.pinv_slices, nullptr);
    std::vector<i_t *> d_X_row_ind_slices(settings.pinv_slices, nullptr);
    for (i_t slice_idx = 0; slice_idx < settings.pinv_slices; ++slice_idx) {
        // Determine the number of columns in this
        // slice
        i_t cols_in_slice = slice_idx < m % settings.pinv_slices ? m / settings.pinv_slices + 1
                                                                 : m / settings.pinv_slices;
        i_t col_start = (m / settings.pinv_slices) * slice_idx +
                        std::min<i_t>(slice_idx, m % settings.pinv_slices);

        // Interpret rows of B in CSR as columns
        // of B_T in CSC
        f_t *d_Bt_slice_dense = nullptr;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_slice_dense, m * cols_in_slice * sizeof(f_t)),
                            "cudaMalloc d_Bt_dense for slice");
        CUDA_CALL_AND_CHECK(cudaMemset(d_Bt_slice_dense, 0, m * cols_in_slice * sizeof(f_t)),
                            "cudaMemset d_Bt_dense for slice");
        int block_size = 256;
        int grid_size = (m + block_size - 1) / block_size;
        densify_Bt_cols<<<grid_size, block_size>>>(m, cols_in_slice, col_start, d_B_col_ptr,
                                                   d_B_row_ind, d_B_values, d_Bt_slice_dense);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "densify_Bt_cols kernel");

        // Prepare X and RHS slice
        cudssMatrix_t d_Bt_matrix_slice_cudss;
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&d_Bt_matrix_slice_cudss, m, cols_in_slice, m,
                                                 d_Bt_slice_dense, CUDA_R_64F,
                                                 CUDSS_LAYOUT_COL_MAJOR),
                             "cudssMatrixCreateDn for B_T");
        f_t *d_X_slice;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_slice, m * cols_in_slice * sizeof(f_t)),
                            "cudaMalloc d_X_slice for pseudo-inverse");
        CUDA_CALL_AND_CHECK(cudaMemset(d_X_slice, 0, m * cols_in_slice * sizeof(f_t)),
                            "cudaMemset d_X_slice for pseudo-inverse");
        cudssMatrix_t d_X_matrix_slice_cudss; // This is the pseudo-inverse of B
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&d_X_matrix_slice_cudss, m, cols_in_slice, m,
                                                 d_X_slice, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                             "cudssMatrixCreateDn for X");

        CUDSS_CALL_AND_CHECK(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_config, solverData,
                                          d_BtB_matrix_cudss, d_X_matrix_slice_cudss,
                                          d_Bt_matrix_slice_cudss),
                             "cudssExecute Solve for B");
        CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cuDSS "
                                                     "solve");

        // Count non-zeros in d_X_slice
        // For slice > 0, we need to preserve d_X_col_ptr[col_start] which is the end of previous
        // slice
        i_t prev_slice_end = 0;
        if (slice_idx > 0) {
            CUDA_CALL_AND_CHECK(cudaMemcpy(&prev_slice_end, d_X_col_ptr + col_start, sizeof(i_t),
                                           cudaMemcpyDeviceToHost),
                                "cudaMemcpy prev_slice_end");
        }

        CUDA_CALL_AND_CHECK(
            cudaMemset(d_X_col_ptr + col_start, 0, (cols_in_slice + 1) * sizeof(i_t)),
            "cudaMemset d_X_col_ptr for slice");
        grid_size = (cols_in_slice + block_size - 1) / block_size;
        count_nnz_cols<<<grid_size, block_size>>>(m, cols_in_slice, col_start, d_X_slice,
                                                  d_X_col_ptr + col_start);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_nnz_cols kernel for X slice");
        // Prefix sum to get column pointers for this slice
        thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_X_col_ptr + col_start);
        thrust::exclusive_scan(thrust::cuda::par, dev_ptr, dev_ptr + cols_in_slice + 1, dev_ptr,
                               i_t(0));
        // Get nnz for this slice
        i_t nnz_X_slice;
        CUDA_CALL_AND_CHECK(cudaMemcpy(&nnz_X_slice, d_X_col_ptr + col_start + cols_in_slice,
                                       sizeof(i_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy nnz_X_slice");
        X_nnz_per_slice[slice_idx] = nnz_X_slice;
        // Allocate row indices and values for
        // this slice
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_row_ind_slices[slice_idx], nnz_X_slice * sizeof(i_t)),
                            "cudaMalloc d_X_row_ind for slice");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_values_slices[slice_idx], nnz_X_slice * sizeof(f_t)),
                            "cudaMalloc d_X_values for slice");
        // Fill CSC structure for this slice
        i_t *d_write_offsets;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_write_offsets, cols_in_slice * sizeof(i_t)),
                            "cudaMalloc d_write_offsets for X "
                            "slice");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_write_offsets, d_X_col_ptr + col_start,
                                       cols_in_slice * sizeof(i_t), cudaMemcpyDeviceToDevice),
                            "cudaMemcpy d_write_offsets for X slice");
        grid_size = (cols_in_slice + block_size - 1) / block_size;
        fill_csc_cols<<<grid_size, block_size>>>(m, cols_in_slice, col_start, d_X_slice,
                                                 d_write_offsets, d_X_row_ind_slices[slice_idx],
                                                 d_X_values_slices[slice_idx]);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_csc_cols kernel for X slice");
        CUDA_CALL_AND_CHECK(cudaFree(d_write_offsets), "cudaFree d_write_offsets for X slice");
        // Shift column pointers to account for previous slices
        i_t offset = prev_slice_end; // Use the accumulated offset
        grid_size = (cols_in_slice + 1 + block_size - 1) / block_size;
        shift_col_ptrs<<<grid_size, block_size>>>(cols_in_slice + 1, d_X_col_ptr + col_start,
                                                  offset);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "shift_col_ptrs kernel for X slice");

        // Cleanup slices
        CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_Bt_matrix_slice_cudss),
                             "cudssMatrixDestroy B_T slice");
        CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_X_matrix_slice_cudss),
                             "cudssMatrixDestroy X slice");
        CUDA_CALL_AND_CHECK(cudaFree(d_Bt_slice_dense), "cudaFree d_Bt_slice_dense");
        CUDA_CALL_AND_CHECK(cudaFree(d_X_slice), "cudaFree d_X_slice");
    }
    // Combine slices into final d_X
    i_t nz_B_pinv = 0;
    for (i_t slice_idx = 0; slice_idx < settings.pinv_slices; ++slice_idx) {
        nz_B_pinv += X_nnz_per_slice[slice_idx];
    }
    if (nz_B_pinv > max_nz_X) {
        if (max_nz_X > 0) {
            // Free previous allocation
            CUDA_CALL_AND_CHECK(cudaFree(d_X_row_ind), "cudaFree d_X_row_ind previous");
            CUDA_CALL_AND_CHECK(cudaFree(d_X_values), "cudaFree d_X_values previous");
        }
        max_nz_X = nz_B_pinv * settings.pinv_buffer_size_multiplier;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_row_ind, max_nz_X * sizeof(i_t)),
                            "cudaMalloc d_X_row_ind");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_values, max_nz_X * sizeof(f_t)), "cudaMalloc d_X_values");
    }
    i_t offset = 0;
    for (i_t slice_idx = 0; slice_idx < settings.pinv_slices; ++slice_idx) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_X_row_ind + offset, d_X_row_ind_slices[slice_idx],
                                       X_nnz_per_slice[slice_idx] * sizeof(i_t),
                                       cudaMemcpyDeviceToDevice),
                            "cudaMemcpy d_X_row_ind slice to "
                            "final");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_X_values + offset, d_X_values_slices[slice_idx],
                                       X_nnz_per_slice[slice_idx] * sizeof(f_t),
                                       cudaMemcpyDeviceToDevice),
                            "cudaMemcpy d_X_values slice to "
                            "final");
        offset += X_nnz_per_slice[slice_idx];
        // Free slice memory
        CUDA_CALL_AND_CHECK(cudaFree(d_X_row_ind_slices[slice_idx]), "cudaFree d_X_row_ind slice");
        CUDA_CALL_AND_CHECK(cudaFree(d_X_values_slices[slice_idx]), "cudaFree d_X_values slice");
    }
    // Set final column pointers
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_X_col_ptr + m, &nz_B_pinv, sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy final nnz to d_X_col_ptr");

    // Set output parameter
    nz_X = nz_B_pinv;

    // cuDSS cleanup
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_BtB_matrix_cudss), "cudssMatrixDestroy B_T B");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(cudss_handle, solverData), "cudssDataDestroy B");

    CUDA_CALL_AND_CHECK(cudaFree(d_BtB_row_ptr), "cudaFree d_BtB_row_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_BtB_col_ind), "cudaFree d_BtB_col_ind");
    CUDA_CALL_AND_CHECK(cudaFree(d_BtB_values), "cudaFree d_BtB_values");

    CUDA_CALL_AND_CHECK(cudaFree(d_basic_list), "cudaFree d_basic_list");
    // Note: d_X is not freed here - caller will
    // free it after use d_B and d_Bt are freed
    // after inverse update later
}

template <typename i_t, typename f_t>
__global__ void fetch_column_as_dense_kernel(i_t m, i_t col_idx, const i_t *__restrict__ B_col_ptr,
                                             const i_t *__restrict__ B_row_ind,
                                             const f_t *__restrict__ B_values,
                                             f_t *__restrict__ col_dense) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    i_t start = B_col_ptr[col_idx];
    i_t end = B_col_ptr[col_idx + 1];

    if (idx >= end - start)
        return;

    i_t row = B_row_ind[start + idx];
    f_t val = B_values[start + idx];
    col_dense[row] = val;
}

template <typename i_t, typename f_t>
void fetch_column_as_dense(i_t m, i_t col_idx, const i_t *d_B_col_ptr, const i_t *d_B_row_ind,
                           const f_t *d_B_values, f_t *col_dense) {
    // Initialize to zero
    CUDA_CALL_AND_CHECK(cudaMemset(col_dense, 0, m * sizeof(f_t)), "cudaMemset col_dense");
    int block_size = 32;
    int grid_size = (m + block_size - 1) / block_size;
    fetch_column_as_dense_kernel<<<grid_size, block_size>>>(
        m, col_idx, d_B_col_ptr, d_B_row_ind, d_B_values, col_dense);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fetch_column_as_dense_kernel");
}

template <typename i_t, typename f_t>
__global__ void fetch_row_as_dense_kernel(i_t m, i_t row_idx, const i_t *__restrict__ B_col_ptr,
                                          const i_t *__restrict__ B_row_ind,
                                          const f_t *__restrict__ B_values,
                                          f_t *__restrict__ row_dense) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t start = B_col_ptr[idx];
    i_t end = B_col_ptr[idx + 1];
    for (i_t i = start; i < end; ++i) {
        i_t row = B_row_ind[i];
        if (row == row_idx) {
            f_t val = B_values[i];
            row_dense[idx] = val;
            break;
        }
    }
}

template <typename i_t, typename f_t>
void fetch_row_as_dense(i_t m, i_t row_idx, const i_t *d_B_col_ptr, const i_t *d_B_row_ind,
                        const f_t *d_B_values, f_t *row_dense, cudaStream_t stream) {
    // Initialize to zero
    CUDA_CALL_AND_CHECK(cudaMemset(row_dense, 0, m * sizeof(f_t)), "cudaMemset row_dense");
    int block_size = 32;
    int grid_size = (m + block_size - 1) / block_size;
    fetch_row_as_dense_kernel<<<grid_size, block_size, 0, stream>>>(
        m, row_idx, d_B_col_ptr, d_B_row_ind, d_B_values, row_dense);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fetch_row_as_dense_kernel");
}

template <typename i_t>
__device__ i_t binary_search(i_t length, const i_t *__restrict__ arr, i_t target) {
    i_t left = 0;
    i_t right = length - 1;
    i_t res = -1;
    while (left <= right) {
        i_t mid = left + (right - left) / 2;
        if (arr[mid] == target) {
            res = mid;
            break;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return res;
}

template <typename i_t, typename f_t>
__global__ void extract_sparse_row(i_t m, const i_t *__restrict__ d_B_pinv_col_ptr,
                                   const i_t *__restrict__ d_B_pinv_row_ind,
                                   const f_t *__restrict__ d_B_pinv_values, f_t alpha, i_t row_idx,
                                   i_t *sparse_indices, f_t *sparse_values, i_t *write_offset) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    // Scan through column idx to find if row_idx exists
    i_t col_start = d_B_pinv_col_ptr[idx];
    i_t col_end = d_B_pinv_col_ptr[idx + 1];
    for (i_t j = col_start; j < col_end; ++j) {
        if (d_B_pinv_row_ind[j] == row_idx) {
            i_t write_pos = atomicAdd(write_offset, 1);
            sparse_indices[write_pos] = idx;
            sparse_values[write_pos] = d_B_pinv_values[j] * alpha;
            break;
        }
    }
}

template <typename i_t, typename f_t>
__global__ void spmv_csc_kernel(i_t n, i_t m, const i_t *__restrict__ col_ptr,
                                const i_t *__restrict__ row_ind, const f_t *__restrict__ values,
                                const f_t *__restrict__ x, f_t *__restrict__ y, f_t alpha,
                                f_t beta) {
    // Compute y = alpha * A * x + beta * y where A is in CSC format
    // Strategy: Each thread processes one column, performing scatter operation
    // This is efficient when columns are relatively sparse and we have enough parallelism

    i_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx >= n)
        return;

    // Get the scaling factor for this column
    f_t x_val = x[col_idx];
    if (abs(x_val) < 1e-20)
        return; // Skip zero columns to avoid unnecessary atomic operations

    f_t scale = alpha * x_val;

    // Iterate over non-zeros in this column and scatter to result
    i_t col_start = col_ptr[col_idx];
    i_t col_end = col_ptr[col_idx + 1];

    for (i_t i = col_start; i < col_end; ++i) {
        i_t row = row_ind[i];
        f_t val = values[i];
        // Use atomicAdd since multiple threads may write to the same output element
        atomicAdd(&y[row], scale * val);
    }
}

// Alternative version: Process columns in chunks to reduce atomics
template <typename i_t, typename f_t>
__global__ void
spmv_csc_atomic_free_kernel(i_t n, i_t m, const i_t *__restrict__ col_ptr,
                            const i_t *__restrict__ row_ind, const f_t *__restrict__ values,
                            const f_t *__restrict__ x, f_t *__restrict__ y, f_t alpha, f_t beta) {
    // Each warp processes one column cooperatively, writing to shared memory first
    // to reduce atomic contention, then flushing to global memory

    extern __shared__ f_t shared_mem[];

    i_t warp_id = threadIdx.x / 32;
    i_t lane_id = threadIdx.x % 32;
    i_t col_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (col_idx >= n)
        return;

    f_t x_val = x[col_idx];
    if (abs(x_val) < 1e-20)
        return;

    f_t scale = alpha * x_val;
    i_t col_start = col_ptr[col_idx];
    i_t col_end = col_ptr[col_idx + 1];
    i_t nnz_in_col = col_end - col_start;

    // Process column elements in parallel within the warp
    for (i_t i = col_start + lane_id; i < col_end; i += 32) {
        i_t row = row_ind[i];
        f_t val = values[i];
        atomicAdd(&y[row], scale * val);
    }
}

template <typename i_t, typename f_t>
__global__ void rank1_symbolic(i_t m, const i_t *__restrict__ d_B_pinv_col_ptr,
                               const i_t *__restrict__ d_B_pinv_row_ind,
                               const f_t *__restrict__ d_vector_U_values,
                               const f_t *__restrict__ d_vector_V_values, i_t *d_new_B_pinv_col_ptr,
                               const f_t scale) {
    // Assuming U @ V^T rank-1 update
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    // Parallelize over columns of V
    // If V[idx] is non-zero, merge B_pinv_row_ind with U's non-zero rows
    f_t v_val = d_vector_V_values[idx];
    i_t col_start = d_B_pinv_col_ptr[idx];
    i_t col_end = d_B_pinv_col_ptr[idx + 1];
    i_t nnz_in_col = col_end - col_start;
    i_t num_existing_rows = nnz_in_col;

    if (abs(v_val) * abs(scale) > 1e-12) {
        // Count new nnz due to U using merge
        i_t i = col_start;
        i_t j = 0; // index for U
        while (i < col_end || j < m) {
            i_t row_idx_B = (i < col_end) ? d_B_pinv_row_ind[i] : m + 1;
            i_t row_idx_U = (j < m) ? j : m + 1;
            if (i < col_end && (row_idx_B < row_idx_U)) {
                // From B_pinv
                i++;
            } else if (j < m && (row_idx_U < row_idx_B)) {
                // From U
                f_t u_val = d_vector_U_values[j];
                if (abs(u_val) * abs(v_val) * abs(scale) > 1e-12) {
                    nnz_in_col++;
                }
                j++;
            } else if (i < col_end && j < m && (row_idx_B == row_idx_U)) {
                // Both have the same row index
                i++;
                j++;
            } else {
                break;
            }
        }
    }
    // Write new nnz to d_new_B_pinv_col_ptr
    d_new_B_pinv_col_ptr[idx] = nnz_in_col;
}

template <typename i_t, typename f_t>
__global__ void rank1_update(i_t m, const i_t *__restrict__ d_B_pinv_col_ptr,
                             const i_t *__restrict__ d_B_pinv_row_ind,
                             const f_t *__restrict__ d_B_pinv_values, i_t *d_new_B_pinv_col_ptr,
                             i_t *d_new_B_pinv_row_ind, f_t *d_new_B_pinv_values,
                             f_t *d_vector_U_values, f_t *d_vector_V_values, const f_t scale) {
    // Assuming U @ V^T rank-1 update
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    // Parallelize over columns of V
    f_t v_val = d_vector_V_values[idx];
    i_t col_start = d_B_pinv_col_ptr[idx];
    i_t col_end = d_B_pinv_col_ptr[idx + 1];
    i_t write_offset = d_new_B_pinv_col_ptr[idx];

    if (abs(v_val) * abs(scale) > 1e-12) {
        // Merge B_pinv column with U's non-zero rows in sorted order
        i_t i = col_start;
        i_t j = 0; // index for U
        while (i < col_end || j < m) {
            i_t row_idx_B = (i < col_end) ? d_B_pinv_row_ind[i] : m + 1;
            i_t row_idx_U = (j < m) ? j : m + 1;
            if (i < col_end && (row_idx_B < row_idx_U)) {
                // Copy from B_pinv
                d_new_B_pinv_row_ind[write_offset] = d_B_pinv_row_ind[i];
                d_new_B_pinv_values[write_offset] = d_B_pinv_values[i];
                i++;
                write_offset++;
            } else if (j < m && (row_idx_U < row_idx_B)) {
                // Copy from U
                f_t u_val = d_vector_U_values[j];
                if (abs(v_val) * abs(u_val) * abs(scale) > 1e-12) {
                    d_new_B_pinv_row_ind[write_offset] = row_idx_U;
                    d_new_B_pinv_values[write_offset] = scale * u_val * v_val;
                    write_offset++;
                }
                j++;
            } else if (i < col_end && j < m && (row_idx_B == row_idx_U)) {
                // Both have the same row index
                d_new_B_pinv_row_ind[write_offset] = row_idx_B;
                d_new_B_pinv_values[write_offset] = d_B_pinv_values[i];
                f_t u_val = d_vector_U_values[j];
                if (abs(v_val) * abs(u_val) * abs(scale) > 1e-12) {
                    d_new_B_pinv_values[write_offset] += scale * u_val * v_val;
                }
                i++;
                j++;
                write_offset++;
            } else {
                break;
            }
        }
    } else {
        // Just copy the entire column from B_pinv
        for (i_t i = col_start; i < col_end; ++i) {
            d_new_B_pinv_row_ind[write_offset] = d_B_pinv_row_ind[i];
            d_new_B_pinv_values[write_offset] = d_B_pinv_values[i];
            write_offset++;
        }
    }
}

template <typename i_t, typename f_t>
bool eta_update_inverse(cublasHandle_t cublas_handle, i_t m, cusparseSpMatDescr_t &B_pinv_cusparse,
                        i_t *&d_B_pinv_col_ptr, i_t *&d_B_pinv_row_ind, f_t *&d_B_pinv_values,
                        i_t *&d_new_B_pinv_col_ptr, i_t *&d_new_B_pinv_row_ind, f_t *&d_new_B_pinv_values,
                        i_t &nz_B_pinv, i_t &max_nz_B_pinv, i_t &max_nz_new_B_pinv, f_t *eta_b_old, f_t *eta_b_new, f_t *eta_v, f_t *eta_c,
                        f_t *eta_d, const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
                        const f_t *d_A_values, const i_t *d_B_col_ptr, const i_t *d_B_row_ind,
                        const f_t *d_B_values, i_t basic_leaving_index, i_t entering_index, const simplex_solver_settings_t<i_t, f_t> &settings) {
    const i_t j = basic_leaving_index; // Index of leaving variable in basis

    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.start("Eta Update fetch columns");
    }
    // Fetch b_old and b_new as dense vectors
    phase2_cu::fetch_column_as_dense(m, basic_leaving_index, d_B_col_ptr, d_B_row_ind, d_B_values,
                                     eta_b_old);
    phase2_cu::fetch_column_as_dense(m, entering_index, d_A_col_ptr, d_A_row_ind, d_A_values,
                                     eta_b_new);
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.stop("Eta Update fetch columns");
    }
    // Sherman-Morrison formula for updating B_inv when replacing column j:
    // B_new = B with column j replaced by b_new
    // B_new_inv = B_inv - (B_inv @ u @ e_j^T @ B_inv) / (1 + e_j^T @ B_inv @ u)
    // where u = b_new - b_old
    //
    // This simplifies to:
    // p = B_inv @ (b_new - b_old)
    // row_j = j-th row of B_inv
    // B_new_inv = B_inv - (p @ row_j) / (1 +
    // p[j])

    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.start("Eta Update compute vectors");
    }
    f_t alpha, beta;

    // Compute u = b_new - b_old (store in eta_c
    // temporarily)
    CUDA_CALL_AND_CHECK(cudaMemcpy(eta_c, eta_b_new, m * sizeof(f_t), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy eta_c = b_new");
    alpha = -1.0;
    CUBLAS_CALL_AND_CHECK(cublasDaxpy(cublas_handle, m, &alpha, eta_b_old, 1, eta_c, 1),
                          "cublasDaxpy compute u = b_new - b_old");

    // Compute p = B_inv @ u
    // Note: eta_p already contains B_inv @ b_old
    // from the caller So we need to recompute: p
    // = B_inv @ (b_new - b_old)
    alpha = 1.0;
    beta = 0.0;

    // Use custom CSC SpMV kernel for B_inv @ u
    // First, initialize output vector with beta * y
    CUDA_CALL_AND_CHECK(cudaMemset(eta_v, 0, m * sizeof(f_t)), "cudaMemset eta_v to zero");

    // Launch spmv_csc_kernel
    i_t block_size = 256;
    i_t grid_size = (m + block_size - 1) / block_size;
    spmv_csc_kernel<<<grid_size, block_size>>>(m, m, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                               d_B_pinv_values,
                                               eta_c, // input vector (u = b_new - b_old)
                                               eta_v, // output vector (p = B_inv @ u)
                                               alpha, beta);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "spmv_csc_kernel launch");

    // Get p[j] and compute denominator: 1 + p[j]
    f_t p_j;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&p_j, eta_v + j, sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy p_j from device");
    f_t denom = 1.0 + p_j;

    if (std::abs(denom) < 1e-10) {
        // Singular or near-singular update -
        // should refactor instead
        return false;
    }

    // Kernel parameters
    block_size = 256;
    grid_size = (m + block_size - 1) / block_size;

    // Extract row j of B_inv (store in eta_d)
    // B_inv is column-major, so row j is at offsets j, j+m, j+2m, ...
    phase2_cu::fetch_row_as_dense(m, j, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, eta_d,
                                  0);
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.stop("Eta Update compute vectors");
    }

    // B_inv = B_inv - (p @ row_j) / denom
    f_t scale = -1.0 / denom;
    // Custom kernel to perform sparse rank-1 update
    // 1. Count the additional number of non-zeros introduced
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.start("Eta Update rank-1 step 1");
    }
    CUDA_CALL_AND_CHECK(cudaMemset(d_new_B_pinv_col_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_new_B_pinv_col_ptr");
    grid_size = (m + block_size - 1) / block_size;
    rank1_symbolic<<<grid_size, block_size>>>(m, d_B_pinv_col_ptr, d_B_pinv_row_ind, eta_v, eta_d,
                                              d_new_B_pinv_col_ptr, scale);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "get_rank1_delta_nnz_nnz kernel");
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.stop("Eta Update rank-1 step 1");
    }
    // 2. Allocate new arrays for updated B_pinv
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.start("Eta Update rank-1 step 2");
    }
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_new_B_pinv_col_ptr);
    thrust::exclusive_scan(thrust::cuda::par, dev_ptr, dev_ptr + m + 1, dev_ptr, i_t(0));
    i_t nz_new_B_pinv = 0;
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(&nz_new_B_pinv, d_new_B_pinv_col_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy nz_new_B_pinv to host");
    if (nz_new_B_pinv > max_nz_new_B_pinv) {
        // Exceed allocated space for new B_pinv, reallocate
        CUDA_CALL_AND_CHECK(cudaFree(d_new_B_pinv_row_ind), "cudaFree d_new_B_pinv_row_ind");
        CUDA_CALL_AND_CHECK(cudaFree(d_new_B_pinv_values), "cudaFree d_new_B_pinv_values");
        max_nz_new_B_pinv = nz_new_B_pinv * settings.pinv_buffer_size_multiplier;
        CUDA_CALL_AND_CHECK(
            cudaMalloc(&d_new_B_pinv_row_ind, max_nz_new_B_pinv * sizeof(i_t)),
            "cudaMalloc d_new_B_pinv_row_ind");
        CUDA_CALL_AND_CHECK(
            cudaMalloc(&d_new_B_pinv_values, max_nz_new_B_pinv * sizeof(f_t)),
            "cudaMalloc d_new_B_pinv_values");
    }
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.stop("Eta Update rank-1 step 2");
    }
    // 3. Launch kernel to perform the rank-1 update and fill new arrays
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.start("Eta Update rank-1 step 3");
    }
    grid_size = (m + block_size - 1) / block_size;
    rank1_update<<<grid_size, block_size>>>(m, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values,
                                            d_new_B_pinv_col_ptr, d_new_B_pinv_row_ind,
                                            d_new_B_pinv_values, eta_v, eta_d, scale);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "rank1_csc_update kernel");
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.stop("Eta Update rank-1 step 3");
    }
    // 4. Clean up and update pointers
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.start("Eta Update cleanup");
    }
    std::swap(d_B_pinv_col_ptr, d_new_B_pinv_col_ptr);
    std::swap(d_B_pinv_row_ind, d_new_B_pinv_row_ind);
    std::swap(d_B_pinv_values, d_new_B_pinv_values);
    std::swap(max_nz_B_pinv, max_nz_new_B_pinv);
    nz_B_pinv = nz_new_B_pinv;
    // Update cusparse matrix descriptor
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(B_pinv_cusparse),
                            "cusparseDestroySpMat B_pinv_cusparse");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&B_pinv_cusparse, m, m, nz_B_pinv, d_B_pinv_col_ptr,
                                              d_B_pinv_row_ind, d_B_pinv_values, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F),
                            "cusparseCreateCSCMat B_pinv_cusparse");
    if (settings.profile) {
        // cudaDeviceSynchronize();
        settings.timer.stop("Eta Update cleanup");
    }

    return true; // Successful update
}

template <typename i_t, typename f_t>
__global__ void denseMatrixSparseVectorMulKernel(i_t m, const f_t *__restrict__ b_inv,
                                                 const i_t *__restrict__ sparse_vec_indices,
                                                 const f_t *__restrict__ sparse_vec_values,
                                                 i_t nz_sparse_vec, f_t *__restrict__ result,
                                                 bool transpose) {
    id_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    f_t sum = 0.0;
    for (i_t k = 0; k < nz_sparse_vec; ++k) {
        i_t vec_idx = sparse_vec_indices[k];
        f_t vec_val = sparse_vec_values[k];

        i_t mat_idx;
        if (transpose) {
            // We want (B_inv^T)_{idx, vec_idx} =
            // (B_inv)_{vec_idx, idx} B_inv is
            // col-major, so element (row=vec_idx,
            // col=idx) is at: col * m + row = idx
            // * m + vec_idx
            mat_idx = idx * m + vec_idx;
        } else {
            // We want (B_inv)_{idx, vec_idx}
            // B_inv is col-major, so element
            // (row=idx, col=vec_idx) is at: col *
            // m + row = vec_idx * m + idx
            mat_idx = vec_idx * m + idx;
        }

        sum += vec_val * b_inv[mat_idx];
    }
    result[idx] = sum;
}

template <typename i_t, typename f_t>
__global__ void construct_c_basic_kernel(i_t m, const i_t *__restrict__ basic_list,
                                         const f_t *__restrict__ objective,
                                         f_t *__restrict__ c_basic) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in
                                   // original matrix
    c_basic[idx] = objective[col_idx];
}

template <typename i_t, typename f_t>
__global__ void scatter_vector_kernel(i_t nz, const i_t *__restrict__ indices,
                                      const f_t *__restrict__ values, f_t *__restrict__ dense) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nz)
        return;
    dense[indices[idx]] = values[idx];
}
template <typename i_t, typename f_t>
void sparse_pinv_solve_gpu_dense_rhs(cusparseHandle_t &cusparse_handle, i_t m,
                                     const i_t *d_B_pinv_col_ptr, const i_t *d_B_pinv_row_ind,
                                     const f_t *d_B_pinv_values, i_t nz_Binv, const f_t *d_rhs,
                                     f_t *&d_x, bool transpose) {
    // Create descriptors
    // Matrix is CSC. We treat it as CSR of the
    // transpose. (B_inv)_CSC = (B_inv^T)_CSR
    cusparseSpMatDescr_t mat_desc;
    cudaDataType computeType = std::is_same<f_t, double>::value ? CUDA_R_64F : CUDA_R_32F;

    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(&mat_desc, m, m, nz_Binv, (void *) d_B_pinv_col_ptr,
                                              (void *) d_B_pinv_row_ind, (void *) d_B_pinv_values,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, computeType),
                            "cusparseCreateCsr");

    cusparseDnVecDescr_t vec_rhs, vec_x;
    CUSPARSE_CALL_AND_CHECK(cusparseCreateDnVec(&vec_rhs, m, (void *) d_rhs, computeType),
                            "cusparseCreateDnVec rhs");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateDnVec(&vec_x, m, d_x, computeType),
                            "cusparseCreateDnVec x");

    f_t alpha = 1.0;
    f_t beta = 0.0;
    // If transpose=false we want B in CSR =>
    // (B_inv)_CSC = (B_inv^T)_CSR -> transpose ->
    // ((B_inv^T)_CSR)^T = B_inv_CSR If
    // transpose=true (solve B_inv^T * b), we need
    // D * b. If transpose=true we want B^T in CSR
    // => (B_inv)_CSC = (B_inv^T)_CSR -> no
    // transpose -> (B_inv^T)_CSR
    cusparseOperation_t op =
        transpose ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;

    // SpMV
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    CUSPARSE_CALL_AND_CHECK(cusparseSpMV_bufferSize(cusparse_handle, op, &alpha, mat_desc, vec_rhs,
                                                    &beta, vec_x, computeType,
                                                    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize),
                            "cusparseSpMV_bufferSize");
    CUDA_CALL_AND_CHECK(cudaMalloc(&dBuffer, bufferSize), "cudaMalloc buffer");

    CUSPARSE_CALL_AND_CHECK(cusparseSpMV(cusparse_handle, op, &alpha, mat_desc, vec_rhs, &beta,
                                         vec_x, computeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer),
                            "cusparseSpMV");

    // Cleanup
    CUDA_CALL_AND_CHECK(cudaFree(dBuffer), "cudaFree dBuffer");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(mat_desc), "cusparseDestroySpMat");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroyDnVec(vec_rhs), "cusparseDestroyDnVec rhs");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroyDnVec(vec_x), "cusparseDestroyDnVec x");
}
template <typename i_t, typename f_t>
void sparse_pinv_solve_gpu_sparse_rhs(cusparseHandle_t &cusparse_handle, i_t m,
                                      const i_t *d_B_pinv_col_ptr, const i_t *d_B_pinv_row_ind,
                                      const f_t *d_B_pinv_values, i_t nz, const i_t *d_rhs_indices,
                                      const f_t *d_rhs_values, i_t nz_rhs, f_t *&d_x,
                                      bool transpose) {
    // Allocate dense vectors
    f_t *d_rhs;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs, m * sizeof(f_t)), "cudaMalloc d_rhs");
    CUDA_CALL_AND_CHECK(cudaMemset(d_rhs, 0, m * sizeof(f_t)), "cudaMemset d_rhs");

    // Copy sparse RHS to device and scatter

    int block_size = 256;
    int grid_size = (nz_rhs + block_size - 1) / block_size;
    scatter_vector_kernel<<<grid_size, block_size>>>(nz_rhs, d_rhs_indices, d_rhs_values, d_rhs);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "scatter_vector_kernel");

    sparse_pinv_solve_gpu_dense_rhs(cusparse_handle, m, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                    d_B_pinv_values, nz, d_rhs, d_x, transpose);

    // Cleanup
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs), "cudaFree d_rhs");
}

template <typename i_t, typename f_t>
void sparse_pinv_solve_host_sparse_rhs(cusparseHandle_t &cusparse_handle, i_t m,
                                       const i_t *d_B_pinv_col_ptr, const i_t *d_B_pinv_row_ind,
                                       const f_t *d_B_pinv_values, i_t nz,
                                       const sparse_vector_t<i_t, f_t> &rhs,
                                       sparse_vector_t<i_t, f_t> &x, bool transpose) {
    // // Allocate dense vectors
    f_t *d_x;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_x, m * sizeof(f_t)), "cudaMalloc d_x");

    // Copy sparse RHS to device and scatter
    i_t *d_rhs_indices;
    f_t *d_rhs_values;
    i_t nz_rhs = rhs.i.size();
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs_indices, nz_rhs * sizeof(i_t)),
                        "cudaMalloc d_rhs_indices");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs_values, nz_rhs * sizeof(f_t)), "cudaMalloc d_rhs_values");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_rhs_indices, rhs.i.data(), nz_rhs * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy rhs indices");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_rhs_values, rhs.x.data(), nz_rhs * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy rhs values");

    sparse_pinv_solve_gpu_sparse_rhs(cusparse_handle, m, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                     d_B_pinv_values, nz, d_rhs_indices, d_rhs_values, nz_rhs, d_x,
                                     transpose);

    // Copy back and sparsify
    std::vector<f_t> h_x_dense(m);
    CUDA_CALL_AND_CHECK(cudaMemcpy(h_x_dense.data(), d_x, m * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy d_x to host");

    x.i.clear();
    x.x.clear();
    for (i_t i = 0; i < m; ++i) {
        if (std::abs(h_x_dense[i]) > 1e-12) {
            x.i.push_back(i);
            x.x.push_back(h_x_dense[i]);
        }
    }

    // Cleanup
    CUDA_CALL_AND_CHECK(cudaFree(d_x), "cudaFree d_x");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs_indices), "cudaFree d_rhs_indices");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs_values), "cudaFree d_rhs_values");
}

// TODO: kernelify
template <typename i_t, typename f_t>
__global__ void compute_reduced_costs_nonbasic_kernel(const f_t *d_objective,
                                                      const i_t *d_A_col_ptr,
                                                      const i_t *d_A_row_ind, const f_t *d_A_values,
                                                      const f_t *d_y, const i_t *d_nonbasic_list,
                                                      f_t *&d_z, const i_t m, const i_t n) {
    // // zN = cN - N'*y
    // for (i_t k = 0; k < n - m; k++) {
    //     const i_t j = nonbasic_list[k];
    //     // z_j <- c_j
    //     z[j] = objective[j];
    //
    //     // z_j <- z_j - A(:, j)'*y
    //     const i_t col_start = A.col_start[j];
    //     const i_t col_end = A.col_start[j + 1];
    //     f_t dot = 0.0;
    //     for (i_t p = col_start; p < col_end;
    //     ++p) {
    //         dot += A.x[p] * y[A.i[p]];
    //     }
    //     z[j] -= dot;
    // }
    // // zB = 0
    // for (i_t k = 0; k < m; ++k) {
    //     z[basic_list[k]] = 0.0;
    // }
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - m)
        return;
    const i_t j = d_nonbasic_list[idx];
    d_z[j] = d_objective[j];

    const i_t col_start = d_A_col_ptr[j];
    const i_t col_end = d_A_col_ptr[j + 1];
    f_t dot = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
        dot += d_A_values[p] * d_y[d_A_row_ind[p]];
    }
    d_z[j] -= dot;
}

template <typename i_t, typename f_t>
__global__ void compute_reduced_costs_basic_kernel(const i_t *d_basic_list, f_t *&z, const i_t m) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;
    z[d_basic_list[idx]] = 0.0;
}

template <typename i_t, typename f_t>
void compute_reduced_costs(const f_t *d_objective, const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
                           const f_t *d_A_values, const f_t *d_y, const i_t *d_basic_list,
                           const i_t *d_nonbasic_list, f_t *&z, const i_t m, const i_t n) {
    int block_size = 256;
    int grid_size_nonbasic = ((n - m) + block_size - 1) / block_size;
    compute_reduced_costs_nonbasic_kernel<<<grid_size_nonbasic, block_size>>>(
        d_objective, d_A_col_ptr, d_A_row_ind, d_A_values, d_y, d_nonbasic_list, z, m, n);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_reduced_costs_nonbasic_kernel");
    int grid_size_basic = (m + block_size - 1) / block_size;
    compute_reduced_costs_basic_kernel<<<grid_size_basic, block_size>>>(d_basic_list, z, m);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_reduced_costs_basic_kernel");
}

template <typename i_t, typename f_t>
i_t compute_delta_x(const lp_problem_t<i_t, f_t> &lp, cusparseHandle_t &cusparse_handle,
                    i_t *d_B_pinv_col_ptr, i_t *d_B_pinv_row_ind, const f_t *d_B_pinv_values,
                    i_t nz_B_pinv, i_t entering_index, i_t leaving_index, i_t basic_leaving_index,
                    i_t direction, const std::vector<i_t> &basic_list,
                    const std::vector<f_t> &delta_x_flip,
                    const sparse_vector_t<i_t, f_t> &rhs_sparse, const std::vector<f_t> &x,
                    sparse_vector_t<i_t, f_t> &scaled_delta_xB_sparse, std::vector<f_t> &delta_x,
                    i_t m) {
    f_t delta_x_leaving = direction == 1 ? lp.lower[leaving_index] - x[leaving_index]
                                         : lp.upper[leaving_index] - x[leaving_index];
    // B*w = -A(:, entering)
    //   ft.b_solve(rhs_sparse,
    //   scaled_delta_xB_sparse, utilde_sparse);
    phase2_cu::sparse_pinv_solve_host_sparse_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                                 d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                                                 rhs_sparse, scaled_delta_xB_sparse, false);
    scaled_delta_xB_sparse.negate();

#ifdef CHECK_B_SOLVE
    std::vector<f_t> scaled_delta_xB(m);
    {
        std::vector<f_t> residual_B(m);
        b_multiply(lp, basic_list, scaled_delta_xB, residual_B);
        f_t err_max = 0;
        for (i_t k = 0; k < m; ++k) {
            const f_t err = std::abs(rhs[k] + residual_B[k]);
            if (err >= 1e-6) {
                settings.log.printf("Bsolve diff %d %e rhs %e "
                                    "residual %e\n",
                                    k, err, rhs[k], residual_B[k]);
            }
            err_max = std::max(err_max, err);
        }
        if (err_max > 1e-6) {
            settings.log.printf("B multiply error %e\n", err_max);
        }
    }
#endif

    f_t scale = scaled_delta_xB_sparse.find_coefficient(basic_leaving_index);
    if (scale != scale) {
        // We couldn't find a coefficient for the
        // basic leaving index. The coefficient
        // might be very small. Switch to a
        // regular solve and try to recover.
        std::vector<f_t> rhs;
        rhs_sparse.to_dense(rhs);
        // const i_t m = basic_list.size();
        std::vector<f_t> scaled_delta_xB(m);
        // ft.b_solve(rhs, scaled_delta_xB);
        f_t *d_scaled_delta_xB;
        f_t *d_rhs;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs, m * sizeof(f_t)), "cudaMalloc d_rhs");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_rhs, rhs.data(), m * sizeof(f_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy rhs to d_rhs");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_scaled_delta_xB, m * sizeof(f_t)),
                            "cudaMalloc d_scaled_delta_xB");
        CUDA_CALL_AND_CHECK(cudaMemset(d_scaled_delta_xB, 0, m * sizeof(f_t)),
                            "cudaMemset d_scaled_delta_xB");
        sparse_pinv_solve_gpu_dense_rhs(cusparse_handle, m, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                        d_B_pinv_values, nz_B_pinv, d_rhs, d_scaled_delta_xB,
                                        false);
        CUDA_CALL_AND_CHECK(cudaMemcpy(scaled_delta_xB.data(), d_scaled_delta_xB, m * sizeof(f_t),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy d_scaled_delta_xB to host");
        CUDA_CALL_AND_CHECK(cudaFree(d_scaled_delta_xB), "cudaFree d_scaled_delta_xB");
        if (scaled_delta_xB[basic_leaving_index] != 0.0 &&
            !std::isnan(scaled_delta_xB[basic_leaving_index])) {
            scaled_delta_xB_sparse.from_dense(scaled_delta_xB);
            scaled_delta_xB_sparse.negate();
            scale = -scaled_delta_xB[basic_leaving_index];
        } else {
            return -1;
        }
    }
    const f_t primal_step_length = delta_x_leaving / scale;
    const i_t scaled_delta_xB_nz = scaled_delta_xB_sparse.i.size();
    for (i_t k = 0; k < scaled_delta_xB_nz; ++k) {
        const i_t j = basic_list[scaled_delta_xB_sparse.i[k]];
        delta_x[j] = primal_step_length * scaled_delta_xB_sparse.x[k];
    }
    delta_x[leaving_index] = delta_x_leaving;
    delta_x[entering_index] = primal_step_length;
    return 0;
}
template <typename i_t>
__global__ void get_row_nnz(i_t m, const i_t *__restrict__ d_B_pinv_col_ptr,
                            const i_t *__restrict__ d_B_pinv_row_ind, i_t row_idx, i_t *nnz) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    // Scan through column idx to find if row_idx exists
    i_t col_start = d_B_pinv_col_ptr[idx];
    i_t col_end = d_B_pinv_col_ptr[idx + 1];
    for (i_t j = col_start; j < col_end; ++j) {
        if (d_B_pinv_row_ind[j] == row_idx) {
            atomicAdd(nnz, 1);
            break;
        }
    }
}

template <typename i_t, typename f_t>
void compute_delta_y(cusparseHandle_t &cusparse_handle, const i_t *d_B_pinv_col_ptr,
                     const i_t *d_B_pinv_row_ind, const f_t *d_B_pinv_values, i_t nz_B_pinv,
                     i_t basic_leaving_index, i_t direction, i_t *&d_delta_y_sparse_indices,
                     f_t *&d_delta_y_sparse_values, i_t &nz_delta_y_sparse, i_t m) {
    i_t *d_nnz;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_nnz, sizeof(i_t)), "cudaMalloc d_nnz");
    CUDA_CALL_AND_CHECK(cudaMemset(d_nnz, 0, sizeof(i_t)), "cudaMemset d_nnz");

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    // Use existing kernel to count NNZ in the row
    phase2_cu::get_row_nnz<<<grid_size, block_size>>>(m, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                                      basic_leaving_index, d_nnz);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "get_row_nnz kernel for delta_y");
    CUDA_CALL_AND_CHECK(cudaMemcpy(&nz_delta_y_sparse, d_nnz, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy nnz");

    CUDA_CALL_AND_CHECK(cudaMalloc(&d_delta_y_sparse_indices, nz_delta_y_sparse * sizeof(i_t)),
                        "cudaMalloc d_delta_y_sparse_indices");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_delta_y_sparse_values, nz_delta_y_sparse * sizeof(f_t)),
                        "cudaMalloc d_delta_y_sparse_values");

    // Reset counter for writing
    CUDA_CALL_AND_CHECK(cudaMemset(d_nnz, 0, sizeof(i_t)), "cudaMemset d_nnz");
    // 3. Extract the row using existing kernel
    phase2_cu::extract_sparse_row<<<grid_size, block_size>>>(
        m, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, (f_t) -direction,
        basic_leaving_index, d_delta_y_sparse_indices, d_delta_y_sparse_values, d_nnz);

    CUDA_CALL_AND_CHECK(cudaMemcpy(&nz_delta_y_sparse, d_nnz, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy nnz");

    CUDA_CALL_AND_CHECK(cudaGetLastError(), "extract_sparse_row kernel for delta_y");
}

template <typename i_t, typename f_t>
void compute_delta_z(const csc_matrix_t<i_t, f_t> &A_transpose,
                     const sparse_vector_t<i_t, f_t> &delta_y, i_t leaving_index, i_t direction,
                     std::vector<i_t> &nonbasic_mark, std::vector<i_t> &delta_z_mark,
                     std::vector<i_t> &delta_z_indices, std::vector<f_t> &delta_z) {
    // delta_zN = - N'*delta_y
    const i_t nz_delta_y = delta_y.i.size();
    for (i_t k = 0; k < nz_delta_y; k++) {
        const i_t i = delta_y.i[k];
        const f_t delta_y_i = delta_y.x[k];
        if (std::abs(delta_y_i) < 1e-12) {
            continue;
        }
        const i_t row_start = A_transpose.col_start[i];
        const i_t row_end = A_transpose.col_start[i + 1];
        for (i_t p = row_start; p < row_end; ++p) {
            const i_t j = A_transpose.i[p];
            if (nonbasic_mark[j] >= 0) {
                delta_z[j] -= delta_y_i * A_transpose.x[p];
                if (!delta_z_mark[j]) {
                    delta_z_mark[j] = 1;
                    delta_z_indices.push_back(j); // Note delta_z_indices has n elements reserved
                }
            }
        }
    }

    // delta_zB = sigma*ei
    delta_z[leaving_index] = direction;

#ifdef CHECK_CHANGE_IN_REDUCED_COST
    delta_y_sparse.to_dense(delta_y);
    std::vector<f_t> delta_z_check(n);
    std::vector<i_t> delta_z_mark_check(n, 0);
    std::vector<i_t> delta_z_indices_check;
    phase2::compute_reduced_cost_update(lp, basic_list, nonbasic_list, delta_y, leaving_index,
                                        direction, delta_z_mark_check, delta_z_indices_check,
                                        delta_z_check);
    f_t error_check = 0.0;
    for (i_t k = 0; k < n; ++k) {
        const f_t diff = std::abs(delta_z[k] - delta_z_check[k]);
        if (diff > 1e-6) {
            printf("delta_z error %d transpose %e no transpose %e diff %e\n", k, delta_z[k],
                   delta_z_check[k], diff);
        }
        error_check = std::max(error_check, diff);
    }
    if (error_check > 1e-6) {
        printf("delta_z error %e\n", error_check);
    }
#endif
}

template <typename i_t, typename f_t>
__global__ void initialize_steepest_edge_norms_find_row_degree_kernel(
    i_t m, const i_t *d_basic_list, const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
    const f_t *d_A_values, i_t *d_row_degree, i_t *d_mapping, f_t *d_coeff) {
    i_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m)
        return;

    const i_t j = d_basic_list[k];
    const i_t col_start = d_A_col_ptr[j];
    const i_t col_end = d_A_col_ptr[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
        const i_t i = d_A_row_ind[p];
        atomicAdd(&d_row_degree[i], 1);
        // column j of A is column k of B
        d_mapping[k] = i;
        d_coeff[k] = d_A_values[p];
    }
}

template <typename i_t, typename f_t>
i_t initialize_steepest_edge_norms(const lp_problem_t<i_t, f_t> &lp,
                                   const simplex_solver_settings_t<i_t, f_t> &settings,
                                   const f_t start_time, const i_t *d_basic_list,
                                   cusparseHandle_t &cusparse_handle, i_t *d_B_pinv_col_ptr,
                                   i_t *d_B_pinv_row_ind, f_t *d_B_pinv_values, i_t nz_B_pinv,
                                   std::vector<f_t> &delta_y_steepest_edge, i_t m) {
    // Start of GPU code, but we fall back to CPU
    // for now i_t *d_row_degree, *d_mapping; f_t
    // *d_coeff;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_row_degree,
    // m * sizeof(i_t)), "cudaMalloc
    // d_row_degree");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_mapping,
    // m * sizeof(i_t)), "cudaMalloc d_mapping");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_coeff, m
    // * sizeof(f_t)), "cudaMalloc d_coeff");
    // CUDA_CALL_AND_CHECK(cudaMemset(d_row_degree,
    // 0, m * sizeof(i_t)), "cudaMemset
    // d_row_degree");
    // CUDA_CALL_AND_CHECK(cudaMemset(d_mapping,
    // -1, m * sizeof(i_t)), "cudaMemset
    // d_mapping");
    // CUDA_CALL_AND_CHECK(cudaMemset(d_coeff, 0,
    // m * sizeof(f_t)), "cudaMemset d_coeff");
    //
    // int block_size = 256;
    // int grid_size = (m + block_size - 1) /
    // block_size;
    // initialize_steepest_edge_norms_find_row_degree_kernel<<<grid_size,
    // block_size>>>(
    //     m, d_basic_list, lp.A.d_col_ptr,
    //     lp.A.d_row_ind, lp.A.d_values,
    //     d_row_degree, d_mapping, d_coeff);
    // CUDA_CALL_AND_CHECK(cudaGetLastError(),
    //                     "initialize_steepest_edge_norms_find_row_degree_kernel");

    // We want to compute B^T delta_y_i = -e_i
    // If there is a column u of B^T such that
    // B^T(:, u) = alpha * e_i than the solve
    // delta_y_i = -1/alpha * e_u So we need to
    // find columns of B^T (or rows of B) with
    // only a single non-zero entry
    f_t start_singleton_rows = tic();
    std::vector<i_t> row_degree(m, 0);
    std::vector<i_t> mapping(m, -1);
    std::vector<f_t> coeff(m, 0.0);
    std::vector<i_t> basic_list(m);
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(basic_list.data(), d_basic_list, m * sizeof(i_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy d_basic_list to basic_list");

    for (i_t k = 0; k < m; ++k) {
        const i_t j = basic_list[k];
        const i_t col_start = lp.A.col_start[j];
        const i_t col_end = lp.A.col_start[j + 1];
        for (i_t p = col_start; p < col_end; ++p) {
            const i_t i = lp.A.i[p];
            row_degree[i]++;
            // column j of A is column k of B
            mapping[k] = i;
            coeff[k] = lp.A.x[p];
        }
    }
    i_t num_singleton_rows = 0;
    for (i_t i = 0; i < m; ++i) {
        if (row_degree[i] == 1) {
            num_singleton_rows++;
        }
    }

    if (num_singleton_rows > 0) {
        settings.log.printf("Found %d singleton rows for "
                            "steepest edge norms in %.2fs\n",
                            num_singleton_rows, toc(start_singleton_rows));
    }

    f_t last_log = tic();
    for (i_t k = 0; k < m; ++k) {
        sparse_vector_t<i_t, f_t> sparse_ei(m, 1);
        sparse_ei.x[0] = -1.0;
        sparse_ei.i[0] = k;
        const i_t j = basic_list[k];
        f_t init = -1.0;
        if (row_degree[mapping[k]] == 1) {
            const i_t u = mapping[k];
            const f_t alpha = coeff[k];
            // dy[u] = -1.0 / alpha;
            f_t my_init = 1.0 / (alpha * alpha);
            init = my_init;

        } else {
            sparse_vector_t<i_t, f_t> sparse_dy(m, 0);
            //   ft.b_transpose_solve(sparse_ei,
            //   sparse_dy);
            phase2_cu::sparse_pinv_solve_host_sparse_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                                         d_B_pinv_row_ind, d_B_pinv_values,
                                                         nz_B_pinv, sparse_ei, sparse_dy, true);
            f_t my_init = 0.0;
            for (i_t p = 0; p < (i_t) (sparse_dy.x.size()); ++p) {
                my_init += sparse_dy.x[p] * sparse_dy.x[p];
            }
            init = my_init;
        }
        // ei[k]          = 0.0;
        // init = vector_norm2_squared<i_t,
        // f_t>(dy);
        assert(init > 0);
        delta_y_steepest_edge[j] = init;

        f_t now = toc(start_time);
        f_t time_since_log = toc(last_log);
        if (time_since_log > 10) {
            last_log = tic();
            settings.log.printf("Initialized %d of %d steepest "
                                "edge norms in %.2fs\n",
                                k, m, now);
        }
        if (toc(start_time) > settings.time_limit) {
            return -1;
        }
        if (settings.concurrent_halt != nullptr &&
            settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
            return -1;
        }
    }
    return 0;
}

template <typename i_t, typename f_t> __global__ void negate_vector_kernel(i_t m, f_t *d_v) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;
    d_v[idx] = -d_v[idx];
}

template <typename i_t, typename f_t>
i_t update_steepest_edge_norms(
    const simplex_solver_settings_t<i_t, f_t> &settings, const std::vector<i_t> &basic_list,
    cusparseHandle_t &cusparse_handle, i_t *d_B_pinv_col_ptr, i_t *d_B_pinv_row_ind,
    f_t *d_B_pinv_values, i_t nz_B_pinv, i_t direction, const i_t *d_delta_y_sparse_indices,
    const f_t *d_delta_y_sparse_values, const i_t nz_delta_y_sparse, f_t dy_norm_squared,
    const sparse_vector_t<i_t, f_t> &scaled_delta_xB, i_t basic_leaving_index, i_t entering_index,
    std::vector<f_t> &v, std::vector<f_t> &delta_y_steepest_edge, i_t m) {
    // B^T delta_y = - direction *
    // e_basic_leaving_index We want B v =  -
    // B^{-T} e_basic_leaving_index
    //   ft.b_solve(delta_y_sparse, v_sparse);
    f_t *d_v;
    CUDA_CALL_AND_CHECK(cudaMalloc((void **) &d_v, m * sizeof(f_t)), "cudaMalloc d_v");
    phase2_cu::sparse_pinv_solve_gpu_sparse_rhs(
        cusparse_handle, m, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
        d_delta_y_sparse_indices, d_delta_y_sparse_values, nz_delta_y_sparse, d_v, false);
    if (direction == -1) {
        // v <- -v
        int block_size = 256;
        int grid_size = (m + block_size - 1) / block_size;
        negate_vector_kernel<<<grid_size, block_size>>>(m, d_v);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "negate_vector_kernel for v");
    }
    CUDA_CALL_AND_CHECK(cudaMemcpy(v.data(), d_v, m * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy d_v to v");
    CUDA_CALL_AND_CHECK(cudaFree(d_v), "cudaFree d_v");

    sparse_vector_t<i_t, f_t> v_sparse(m, 0);
    v_sparse.i.clear();
    v_sparse.x.clear();
    for (i_t i = 0; i < m; ++i) {
        if (std::abs(v[i]) > 1e-12) {
            v_sparse.i.push_back(i);
            v_sparse.x.push_back(v[i]);
        }
    }

    const i_t leaving_index = basic_list[basic_leaving_index];
    const f_t prev_dy_norm_squared = delta_y_steepest_edge[leaving_index];
#ifdef STEEPEST_EDGE_DEBUG
    const f_t err = std::abs(dy_norm_squared - prev_dy_norm_squared) / (1.0 + dy_norm_squared);
    if (err > 1e-3) {
        settings.log.printf("i %d j %d leaving norm error %e "
                            "computed %e previous estimate %e\n",
                            basic_leaving_index, leaving_index, err, dy_norm_squared,
                            prev_dy_norm_squared);
    }
#endif

    // B*w = A(:, leaving_index)
    // B*scaled_delta_xB = -A(:, leaving_index) so
    // w = -scaled_delta_xB
    const f_t wr = -scaled_delta_xB.find_coefficient(basic_leaving_index);
    if (wr == 0) {
        return -1;
    }
    const f_t omegar = dy_norm_squared / (wr * wr);
    const i_t scaled_delta_xB_nz = scaled_delta_xB.i.size();
    for (i_t h = 0; h < scaled_delta_xB_nz; ++h) {
        const i_t k = scaled_delta_xB.i[h];
        const i_t j = basic_list[k];
        if (k == basic_leaving_index) {
            const f_t w_squared = scaled_delta_xB.x[h] * scaled_delta_xB.x[h];
            delta_y_steepest_edge[j] = (1.0 / w_squared) * dy_norm_squared;
        } else {
            const f_t wk = -scaled_delta_xB.x[h];
            f_t new_val = delta_y_steepest_edge[j] + wk * (2.0 * v[k] / wr + wk * omegar);
            new_val = std::max(new_val, 1e-4);
#ifdef STEEPEST_EDGE_DEBUG
            if (!(new_val >= 0)) {
                settings.log.printf("new val %e\n", new_val);
                settings.log.printf("k %d j %d norm old %e wk %e "
                                    "vk %e wr %e omegar %e\n",
                                    k, j, delta_y_steepest_edge[j], wk, v[k], wr, omegar);
            }
#endif
            assert(new_val >= 0.0);
            delta_y_steepest_edge[j] = new_val;
        }
    }

    const i_t v_nz = v_sparse.i.size();
    for (i_t k = 0; k < v_nz; ++k) {
        v[v_sparse.i[k]] = 0.0;
    }

    return 0;
}
// Compute steepest edge info for entering
// variable
template <typename i_t, typename f_t>
i_t compute_steepest_edge_norm_entering(const simplex_solver_settings_t<i_t, f_t> &settings, i_t m,
                                        cusparseHandle_t &cusparse_handle, i_t *d_B_pinv_col_ptr,
                                        i_t *d_B_pinv_row_ind, f_t *d_B_pinv_values, i_t nz_B_pinv,
                                        i_t basic_leaving_index, i_t entering_index,
                                        std::vector<f_t> &steepest_edge_norms) {
    sparse_vector_t<i_t, f_t> es_sparse(m, 1);
    es_sparse.i[0] = basic_leaving_index;
    es_sparse.x[0] = -1.0;
    sparse_vector_t<i_t, f_t> delta_ys_sparse(m, 0);
    //   // ft.b_transpose_solve(es_sparse,
    //   delta_ys_sparse);
    phase2_cu::sparse_pinv_solve_host_sparse_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                                 d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                                                 es_sparse, delta_ys_sparse, true);
    steepest_edge_norms[entering_index] = delta_ys_sparse.norm2_squared();

#ifdef STEEPEST_EDGE_DEBUG
    settings.log.printf("Steepest edge norm %e for entering j %d "
                        "at i %d\n",
                        steepest_edge_norms[entering_index], entering_index, basic_leaving_index);
#endif
    return 0;
}

template <typename i_t, typename f_t>
__global__ void compute_primal_variables_nonbasiclist_kernel(
    const i_t m, const i_t n, const i_t *d_A_col_ptr, const i_t *d_A_row_ind, const f_t *d_A_values,
    const i_t *d_nonbasic_list, const f_t *d_x, const f_t tight_tol, f_t *rhs) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n - m)
        return;
    const i_t j = d_nonbasic_list[idx];
    const i_t col_start = d_A_col_ptr[j];
    const i_t col_end = d_A_col_ptr[j + 1];
    const f_t xj = d_x[j];
    if (std::abs(xj) < tight_tol * 10)
        return;
    for (i_t p = col_start; p < col_end; ++p) {
        atomicAdd(&rhs[d_A_row_ind[p]], -xj * d_A_values[p]);
    }
}

template <typename i_t, typename f_t>
__global__ void compute_primal_variables_basiclist_kernel(i_t m, const i_t *d_basic_list,
                                                          const f_t *d_xB, f_t *d_x) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = d_basic_list[idx]; // Column index in
                                     // original matrix
    d_x[col_idx] = d_xB[idx];
}

template <typename i_t, typename f_t>
void compute_primal_variables(cusparseHandle_t &cusparse_handle, const i_t *d_B_pinv_col_ptr,
                              const i_t *d_B_pinv_row_ind, const f_t *d_B_pinv_values,
                              const i_t nz_B_pinv, const f_t *lp_rhs, const i_t *d_A_col_ptr,
                              const i_t *d_A_row_ind, const f_t *d_A_values,
                              const i_t *d_basic_list, const i_t *d_nonbasic_list, f_t tight_tol,
                              f_t *&d_x, const i_t m, const i_t n) {
    f_t *d_rhs;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs, m * sizeof(f_t)), "cudaMalloc d_rhs");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_rhs, lp_rhs, m * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy lp_rhs");
    // rhs = b - sum_{j : x_j = l_j} A(:, j) *
    // l(j)
    //         - sum_{j : x_j = u_j} A(:, j) *
    //         u(j)
    int block_size = 256;
    int grid_size_nonbasic = ((n - m) + block_size - 1) / block_size;
    compute_primal_variables_nonbasiclist_kernel<<<grid_size_nonbasic, block_size>>>(
        m, n, d_A_col_ptr, d_A_row_ind, d_A_values, d_nonbasic_list, d_x, tight_tol, d_rhs);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_primal_variables_nonbasiclist_"
                                            "kernel");

    f_t *d_xB;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_xB, m * sizeof(f_t)), "cudaMalloc d_xB");
    //   ft.b_solve(rhs, xB);
    phase2_cu::sparse_pinv_solve_gpu_dense_rhs<i_t, f_t>(cusparse_handle, m, d_B_pinv_col_ptr,
                                                         d_B_pinv_row_ind, d_B_pinv_values,
                                                         nz_B_pinv, d_rhs, d_xB, false);

    int grid_size_basic = (m + block_size - 1) / block_size;
    compute_primal_variables_basiclist_kernel<<<grid_size_basic, block_size>>>(m, d_basic_list,
                                                                               d_xB, d_x);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_primal_variables_basiclist_"
                                            "kernel");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs), "cudaFree d_rhs");
    CUDA_CALL_AND_CHECK(cudaFree(d_xB), "cudaFree d_xB");
}
template <typename i_t, typename f_t>
__global__ void update_dual_variable_single_kernel(i_t index, f_t step_length, f_t *d_var,
                                                   const f_t *d_delta_var) {
    d_var[index] += step_length * d_delta_var[index];
}

template <typename i_t, typename f_t>
__global__ void update_dual_variable_kernel(i_t nz_delta_var, const i_t *d_delta_var_indices,
                                            const f_t *d_delta_var_values, f_t step_length,
                                            f_t *d_var) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nz_delta_var) {
        const i_t i = d_delta_var_indices[idx];
        const f_t delta_val = d_delta_var_values[idx];
        d_var[i] += step_length * delta_val;
    }
}

template <typename i_t, typename f_t>
void update_dual_variables(const i_t *d_delta_y_indices, const f_t *d_delta_y_values,
                           i_t nz_delta_y, const i_t *d_delta_z_indices,
                           const f_t *d_delta_z_values, i_t nz_delta_z, f_t step_length,
                           i_t leaving_index, f_t *&d_y, f_t *&d_z) {
    // Update dual variables
    // y <- y + steplength * delta_y
    const i_t block_size = 256;
    i_t grid_size = (nz_delta_y + block_size - 1) / block_size;
    update_dual_variable_kernel<<<grid_size, block_size>>>(nz_delta_y, d_delta_y_indices,
                                                           d_delta_y_values, step_length, d_y);

    CUDA_CALL_AND_CHECK(cudaGetLastError(), "update_dual_variable_kernel");

    grid_size = (nz_delta_z + block_size - 1) / block_size;
    update_dual_variable_kernel<<<grid_size, block_size>>>(nz_delta_z, d_delta_z_indices,
                                                           d_delta_z_values, step_length, d_z);
    // z <- z + steplength * delta_z
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "update_dual_variable_kernel");

    // d_z[leaving_index] += step_length * d_delta_z[leaving_index]; TODO: figure out why this is
    // done twice??
    update_dual_variable_single_kernel<<<1, 1>>>(leaving_index, step_length, d_z, d_delta_z_values);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "update_dual_variable_single_kernel");
}

template <typename i_t, typename f_t>
i_t compute_primal_solution_from_basis(const lp_problem_t<i_t, f_t> &lp, f_t *d_lp_rhs,
                                       const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
                                       const f_t *d_A_values, cusparseHandle_t &cusparse_handle,
                                       i_t *d_B_pinv_col_ptr, i_t *d_B_pinv_row_ind,
                                       f_t *d_B_pinv_values, i_t nz_B_pinv, const i_t *d_basic_list,
                                       const i_t *d_nonbasic_list,
                                       const std::vector<variable_status_t> &vstatus,
                                       std::vector<f_t> &x, f_t *d_x, const i_t m, const i_t n) {
    std::vector<i_t> nonbasic_list(n - m);
    CUDA_CALL_AND_CHECK(cudaMemcpy(nonbasic_list.data(), d_nonbasic_list, (n - m) * sizeof(i_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy d_nonbasic_list to nonbasic_list");
    for (i_t k = 0; k < n - m; ++k) {
        const i_t j = nonbasic_list[k];
        if (vstatus[j] == variable_status_t::NONBASIC_LOWER ||
            vstatus[j] == variable_status_t::NONBASIC_FIXED) {
            x[j] = lp.lower[j];
        } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
            x[j] = lp.upper[j];
        } else if (vstatus[j] == variable_status_t::NONBASIC_FREE) {
            x[j] = 0.0;
        }
    }

    CUDA_CALL_AND_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy x to d_x");

    // rhs = b - sum_{j : x_j = l_j} A(:, j) l(j)
    // - sum_{j : x_j = u_j} A(:, j) * u(j)
    // for (i_t k = 0; k < n - m; ++k) {
    //     const i_t j = nonbasic_list[k];
    //     const i_t col_start = lp.A.col_start[j];
    //     const i_t col_end = lp.A.col_start[j + 1];
    //     const f_t xj = x[j];
    //     for (i_t p = col_start; p < col_end; ++p) {
    //         rhs[lp.A.i[p]] -= xj * lp.A.x[p];
    //     }
    // }
    i_t block_size = 256;
    int grid_size = ((n - m) + block_size - 1) / block_size;
    compute_primal_variables_nonbasiclist_kernel<<<grid_size, block_size>>>(
        m, n, d_A_col_ptr, d_A_row_ind, d_A_values, d_nonbasic_list, d_x, 0.0, d_lp_rhs);

    std::vector<f_t> xB(m);
    f_t *d_xB;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_xB, m * sizeof(f_t)), "cudaMalloc d_xB");

    //   ft.b_solve(rhs, xB);
    phase2_cu::sparse_pinv_solve_gpu_dense_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                               d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                                               d_lp_rhs, d_xB, false);

    grid_size = (m + block_size - 1) / block_size;
    compute_primal_variables_basiclist_kernel<<<grid_size, block_size>>>(m, d_basic_list, d_xB,
                                                                         d_x);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_primal_variables_basiclist_kernel");
    // for (i_t k = 0; k < m; ++k) {
    //     const i_t j = basic_list[k];
    //     x[j] = xB[k];
    // }
    return 0;
}

template <typename i_t, typename f_t>
void compute_dual_solution_from_basis(const f_t *d_lp_objective, const i_t *d_A_col_ptr,
                                      const i_t *d_A_row_ind, const f_t *d_A_values,
                                      cusparseHandle_t &cusparse_handle, i_t *d_B_pinv_col_ptr,
                                      i_t *d_B_pinv_row_ind, f_t *d_B_pinv_values, i_t nz_B_pinv,
                                      const i_t *d_basic_list, const i_t *d_nonbasic_list,
                                      f_t *&d_y, f_t *&d_z, i_t m, i_t n) {
    f_t *d_cB;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_cB, m * sizeof(f_t)), "cudaMalloc d_cB");

    i_t block_size = 256;
    int grid_size_basic = (m + block_size - 1) / block_size;
    construct_c_basic_kernel<<<grid_size_basic, block_size>>>(m, d_basic_list, d_lp_objective,
                                                              d_cB);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "construct_c_basic_kernel");

    phase2_cu::sparse_pinv_solve_gpu_dense_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                               d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, d_cB,
                                               d_y, true);

    // We want A'y + z = c
    // A = [ B N ]
    // B' y = c_B, z_B = 0
    // N' y + z_N = c_N

    compute_reduced_costs<i_t, f_t>(d_lp_objective, d_A_col_ptr, d_A_row_ind, d_A_values, d_y,
                                    d_basic_list, d_nonbasic_list, d_z, m, n);
    // zN = cN - N'*y
    // for (i_t k = 0; k < n - m; k++) {
    //     const i_t j = nonbasic_list[k];
    //     // z_j <- c_j
    //     z[j] = lp.objective[j];

    //     // z_j <- z_j - A(:, j)'*y
    //     const i_t col_start = lp.A.col_start[j];
    //     const i_t col_end = lp.A.col_start[j + 1];
    //     f_t dot = 0.0;
    //     for (i_t p = col_start; p < col_end; ++p) {
    //         dot += lp.A.x[p] * y[lp.A.i[p]];
    //     }
    //     z[j] -= dot;
    // }
    // // zB = 0
    // for (i_t k = 0; k < m; ++k) {
    //     z[basic_list[k]] = 0.0;
    // }
}
template <typename i_t, typename f_t>
void prepare_optimality(const lp_problem_t<i_t, f_t> &lp,
                        const simplex_solver_settings_t<i_t, f_t> &settings,
                        const f_t *d_lp_objective, const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
                        const f_t *d_A_values, cusparseHandle_t &cusparse_handle,
                        i_t *d_B_pinv_col_ptr, i_t *d_B_pinv_row_ind, f_t *d_B_pinv_values,
                        i_t nz_B_pinv, const std::vector<f_t> &objective,
                        const std::vector<i_t> &basic_list, const i_t *d_basic_list,
                        const std::vector<i_t> &nonbasic_list, const i_t *d_nonbasic_list,
                        const std::vector<variable_status_t> &vstatus, int phase, f_t start_time,
                        f_t max_val, i_t iter, const std::vector<f_t> &x, std::vector<f_t> &y,
                        std::vector<f_t> &z, lp_solution_t<i_t, f_t> &sol) {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;

    sol.objective = compute_objective(lp, sol.x);
    sol.user_objective = compute_user_objective(lp, sol.objective);
    f_t perturbation = phase2::amount_of_perturbation(lp, objective);
    if (perturbation > 1e-6 && phase == 2) {
        // Try to remove perturbation
        std::vector<f_t> unperturbed_y(m);
        std::vector<f_t> unperturbed_z(n);
        f_t *d_unperturbed_y;
        f_t *d_unperturbed_z;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_y, m * sizeof(f_t)),
                            "cudaMalloc unperturbed_y");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_z, n * sizeof(f_t)),
                            "cudaMalloc unperturbed_z");
        phase2_cu::compute_dual_solution_from_basis(
            d_lp_objective, d_A_col_ptr, d_A_row_ind, d_A_values, cusparse_handle, d_B_pinv_col_ptr,
            d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, d_basic_list, d_nonbasic_list,
            d_unperturbed_y, d_unperturbed_z, m, n);
        CUDA_CALL_AND_CHECK(cudaMemcpy(unperturbed_y.data(), d_unperturbed_y, m * sizeof(f_t),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy unperturbed_y to host");
        CUDA_CALL_AND_CHECK(cudaMemcpy(unperturbed_z.data(), d_unperturbed_z, n * sizeof(f_t),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy unperturbed_z to host");
        CUDA_CALL_AND_CHECK(cudaFree(d_unperturbed_y), "cudaFree d_unperturbed_y");
        CUDA_CALL_AND_CHECK(cudaFree(d_unperturbed_z), "cudaFree d_unperturbed_z");
        {
            const f_t dual_infeas = phase2::dual_infeasibility(
                lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
            if (dual_infeas <= settings.dual_tol) {
                settings.log.printf("Removed perturbation of "
                                    "%.2e.\n",
                                    perturbation);
                z = unperturbed_z;
                y = unperturbed_y;
                perturbation = 0.0;
            } else {
                settings.log.printf("Failed to remove "
                                    "perturbation of %.2e.\n",
                                    perturbation);
            }
        }
    }

    sol.l2_primal_residual = phase2::l2_primal_residual(lp, sol);
    sol.l2_dual_residual = phase2::l2_dual_residual(lp, sol);
    const f_t dual_infeas = phase2::dual_infeasibility(lp, settings, vstatus, z, 0.0, 0.0);
    const f_t primal_infeas = phase2::primal_infeasibility(lp, settings, vstatus, x);
    if (phase == 1 && iter > 0) {
        settings.log.printf("Dual phase I complete. Iterations "
                            "%d. Time %.2f\n",
                            iter, toc(start_time));
    }
    if (phase == 2) {
        if (!settings.inside_mip) {
            settings.log.printf("\n");
            settings.log.printf("Optimal solution found in %d "
                                "iterations and %.2fs\n",
                                iter, toc(start_time));
            settings.log.printf("Objective %+.8e\n", sol.user_objective);
            settings.log.printf("\n");
            settings.log.printf("Primal infeasibility (abs): "
                                "%.2e\n",
                                primal_infeas);
            settings.log.printf("Dual infeasibility (abs):   "
                                "%.2e\n",
                                dual_infeas);
            settings.log.printf("Perturbation:               "
                                "%.2e\n",
                                perturbation);
        } else {
            settings.log.printf("\n");
            settings.log.printf("Root relaxation solution found "
                                "in %d iterations and %.2fs\n",
                                iter, toc(start_time));
            settings.log.printf("Root relaxation objective "
                                "%+.8e\n",
                                sol.user_objective);
            settings.log.printf("\n");
        }
    }
}

template <typename i_t, typename f_t>
void adjust_for_flips(cusparseHandle_t &cusparse_handle, i_t *d_B_pinv_col_ptr,
                      i_t *d_B_pinv_row_ind, const f_t *d_B_pinv_values, i_t nz_B_pinv,
                      const std::vector<i_t> &basic_list, const std::vector<i_t> &delta_z_indices,
                      std::vector<i_t> &atilde_index, std::vector<f_t> &atilde,
                      std::vector<i_t> &atilde_mark, sparse_vector_t<i_t, f_t> &delta_xB_0_sparse,
                      std::vector<f_t> &delta_x_flip, std::vector<f_t> &x) {
    const i_t m = basic_list.size();
    const i_t atilde_nz = atilde_index.size();
    // B*delta_xB_0 = atilde
    sparse_vector_t<i_t, f_t> atilde_sparse(m, atilde_nz);
    for (i_t k = 0; k < atilde_nz; ++k) {
        atilde_sparse.i[k] = atilde_index[k];
        atilde_sparse.x[k] = atilde[atilde_index[k]];
    }
    //   ft.b_solve(atilde_sparse,
    //   delta_xB_0_sparse);
    phase2_cu::sparse_pinv_solve_host_sparse_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                                 d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                                                 atilde_sparse, delta_xB_0_sparse, false);
    const i_t delta_xB_0_nz = delta_xB_0_sparse.i.size();
    for (i_t k = 0; k < delta_xB_0_nz; ++k) {
        const i_t j = basic_list[delta_xB_0_sparse.i[k]];
        x[j] += delta_xB_0_sparse.x[k];
    }

    for (i_t j : delta_z_indices) {
        x[j] += delta_x_flip[j];
        delta_x_flip[j] = 0.0;
    }

    // Clear atilde
    for (i_t k = 0; k < (i_t) (atilde_index.size()); ++k) {
        atilde[atilde_index[k]] = 0.0;
    }
    // Clear atilde_mark
    for (i_t k = 0; k < (i_t) (atilde_mark.size()); ++k) {
        atilde_mark[k] = 0;
    }
    atilde_index.clear();
}

template <typename i_t, typename f_t>
__global__ void sparse_vector_squared_norm_kernel(i_t nz, const i_t *d_indices, const f_t *d_values,
                                                  f_t *d_partial_sums) {
    extern __shared__ f_t sdata[];
    i_t tid = threadIdx.x;
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    f_t my_sum = 0.0;
    if (idx < nz) {
        f_t val = d_values[idx];
        my_sum = val * val;
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Reduce within block
    for (i_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

template <typename i_t, typename f_t>
f_t sparse_vector_squared_norm_gpu(i_t nz, const i_t *d_indices, const f_t *d_values) {
    const i_t block_size = 256;
    i_t grid_size = (nz + block_size - 1) / block_size;
    f_t *d_partial_sums;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_partial_sums, grid_size * sizeof(f_t)),
                        "cudaMalloc d_partial_sums");
    sparse_vector_squared_norm_kernel<<<grid_size, block_size, block_size * sizeof(f_t)>>>(
        nz, d_indices, d_values, d_partial_sums);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "sparse_vector_squared_norm_kernel");
    std::vector<f_t> h_partial_sums(grid_size);
    CUDA_CALL_AND_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, grid_size * sizeof(f_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy d_partial_sums to host");
    CUDA_CALL_AND_CHECK(cudaFree(d_partial_sums), "cudaFree d_partial_sums");
    f_t total_sum = 0.0;
    for (i_t i = 0; i < grid_size; ++i) {
        total_sum += h_partial_sums[i];
    }
    return total_sum;
}
// TODO: end kernelify
} // namespace phase2_cu

template <typename i_t, typename f_t>
struct DualSimplexWorkspace {
    // --- Handles ---
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cusparseHandle_t cusparse_pinv_handle;
    cudssHandle_t cudss_handle;
    cudssConfig_t cudss_config;
    cusparseSpMatDescr_t B_pinv_cusparse = nullptr;

    // --- Matrix A (Static - loaded once) ---
    i_t *d_A_col_ptr = nullptr;
    i_t *d_A_row_ind = nullptr;
    f_t *d_A_values = nullptr;

    // --- Eta Vectors & Temp ---
    f_t *eta_b_old, *eta_b_new, *eta_v, *eta_c, *eta_d;
    f_t *d_temp_vector_m;
    
    // --- Basis & Pinv Buffers ---
    i_t *d_B_col_ptr, *d_B_row_ind;
    f_t *d_B_values;

    i_t *d_B_pinv_col_ptr, *d_B_pinv_row_ind;
    f_t *d_B_pinv_values;
    
    // Pinv Update Buffers
    i_t *d_B_pinv_col_ptr_buffer, *d_B_pinv_row_ind_buffer;
    f_t *d_B_pinv_values_buffer;

    // --- Lists & Vectors ---
    i_t *d_basic_list, *d_nonbasic_list;
    f_t *d_objective, *d_c_basic, *d_lp_rhs;
    f_t *d_x, *d_y, *d_z;

    // --- Scratch Buffers (To replace mallocs INSIDE the loop) ---
    // These replace the d_delta_y/z mallocs inside the while loop
    i_t *d_scratch_indices_m, *d_scratch_indices_n;
    f_t *d_scratch_values_m, *d_scratch_values_n;

    // Extra scratch for unperturbed x calculation
    f_t *d_temp_x;
    

    // TODO don't know if still necessary
    // --- State Check ---
    bool is_initialized = false;
};


template <typename i_t, typename f_t>
void initialize_workspace(DualSimplexWorkspace<i_t, f_t> &ws, 
                          const lp_problem_t<i_t, f_t> &lp, 
                          const simplex_solver_settings_t<i_t, f_t> &settings) {
    
    i_t m = lp.num_rows;
    i_t n = lp.num_cols;

    // 1. Create Handles
    CUBLAS_CALL_AND_CHECK(cublasCreate(&ws.cublas_handle), "cublasCreate");
    CUSPARSE_CALL_AND_CHECK(cusparseCreate(&ws.cusparse_handle), "cusparseCreate");
    CUSPARSE_CALL_AND_CHECK(cusparseCreate(&ws.cusparse_pinv_handle), "cusparseCreate pinv");
    CUDSS_CALL_AND_CHECK(cudssCreate(&ws.cudss_handle), "cudssCreateHandle B");
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&ws.cudss_config), "cudssCreateConfig B");
    
    // CUDSS Config
    i_t use_matching = 1;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(ws.cudss_config, CUDSS_CONFIG_USE_MATCHING, &use_matching, sizeof(i_t)),
        "cudssConfigSetParameter ENABLE_MATCHINGS B");
    cudssAlgType_t matching_alg = CUDSS_ALG_5;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(ws.cudss_config, CUDSS_CONFIG_MATCHING_ALG, &matching_alg,
                                        sizeof(cudssAlgType_t)),
                         "cudssConfigSetParameter MATCHING_ALG_TYPE B");

    // TODO jujupieper make sure this makes sense here
    // Analyze matrix A
    problem_analyzer_t<i_t, f_t> analyzer(lp, settings);
    analyzer.analyze();
    analyzer.display_analysis();

    // 2. Move Static Matrix A (Only done ONCE now!)
    phase2_cu::move_A_to_device(lp.A, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values);

    // TODO jujupieper IMPORTANT DO THESE CHANGE AT ALL?????
    // 3. Allocate Vectors (m or n size)
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.eta_b_old, m * sizeof(f_t)), "cudaMalloc eta_b_old");
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.eta_b_new, m * sizeof(f_t)), "cudaMalloc eta_b_new");
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.eta_v, m * sizeof(f_t)), "cudaMalloc eta_v");
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.eta_c, m * sizeof(f_t)), "cudaMalloc eta_c");
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.eta_d, m * sizeof(f_t)), "cudaMalloc eta_d");

    // cudaMalloc(&ws.d_temp_vector_m, m * sizeof(f_t));
    // TODO i did not change this one yet cause i am worried i might break something
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.d_temp_vector_m, m * sizeof(f_t)),
                        "cudaMalloc d_temp_vector_m");
    
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    // cudaMalloc(&ws.d_basic_list, m * sizeof(i_t));
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.d_nonbasic_list, (n - m) * sizeof(i_t)),
                        "cudaMalloc d_nonbasic_list");
    // cudaMalloc(&ws.d_nonbasic_list, (n - m) * sizeof(i_t));
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.d_objective, n * sizeof(f_t)), "cudaMalloc d_objective");
    // cudaMalloc(&ws.d_objective, n * sizeof(f_t));
    
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.d_c_basic, m * sizeof(f_t)), "cudaMalloc d_c_basic");
    // cudaMalloc(&ws.d_c_basic, m * sizeof(f_t));
    cudaMalloc(&ws.d_lp_rhs, m * sizeof(f_t));
    cudaMalloc(&ws.d_y, m * sizeof(f_t));
    
    cudaMalloc(&ws.d_x, n * sizeof(f_t));
    cudaMalloc(&ws.d_z, n * sizeof(f_t));

    // 4. Allocate Matrix B and Pinv containers
    // Note: We allocate assuming Worst Case or use the multipliers from settings
    // cudaMalloc(&ws.d_B_col_ptr, (m + 1) * sizeof(i_t));
    CUDA_CALL_AND_CHECK(cudaMalloc(&ws.d_B_col_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_col_ptr");
    // Assuming B is sparse but could be full in worst case, or rely on internal logic to resize if needed.
    // For safety in persistent mode, explicit sizing logic from your original code:
    // Original code allocated these inside compute_inverse usually, or here. 
    // We will allocate max anticipated size.
    // WARNING: If B grows beyond this, you need realloc logic inside the solver.
    // For now, allocating generous buffers:
    cudaMalloc(&ws.d_B_row_ind, m * 100 * sizeof(i_t)); // Heuristic
    cudaMalloc(&ws.d_B_values, m * 100 * sizeof(f_t));

    cudaMalloc(&ws.d_B_pinv_col_ptr, (m + 1) * sizeof(i_t));
    // Heuristic for Pinv size:
    i_t max_pinv_nz = m * m * 0.1; // 10% density? Adjust based on problem.
    cudaMalloc(&ws.d_B_pinv_row_ind, max_pinv_nz * sizeof(i_t));
    cudaMalloc(&ws.d_B_pinv_values, max_pinv_nz * sizeof(f_t));

    // Pinv Buffers
    i_t buffer_size = static_cast<i_t>(settings.pinv_buffer_size_multiplier * max_pinv_nz);
    cudaMalloc(&ws.d_B_pinv_col_ptr_buffer, (m + 1) * sizeof(i_t));
    cudaMalloc(&ws.d_B_pinv_row_ind_buffer, buffer_size * sizeof(i_t));
    cudaMalloc(&ws.d_B_pinv_values_buffer, buffer_size * sizeof(f_t));

    // 5. Allocate Scratchpad (Reuse these instead of malloc inside loop)
    // TODO jujupieper don't do this for now this is confusing you
    // cudaMalloc(&ws.d_scratch_indices_m, m * sizeof(i_t));
    // cudaMalloc(&ws.d_scratch_values_m, m * sizeof(f_t));
    // cudaMalloc(&ws.d_scratch_indices_n, n * sizeof(i_t));
    // cudaMalloc(&ws.d_scratch_values_n, n * sizeof(f_t));

    // Initialize constant data
    // TODO jujupieper IMPORTANT --> Doesn't this change?
    // cudaMemcpy(ws.d_objective, lp.objective.data(), n * sizeof(f_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.d_lp_rhs, lp.rhs.data(), m * sizeof(f_t), cudaMemcpyHostToDevice);

    ws.is_initialized = true;
}

template <typename i_t, typename f_t>
void free_workspace(DualSimplexWorkspace<i_t, f_t> &ws) {
    if (!ws.is_initialized) return;

    // cublasDestroy(ws.cublas_handle);
    // cusparseDestroy(ws.cusparse_handle);
    // if(ws.B_pinv_cusparse) cusparseDestroySpMat(ws.B_pinv_cusparse);
    // cusparseDestroy(ws.cusparse_pinv_handle);
    // cudssConfigDestroy(ws.cudss_config);
    // cudssDestroy(ws.cudss_handle);
    CUBLAS_CALL_AND_CHECK(cublasDestroy(ws.cublas_handle), "cublasDestroy");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroy(ws.cusparse_handle), "cusparseDestroy");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(ws.B_pinv_cusparse),
                            "cusparseDestroyMatDescr B_pinv");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroy(ws.cusparse_pinv_handle), "cusparseDestroy Pinv");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(ws.cudss_config), "cudssConfigDestroy B");
    CUDSS_CALL_AND_CHECK(cudssDestroy(ws.cudss_handle), "cudssDestroyHandle B");

    cudaFree(ws.d_A_col_ptr); cudaFree(ws.d_A_row_ind); cudaFree(ws.d_A_values);
    cudaFree(ws.eta_b_old); cudaFree(ws.eta_b_new); cudaFree(ws.eta_v);
    cudaFree(ws.eta_c); cudaFree(ws.eta_d); cudaFree(ws.d_temp_vector_m);
    cudaFree(ws.d_basic_list); cudaFree(ws.d_nonbasic_list);
    cudaFree(ws.d_objective); cudaFree(ws.d_c_basic); cudaFree(ws.d_lp_rhs);
    cudaFree(ws.d_x); cudaFree(ws.d_y); cudaFree(ws.d_z);
    
    // cudaFree(ws.d_B_col_ptr); 
    CUDA_CALL_AND_CHECK(cudaFree(ws.d_B_col_ptr), "cudaFree d_B_col_ptr");
    cudaFree(ws.d_B_row_ind); cudaFree(ws.d_B_values);
    cudaFree(ws.d_B_pinv_col_ptr); cudaFree(ws.d_B_pinv_row_ind); cudaFree(ws.d_B_pinv_values);
    cudaFree(ws.d_B_pinv_col_ptr_buffer); cudaFree(ws.d_B_pinv_row_ind_buffer); cudaFree(ws.d_B_pinv_values_buffer);
    
    // cudaFree(ws.d_scratch_indices_m); cudaFree(ws.d_scratch_values_m);
    // cudaFree(ws.d_scratch_indices_n); cudaFree(ws.d_scratch_values_n);
}

struct ObjectiveRank {
    double value;
    int    rank;
};

template <typename i_t, typename f_t>
dual::status_t dual_phase2_cu_parallel_pivot(i_t phase, i_t slack_basis, f_t start_time,
                              const lp_problem_t<i_t, f_t> &lp,
                              const simplex_solver_settings_t<i_t, f_t> &settings,
                              std::vector<variable_status_t> &vstatus, lp_solution_t<i_t, f_t> &sol,
                              i_t &iter, std::vector<f_t> &delta_y_steepest_edge) {

    // NOTE: Assumes consistent solutions between MPI ranks
    
    // --- 1. MPI & GPU SETUP ---
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign a specific GPU to this MPI rank (Round-Robin)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        int device_id = rank % num_gpus;
        cudaSetDevice(device_id);
    }


    // TODO jujupieper create the necessary mallocs
    DualSimplexWorkspace<i_t, f_t> workspace;
    initialize_workspace(workspace, lp, settings);



    // --- 2. LOCAL CONFIGURATION (The "Strategy Divergence") ---
    // We make a local copy of settings so we can modify limits and pivot rules per rank.
    // TODO jujupieper you don't need to copy this but like its fine ig
    simplex_solver_settings_t<i_t, f_t> local_settings = settings;

    // Rank 0: Default Strategy
    // Rank 1: Alternative Strategy (e.g., more aggressive steepest edge)
    if (rank == 1) {
        local_settings.steepest_edge_ratio = 0.9; 
        // TODO jujupieper need to change this for dantzig with consistent perturbation or something similar
        // local_settings.use_harris_ratio = !settings.use_harris_ratio; // Example divergence

        // TODO jujupieper also put in different limits here for iterations
    }

    // Define the "Sync Frequency" (How many pivots before we compare notes?)
    const i_t SYNC_INTERVAL = 1000; 
    const i_t global_iter_limit = settings.iteration_limit;

    dual::status_t status = dual::status_t::UNSET;
    bool global_optimal_found = false;

    // --- 3. THE OUTER LOOP (Run -> Pause -> Sync -> Repeat) ---
    while (iter < global_iter_limit && !global_optimal_found) {
        
        // A. Set the "Chunk" Limit
        // We tell the inner solver to stop after SYNC_INTERVAL steps 
        // OR if it hits the global limit.
        local_settings.iteration_limit = std::min(iter + SYNC_INTERVAL, global_iter_limit);
        settings.log.printf("Thread %d, iteration %d \n", rank, iter);

        // B. Run the Solver (The "Worker" Phase)
        // This will run until it hits local_settings.iteration_limit or finds OPTIMAL.
        // TODO jujupieper I need to create a version of dual_phase2_cu that does not redo everything here
        status = dual_phase2_cu_persistent(rank, phase, slack_basis, start_time,
                                lp,
                                local_settings,
                                vstatus, sol,
                                iter, delta_y_steepest_edge, workspace);

        // C. MPI Synchronization (The "Meeting" Phase)
        
        // C1. Check who is winning
        ObjectiveRank my_info;
        my_info.rank = rank;
        
        // If we found optimal or ran successfully, report objective.
        // If we failed/unbounded, report Infinity so we don't become the winner.
        if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT) {
             // Assuming sol.objective_value is populated. 
             // If not, calculate it: compute_perturbed_objective(lp.objective, sol.x);
             //  my_info.value = sol.objective_value; 
            //  TODO jujupieper IMPORTANT check if this can be done quicker
             cuopt::linear_programming::dual_simplex::phase2::compute_perturbed_objective(lp.objective, sol.x);
             
        } else {
             my_info.value = std::numeric_limits<double>::infinity();
        }

        ObjectiveRank winner_info;
        // Find the rank with the MINIMUM objective value
        MPI_Allreduce(&my_info, &winner_info, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        // TODO jujupieper if you want to get mulitple scores and work on those then:
        // MPI_Allgather

        // C2. Check for Global Termination
        // If the winner found an optimal solution (and returned OPTIMAL status), we are done.
        // We need to broadcast the STATUS of the winner to know if we should stop.
        int winner_status_int = (int)status;
        MPI_Bcast(&winner_status_int, 1, MPI_INT, winner_info.rank, MPI_COMM_WORLD);
        
        if ((dual::status_t)winner_status_int == dual::status_t::OPTIMAL) {
            global_optimal_found = true;
            status = dual::status_t::OPTIMAL; // Everyone exits with OPTIMAL
        }

        // C3. SYNC THE BASIS (The "Jump" Step)
        // If I am NOT the winner, I must adopt the winner's state.
        if (rank != winner_info.rank) {
             // 1. Receive the winning Basis
             // Note: MPI needs raw pointers. Ensure vstatus is a vector of ints/enums 
             // compatible with MPI_INT. If variable_status_t is 8-bit, use MPI_BYTE.
             MPI_Bcast(vstatus.data(), vstatus.size(), MPI_INT, winner_info.rank, MPI_COMM_WORLD);

             // 2. CRITICAL: Reset Weights ("Cold Start" Fix)
             // Since we have a new basis but old weights, we MUST clear this 
             // to force dual_phase2_cu to re-calculate clean weights next time.
             delta_y_steepest_edge.clear();
             
             // 3. Optional: Sync iteration count if you want logs to look consistent
             MPI_Bcast(&iter, 1, MPI_INT, winner_info.rank, MPI_COMM_WORLD);
        } else {
             // I am the winner. Broadcast my basis to the others.
             MPI_Bcast(vstatus.data(), vstatus.size(), MPI_INT, rank, MPI_COMM_WORLD);
             
             // I keep my 'delta_y_steepest_edge' because it matches my basis.
        }

        // Check for global timeout
        // (Assuming you have a way to check current time vs settings.time_limit)
        // if (toc(start_time) > settings.time_limit) return dual::status_t::TIME_LIMIT;
    }

    // free_workspace(workspace);
    return status;
}

template <typename i_t, typename f_t>
dual::status_t dual_phase2_cu_persistent(int rank, i_t phase, i_t slack_basis, f_t start_time,
                              const lp_problem_t<i_t, f_t> &lp,
                              const simplex_solver_settings_t<i_t, f_t> &settings,
                              std::vector<variable_status_t> &vstatus, lp_solution_t<i_t, f_t> &sol,
                              i_t &iter, std::vector<f_t> &delta_y_steepest_edge,
                              DualSimplexWorkspace<i_t, f_t> &ws) {
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

    settings.log.printf("Dual Simplex Phase %d on rank %d\n", phase, rank);
    std::vector<variable_status_t> vstatus_old = vstatus;
    std::vector<f_t> z_old = z;

    phase2::bound_info(lp, settings);


    // TODO jujupieper IMPORTANT this is where Yi wants me to check for the inverse update -> Not here but elsewhere
    get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
    assert(superbasic_list.size() == 0);
    assert(nonbasic_list.size() == n - m);

    // Analyze matrix A
    // problem_analyzer_t<i_t, f_t> analyzer(lp, settings);
    // analyzer.analyze();
    // analyzer.display_analysis();

    // SETUP GPU ALLOCATIONS AND HANDLES
    // Create all handles
    // cublasHandle_t cublas_handle;
    // CUBLAS_CALL_AND_CHECK(cublasCreate(&cublas_handle), "cublasCreate");
    // cusparseHandle_t cusparse_handle;
    // CUSPARSE_CALL_AND_CHECK(cusparseCreate(&cusparse_handle), "cusparseCreate");
    // cusparseHandle_t cusparse_pinv_handle;
    // CUSPARSE_CALL_AND_CHECK(cusparseCreate(&cusparse_pinv_handle), "cusparseCreate pinv");
    // cudssHandle_t cudss_handle;
    // CUDSS_CALL_AND_CHECK(cudssCreate(&cudss_handle), "cudssCreateHandle B");
    // cudssConfig_t cudss_config;
    // CUDSS_CALL_AND_CHECK(cudssConfigCreate(&cudss_config), "cudssCreateConfig B");
    // i_t use_matching = 1;
    // CUDSS_CALL_AND_CHECK(
    //     cudssConfigSet(cudss_config, CUDSS_CONFIG_USE_MATCHING, &use_matching, sizeof(i_t)),
    //     "cudssConfigSetParameter ENABLE_MATCHINGS B");
    // cudssAlgType_t matching_alg = CUDSS_ALG_5;
    // CUDSS_CALL_AND_CHECK(cudssConfigSet(cudss_config, CUDSS_CONFIG_MATCHING_ALG, &matching_alg,
    //                                     sizeof(cudssAlgType_t)),
    //                      "cudssConfigSetParameter MATCHING_ALG_TYPE B");

    // Create matrix representations
    // cusparseSpMatDescr_t B_pinv_cusparse;

    // Move A to device
    // i_t *d_A_col_ptr;
    // i_t *d_A_row_ind;
    // f_t *d_A_values;
    // phase2_cu::move_A_to_device(lp.A, d_A_col_ptr, d_A_row_ind, d_A_values);

    // Create dense vectors for eta updates
    // TODO: remove once we have dense sparse mv
    // on device
    // f_t *eta_b_old, *eta_b_new, *eta_v, *eta_c, *eta_d;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&eta_b_old, m * sizeof(f_t)), "cudaMalloc eta_b_old");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&eta_b_new, m * sizeof(f_t)), "cudaMalloc eta_b_new");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&eta_v, m * sizeof(f_t)), "cudaMalloc eta_v");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&eta_c, m * sizeof(f_t)), "cudaMalloc eta_c");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&eta_d, m * sizeof(f_t)), "cudaMalloc eta_d");

    // Compute Moore-Penrose pseudo-inverse of B
    // i_t *d_B_col_ptr;
    i_t *d_B_row_ind;
    f_t *d_B_values;
    i_t nz_B;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_col_ptr");

    i_t len_B_pinv = 0;
    i_t *d_B_pinv_col_ptr;
    i_t *d_B_pinv_row_ind;
    f_t *d_B_pinv_values;
    i_t nz_B_pinv;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_col_ptr, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_B_pinv_col_ptr");
    
    phase2_cu::compute_inverse<i_t, f_t>(
        ws.cusparse_handle, ws.cudss_handle, ws.cudss_config, m, n, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values,
        ws.d_B_col_ptr, d_B_row_ind, d_B_values, basic_list, d_B_pinv_col_ptr, d_B_pinv_row_ind,
        d_B_pinv_values, nz_B, nz_B_pinv, len_B_pinv, settings);

    // We pre-allocate buffer for eta updates
    i_t len_B_pinv_buffer = static_cast<i_t>(settings.pinv_buffer_size_multiplier * nz_B_pinv);
    i_t *d_B_pinv_col_ptr_buffer;
    i_t *d_B_pinv_row_ind_buffer;
    f_t *d_B_pinv_values_buffer;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_col_ptr_buffer, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_B_pinv_col_ptr_buffer");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_row_ind_buffer, len_B_pinv_buffer * sizeof(i_t)),
                        "cudaMalloc d_B_pinv_row_ind_buffer");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_values_buffer, len_B_pinv_buffer * sizeof(f_t)),
                        "cudaMalloc d_B_pinv_values_buffer");

    // TODO jujupieper IMPORTANT I will leave this here cause i think it is important
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&ws.B_pinv_cusparse, m, m, nz_B_pinv, d_B_pinv_col_ptr,
                                              d_B_pinv_row_ind, d_B_pinv_values, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F),
                            "cusparseCreateCsc B_pinv_cusparse");

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    f_t *d_temp_vector_m; // Temporary vector of size m on device, so we dont have to keep
                          // allocating one, never make assumptions about its contents!!!
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_temp_vector_m, m * sizeof(f_t)),
                        "cudaMalloc d_temp_vector_m");

    // i_t *d_basic_list, *d_nonbasic_list;
    // f_t *d_c_basic, *d_objective;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(ws.d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy basic_list to device");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_nonbasic_list, (n - m) * sizeof(i_t)),
    //                     "cudaMalloc d_nonbasic_list");
    CUDA_CALL_AND_CHECK(cudaMemcpy(ws.d_nonbasic_list, nonbasic_list.data(), (n - m) * sizeof(i_t),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy nonbasic_list to device");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_objective, n * sizeof(f_t)), "cudaMalloc d_objective");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(ws.d_objective, objective.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy objective to device");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_c_basic, m * sizeof(f_t)), "cudaMalloc d_c_basic");
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    // for (i_t k = 0; k < m; ++k) {
    //     const i_t j = basic_list[k]; // j = col
    //     idx in A c_basic[k] = objective[j]; //
    //     costs of objective that wont be zeroed
    //     out
    // }
    phase2_cu::construct_c_basic_kernel<<<grid_size, block_size>>>(m, ws.d_basic_list, ws.d_objective,
                                                                   ws.d_c_basic);
    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "construct_c_basic_kernel");
    std::vector<i_t> fake_indices(m);

    f_t *d_y;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_y, m * sizeof(f_t)), "cudaMalloc d_y");
    // Solve B'*y = cB
    phase2_cu::sparse_pinv_solve_gpu_dense_rhs(ws.cusparse_handle, m, d_B_pinv_col_ptr,
                                               d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                                               ws.d_c_basic, d_y, true);

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    f_t *d_z;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_z, n * sizeof(f_t)), "cudaMalloc d_z");
    phase2_cu::compute_reduced_costs(ws.d_objective, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values, d_y,
                                     ws.d_basic_list, ws.d_nonbasic_list, d_z, m, n);

    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "compute_reduced_costs");

    // Copy z back to host
    CUDA_CALL_AND_CHECK(cudaMemcpy(z.data(), d_z, n * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy z to host");
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

    f_t *d_lp_rhs;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_lp_rhs, m * sizeof(f_t)), "cudaMalloc d_lp_rhs");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_lp_rhs, lp.rhs.data(), m * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy lp.rhs to device");

    f_t *d_x;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_x, n * sizeof(f_t)), "cudaMalloc d_x");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy x to device");

    phase2_cu::compute_primal_variables(ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                        d_B_pinv_values, nz_B_pinv, d_lp_rhs, ws.d_A_col_ptr,
                                        ws.d_A_row_ind, ws.d_A_values, ws.d_basic_list, ws.d_nonbasic_list,
                                        settings.tight_tol, d_x, m, n);

    CUDA_CALL_AND_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy x to host");
    CUDA_CALL_AND_CHECK(cudaMemcpy(y.data(), d_y, m * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy y to host");
    CUDA_CALL_AND_CHECK(cudaFree(d_x), "cudaFree d_x");
    CUDA_CALL_AND_CHECK(cudaFree(d_lp_rhs), "cudaFree d_lp_rhs");

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    if (delta_y_steepest_edge.size() == 0) {
        delta_y_steepest_edge.resize(n);
        if (slack_basis) { // TODO: left off of GPU for now, bc i think is only called at the start,
                           // but needs to be confirmed
            phase2::initialize_steepest_edge_norms_from_slack_basis(basic_list, nonbasic_list,
                                                                    delta_y_steepest_edge);
        } else {
            std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
            if (phase2_cu::initialize_steepest_edge_norms(
                    lp, settings, start_time, ws.d_basic_list, ws.cusparse_handle, d_B_pinv_col_ptr,
                    d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, delta_y_steepest_edge,
                    m) == -1) { // TODO: opted against GPUifing delta_y_steepest_edge for
                                // now
                return dual::status_t::TIME_LIMIT;
            }
        }
    } else {
        settings.log.printf("using exisiting steepest edge %e\n",
                            vector_norm2<i_t, f_t>(delta_y_steepest_edge));
    }

    if (phase == 2) {
        settings.log.printf(" Iter     Objective           Num "
                            "Inf.  Sum Inf.     Perturb  Time\n");
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
    // since x is only read in the above func, d_x and host x are still in sync

    csc_matrix_t<i_t, f_t> A_transpose(1, 1, 0);
    lp.A.transpose(A_transpose);

    f_t obj = compute_objective(lp, x); // TODO: maybe use a reduction on GPU here
    const i_t start_iter = iter;

    i_t sparse_delta_z = 0;
    i_t dense_delta_z = 0;
    phase2::phase2_timers_t<i_t, f_t> timers(settings.profile && phase == 2);

    // TODO jujupieper IMPORTANT STOP HERE FOR NOW

    while (iter < iter_limit) {
        // Pricing
        i_t direction = 0;
        i_t basic_leaving_index = -1;
        i_t leaving_index = -1;
        f_t max_val = 0.0;
        timers.start_timer();
        if (settings.use_steepest_edge_pricing) {
            leaving_index = phase2::steepest_edge_pricing_with_infeasibilities(
                lp, settings, x, delta_y_steepest_edge, basic_mark, squared_infeasibilities,
                infeasibility_indices, direction, basic_leaving_index, max_val);
            // x is only read here, so d_x and host x are in sync
        } else {
            // Max infeasibility pricing
            leaving_index = phase2::phase2_pricing(lp, settings, x, basic_list, direction,
                                                   basic_leaving_index, primal_infeasibility);
        }
        timers.pricing_time += timers.stop_timer();
        if (leaving_index == -1) {
            phase2_cu::prepare_optimality(lp, settings, ws.d_objective, ws.d_A_col_ptr, ws.d_A_row_ind,
                                          ws.d_A_values, ws.cusparse_handle, d_B_pinv_col_ptr,
                                          d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, objective,
                                          basic_list, ws.d_basic_list, nonbasic_list, ws.d_nonbasic_list,
                                          vstatus, phase, start_time, max_val, iter, x, y, z, sol);
            status = dual::status_t::OPTIMAL;
            break;
        }

        // BTran
        // BT*delta_y = -delta_zB = -sigma*ei
        timers.start_timer();
        // sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
        i_t *d_delta_y_sparse_indices;
        f_t *d_delta_y_sparse_values;
        i_t nz_delta_y_sparse = 0;

        i_t *d_delta_z_sparse_indices;
        f_t *d_delta_z_sparse_values;
        i_t nz_delta_z_sparse = 0;

        phase2_cu::compute_delta_y(ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                   d_B_pinv_values, nz_B_pinv, basic_leaving_index, direction,
                                   d_delta_y_sparse_indices, d_delta_y_sparse_values,
                                   nz_delta_y_sparse, m);
        timers.btran_time += timers.stop_timer();

        f_t steepest_edge_norm_check = phase2_cu::sparse_vector_squared_norm_gpu(
            nz_delta_y_sparse, d_delta_y_sparse_indices, d_delta_y_sparse_values);

        if (delta_y_steepest_edge[leaving_index] <
            settings.steepest_edge_ratio * steepest_edge_norm_check) {
            constexpr bool verbose = false;
            if constexpr (verbose) {
                settings.log.printf("iteration restart due to "
                                    "steepest edge. Leaving %d. "
                                    "Actual %.2e "
                                    "from update %.2e\n",
                                    leaving_index, steepest_edge_norm_check,
                                    delta_y_steepest_edge[leaving_index]);
            }
            delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
            continue;
        }

        timers.start_timer();
        // i_t delta_y_nz0 = 0;
        // const i_t nz_delta_y = delta_y_sparse.i.size();
        // for (i_t k = 0; k < nz_delta_y; k++) {
        //     if (std::abs(delta_y_sparse.x[k]) > 1e-12) {
        //         delta_y_nz0++;
        //     }
        // }
        const f_t delta_y_nz_percentage = nz_delta_y_sparse / static_cast<f_t>(m) * 100.0;
        const bool use_transpose = delta_y_nz_percentage <= 30.0;
        sparse_vector_t<i_t, f_t> delta_y_sparse(m, nz_delta_y_sparse);
        CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.i.data(), d_delta_y_sparse_indices,
                                       nz_delta_y_sparse * sizeof(i_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy delta_y_sparse indices to host");
        CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.x.data(), d_delta_y_sparse_values,
                                       nz_delta_y_sparse * sizeof(f_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy delta_y_sparse values to host");
        if (use_transpose) {
            sparse_delta_z++;

            phase2_cu::compute_delta_z(A_transpose, delta_y_sparse, leaving_index, direction,
                                       nonbasic_mark, delta_z_mark, delta_z_indices, delta_z);
        } else {
            dense_delta_z++;
            // delta_zB = sigma*ei
            delta_y_sparse.to_dense(delta_y);
            phase2::compute_reduced_cost_update(lp, basic_list, nonbasic_list, delta_y,
                                                leaving_index, direction, delta_z_mark,
                                                delta_z_indices, delta_z);
        }

        nz_delta_z_sparse = delta_z_indices.size();

        std::vector<f_t> delta_z_values_packed;
        delta_z_values_packed.reserve(nz_delta_z_sparse);
        for (i_t idx : delta_z_indices) {
            delta_z_values_packed.push_back(delta_z[idx]);
        }

        CUDA_CALL_AND_CHECK(cudaMalloc(&d_delta_z_sparse_indices, nz_delta_z_sparse * sizeof(i_t)),
                            "cudaMalloc d_delta_z_sparse_indices");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_delta_z_sparse_values, nz_delta_z_sparse * sizeof(f_t)),
                            "cudaMalloc d_delta_z_sparse_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_delta_z_sparse_indices, delta_z_indices.data(),
                                       nz_delta_z_sparse * sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy delta_z_indices to device");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_delta_z_sparse_values, delta_z.data(),
                                       nz_delta_z_sparse * sizeof(f_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy delta_z values to device");
        timers.delta_z_time += timers.stop_timer();

        // Ratio test
        f_t step_length;
        i_t entering_index = -1;
        i_t nonbasic_entering_index = -1;
        const bool harris_ratio = settings.use_harris_ratio;
        const bool bound_flip_ratio = settings.use_bound_flip_ratio;
        if (harris_ratio) {
            f_t max_step_length =
                phase2::first_stage_harris(lp, vstatus, nonbasic_list, z, delta_z);
            entering_index =
                phase2::second_stage_harris(lp, vstatus, nonbasic_list, z, delta_z, max_step_length,
                                            step_length, nonbasic_entering_index);
        } else if (bound_flip_ratio) {
            timers.start_timer();
            f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                                       : (x[leaving_index] - lp.upper[leaving_index]);
            bound_flipping_ratio_test_t<i_t, f_t> bfrt(
                settings, start_time, m, n, slope, lp.lower, lp.upper, bounded_variables, vstatus,
                nonbasic_list, z, delta_z, delta_z_indices, nonbasic_mark);
            entering_index = bfrt.compute_step_length(step_length, nonbasic_entering_index);
            timers.bfrt_time += timers.stop_timer();
        } else {
            entering_index =
                phase2::phase2_ratio_test(lp, settings, vstatus, nonbasic_list, z, delta_z,
                                          step_length, nonbasic_entering_index);
        }
        if (entering_index == -2) {
            return dual::status_t::TIME_LIMIT;
        }
        if (entering_index == -3) {
            return dual::status_t::CONCURRENT_LIMIT;
        }
        if (entering_index == -1) {
            settings.log.printf("No entering variable found. "
                                "Iter %d\n",
                                iter);
            settings.log.printf("Scaled infeasibility %e\n", max_val);
            f_t perturbation = phase2::amount_of_perturbation(lp, objective);

            if (perturbation > 0.0 && phase == 2) {
                // Try to remove perturbation
                std::vector<f_t> unperturbed_y(m);
                std::vector<f_t> unperturbed_z(n);
                f_t *d_unperturbed_y, *d_unperturbed_z;
                CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_y, m * sizeof(f_t)),
                                    "cudaMalloc unperturbed_y");
                CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_z, n * sizeof(f_t)),
                                    "cudaMalloc unperturbed_z");
                phase2_cu::compute_dual_solution_from_basis(
                    ws.d_objective, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values, ws.cusparse_handle,
                    d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, ws.d_basic_list,
                    ws.d_nonbasic_list, d_unperturbed_y, d_unperturbed_z, m, n);
                CUDA_CALL_AND_CHECK(cudaMemcpy(unperturbed_y.data(), d_unperturbed_y,
                                               m * sizeof(f_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy unperturbed_y to host");
                CUDA_CALL_AND_CHECK(cudaMemcpy(unperturbed_z.data(), d_unperturbed_z,
                                               n * sizeof(f_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy unperturbed_z to host");
                CUDA_CALL_AND_CHECK(cudaFree(d_unperturbed_y), "cudaFree d_unperturbed_y");
                CUDA_CALL_AND_CHECK(cudaFree(d_unperturbed_z), "cudaFree d_unperturbed_z");
                {
                    const f_t dual_infeas =
                        phase2::dual_infeasibility(lp, settings, vstatus, unperturbed_z,
                                                   settings.tight_tol, settings.dual_tol);
                    settings.log.printf("Dual infeasibility "
                                        "after removing "
                                        "perturbation %e\n",
                                        dual_infeas);
                    f_t *d_unperturbed_x;
                    CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_x, n * sizeof(f_t)),
                                        "cudaMalloc unperturbed_x");
                    if (dual_infeas <= settings.dual_tol) {
                        settings.log.printf("Removed "
                                            "perturbation of "
                                            "%.2e.\n",
                                            perturbation);
                        z = unperturbed_z;
                        y = unperturbed_y;
                        perturbation = 0.0;

                        std::vector<f_t> unperturbed_x(n);

                        phase2_cu::compute_primal_solution_from_basis(
                            lp, d_lp_rhs, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values, ws.cusparse_handle,
                            d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                            ws.d_basic_list, ws.d_nonbasic_list, vstatus, unperturbed_x, d_unperturbed_x,
                            m, n);
                        x = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                            lp, settings, basic_list, x, squared_infeasibilities,
                            infeasibility_indices);
                        settings.log.printf("Updated primal "
                                            "infeasibility: %e\n",
                                            primal_infeasibility);

                        objective = lp.objective;
                        // Need to reset the
                        // objective value, since
                        // we have recomputed x
                        obj = phase2::compute_perturbed_objective(objective, x);
                        if (dual_infeas <= settings.dual_tol &&
                            primal_infeasibility <= settings.primal_tol) {
                            phase2_cu::prepare_optimality(
                                lp, settings, ws.d_objective, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values,
                                ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                d_B_pinv_values, nz_B_pinv, lp.objective, basic_list, ws.d_basic_list,
                                nonbasic_list, ws.d_nonbasic_list, vstatus, phase, start_time, max_val,
                                iter, x, y, z, sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf("Continuing with "
                                            "perturbation "
                                            "removed and "
                                            "steepest edge norms "
                                            "reset\n");
                        // Clear delta_z before
                        // restarting the
                        // iteration
                        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark,
                                              delta_z_indices, delta_z);
                        continue;
                    } else {
                        std::vector<f_t> unperturbed_x(n);
                        phase2_cu::compute_primal_solution_from_basis(
                            lp, d_lp_rhs, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values, ws.cusparse_handle,
                            d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                            ws.d_basic_list, ws.d_nonbasic_list, vstatus, unperturbed_x, d_unperturbed_x,
                            m, n);
                        x = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                            lp, settings, basic_list, x, squared_infeasibilities,
                            infeasibility_indices);

                        const f_t orig_dual_infeas = phase2::dual_infeasibility(
                            lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);

                        if (primal_infeasibility <= settings.primal_tol &&
                            orig_dual_infeas <= settings.dual_tol) {
                            phase2_cu::prepare_optimality(
                                lp, settings, ws.d_objective, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values,
                                ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                d_B_pinv_values, nz_B_pinv, objective, basic_list, ws.d_basic_list,
                                nonbasic_list, ws.d_nonbasic_list, vstatus, phase, start_time, max_val,
                                iter, x, y, z, sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf("Failed to remove "
                                            "perturbation of "
                                            "%.2e.\n",
                                            perturbation);
                    }
                }
            }

            const f_t dual_infeas = phase2::dual_infeasibility(
                lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
            settings.log.printf("Dual infeasibility %e\n", dual_infeas);
            const f_t primal_inf = phase2::primal_infeasibility(lp, settings, vstatus, x);
            settings.log.printf("Primal infeasibility %e\n", primal_inf);
            settings.log.printf("Steepest edge %e\n", max_val);
            if (dual_infeas > settings.dual_tol) {
                settings.log.printf("Numerical issues "
                                    "encountered. No entering "
                                    "variable found with "
                                    "large infeasibility.\n");
                return dual::status_t::NUMERICAL;
            }
            return dual::status_t::DUAL_UNBOUNDED;
        }

        timers.start_timer();
        // Update dual variables
        // y <- y + steplength * delta_y
        // z <- z + steplength * delta_z
        // phase2_cu::update_dual_variables(d_delta_y_sparse_indices, d_delta_y_sparse_values,
        // nz_delta_y_sparse, d_delta_z_sparse_indices, d_delta_z_sparse_values, nz_delta_z_sparse,
        // step_length, leaving_index, d_y, d_z);
        phase2::update_dual_variables(delta_y_sparse, delta_z_indices, delta_z, step_length,
                                      leaving_index, y, z);

        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.i.data(), d_delta_y_sparse_indices,
        // nz_delta_y_sparse * sizeof(i_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_y_sparse.i");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.x.data(), d_delta_y_sparse_values,
        // nz_delta_y_sparse * sizeof(f_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_y_sparse.x");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_z_indices.data(), d_delta_z_sparse_indices,
        // nz_delta_z_sparse * sizeof(i_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_z_indices");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_z.data(), d_delta_z_sparse_values, nz_delta_z_sparse
        // * sizeof(f_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_z");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(y.data(), d_y, sizeof(y), cudaMemcpyDeviceToHost),
        // "cudaMemcpy y"); CUDA_CALL_AND_CHECK(cudaMemcpy(z.data(), d_z, sizeof(z),
        // cudaMemcpyDeviceToHost), "cudaMemcpy z");

        // z[leaving_index] += step_length * delta_z[leaving_index];
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // Update primal variable
        const i_t num_flipped = phase2::flip_bounds(
            lp, settings, bounded_variables, objective, z, delta_z_indices, nonbasic_list,
            entering_index, vstatus, delta_x_flip, atilde_mark, atilde, atilde_index);

        timers.flip_time += timers.stop_timer();

        sparse_vector_t<i_t, f_t> delta_xB_0_sparse(m, 0);
        if (num_flipped > 0) {
            timers.start_timer();
            phase2_cu::adjust_for_flips(ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                        d_B_pinv_values, nz_B_pinv, basic_list, delta_z_indices,
                                        atilde_index, atilde, atilde_mark, delta_xB_0_sparse,
                                        delta_x_flip, x);
            timers.ftran_time += timers.stop_timer();
        }

        timers.start_timer();
        sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
        sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
        if (phase2_cu::compute_delta_x(lp, ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                       d_B_pinv_values, nz_B_pinv, entering_index, leaving_index,
                                       basic_leaving_index, direction, basic_list, delta_x_flip,
                                       rhs_sparse, x, scaled_delta_xB_sparse, delta_x, m) == -1) {
            settings.log.printf("Failed to compute delta_x. Iter "
                                "%d\n",
                                iter);
            return dual::status_t::NUMERICAL;
        }

        timers.ftran_time += timers.stop_timer();

        timers.start_timer();
        const i_t steepest_edge_status = phase2_cu::update_steepest_edge_norms(
            settings, basic_list, ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
            d_B_pinv_values, nz_B_pinv, direction, d_delta_y_sparse_indices,
            d_delta_y_sparse_values, nz_delta_y_sparse, steepest_edge_norm_check,
            scaled_delta_xB_sparse, basic_leaving_index, entering_index, v, delta_y_steepest_edge,
            m);
        assert(steepest_edge_status == 0);
        timers.se_norms_time += timers.stop_timer();

        timers.start_timer();
        // x <- x + delta_x
        phase2::update_primal_variables(scaled_delta_xB_sparse, basic_list, delta_x, entering_index,
                                        x);
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // TODO(CMM): Do I also need to update the
        // objective due to the bound flips?
        // TODO(CMM): I'm using the unperturbed
        // objective here, should this be the
        // perturbed objective?
        phase2::update_objective(basic_list, scaled_delta_xB_sparse.i, lp.objective, delta_x,
                                 entering_index, obj);
        timers.objective_time += timers.stop_timer();

        timers.start_timer();
        // Update primal infeasibilities due to
        // changes in basic variables from
        // flipping bounds
        phase2::update_primal_infeasibilities(
            lp, settings, basic_list, x, entering_index, leaving_index, delta_xB_0_sparse.i,
            squared_infeasibilities, infeasibility_indices, primal_infeasibility);
        // Update primal infeasibilities due to
        // changes in basic variables from the
        // leaving and entering variables
        phase2::update_primal_infeasibilities(
            lp, settings, basic_list, x, entering_index, leaving_index, scaled_delta_xB_sparse.i,
            squared_infeasibilities, infeasibility_indices, primal_infeasibility);
        // Update the entering variable
        phase2::update_single_primal_infeasibility(lp.lower, lp.upper, x, settings.primal_tol,
                                                   squared_infeasibilities, infeasibility_indices,
                                                   entering_index, primal_infeasibility);

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

        CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "Before basis update");
        // Update basic and nonbasic lists and
        // marks

        basic_list[basic_leaving_index] = entering_index;
        nonbasic_list[nonbasic_entering_index] = leaving_index;
        nonbasic_mark[entering_index] = -1;
        nonbasic_mark[leaving_index] = nonbasic_entering_index;
        basic_mark[leaving_index] = -1;
        basic_mark[entering_index] = basic_leaving_index;
        CUDA_CALL_AND_CHECK(cudaMemcpy(ws.d_basic_list + basic_leaving_index, &entering_index,
                                       sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy entering_index to "
                            "d_basic_list");
        CUDA_CALL_AND_CHECK(cudaMemcpy(ws.d_nonbasic_list + nonbasic_entering_index, &leaving_index,
                                       sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy leaving_index to "
                            "d_nonbasic_list");

        timers.start_timer();
        // Refactor or update the basis
        // factorization
        bool should_refactor = (iter + 1) % settings.refactor_frequency == 0;

        if (!should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Update 1");
            }

            // should_refactor = !phase2_cu::eta_update_inverse(
            //     cublas_handle, m, d_B_pinv, eta_b_old, eta_b_new, eta_v, eta_c, eta_d,
            //     d_A_col_ptr, d_A_row_ind, d_A_values, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values,
            //     basic_leaving_index, entering_index);
            should_refactor = !phase2_cu::eta_update_inverse(
                ws.cublas_handle, m, ws.B_pinv_cusparse, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                d_B_pinv_values, d_B_pinv_col_ptr_buffer, d_B_pinv_row_ind_buffer, d_B_pinv_values_buffer, nz_B_pinv, len_B_pinv, len_B_pinv_buffer, ws.eta_b_old, ws.eta_b_new, ws.eta_v, ws.eta_c, ws.eta_d, ws.d_A_col_ptr,
                ws.d_A_row_ind, ws.d_A_values, ws.d_B_col_ptr, d_B_row_ind, d_B_values, basic_leaving_index,
                entering_index, settings);

            if (settings.profile) {
                settings.timer.stop("Inverse Update 1");
            }
        }

        // Free old B and Bt and recompute
        CUDA_CALL_AND_CHECK(cudaFree(d_B_row_ind), "cudaFree d_B_col_ptr");
        CUDA_CALL_AND_CHECK(cudaFree(d_B_values), "cudaFree d_B_values");

        if (!should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Update 2");
            }

            // Move basic list to device
            // TODONE: as above consider keeping
            // this on device i_t *d_basic_list;
            // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list,
            // m * sizeof(i_t)),
            //                     "cudaMalloc
            //                     d_basic_list");
            // CUDA_CALL_AND_CHECK(cudaMemcpy(d_basic_list,
            // basic_list.data(), m * sizeof(i_t),
            //                                cudaMemcpyHostToDevice),
            //                     "cudaMemcpy to
            //                     d_basic_list");

            // TODO: It's probably not smart to rebuild B and Bt from scratch every time
            // Instead use a 95% threshold (or higher) to determine the size per row to
            // allocate s.t. we don't have to realloc every time
            phase2_cu::build_basis_on_device(m, ws.d_A_col_ptr, ws.d_A_row_ind, ws.d_A_values, ws.d_basic_list,
                                             ws.d_B_col_ptr, d_B_row_ind, d_B_values, nz_B);

            if (settings.profile) {
                settings.timer.stop("Inverse Update 2");
            }
        }

        if (should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Refactorizaton");
            }

            // Recompute d_B_pinv
            phase2_cu::compute_inverse<i_t, f_t>(
                ws.cusparse_handle, ws.cudss_handle, ws.cudss_config, m, n, ws.d_A_col_ptr, ws.d_A_row_ind,
                ws.d_A_values, ws.d_B_col_ptr, d_B_row_ind, d_B_values, basic_list, d_B_pinv_col_ptr,
                d_B_pinv_row_ind, d_B_pinv_values, nz_B, nz_B_pinv, len_B_pinv, settings);

            CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(ws.B_pinv_cusparse),
                                    "cusparseDestroyMatDescr B_pinv");
            CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&ws.B_pinv_cusparse, m, m, nz_B_pinv,
                                                            d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                                            d_B_pinv_values,
                                                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                            CUSPARSE_INDEX_BASE_ZERO,
                                                            CUDA_R_64F),
                                    "cusparseCreateCsc B_pinv");

            phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);
            phase2::compute_initial_primal_infeasibilities(
                lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

            if (settings.profile) {
                settings.timer.stop("Inverse Refactorizaton");
            }
        }
        timers.lu_update_time += timers.stop_timer();

        timers.start_timer();
        phase2_cu::compute_steepest_edge_norm_entering(
            settings, m, ws.cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values,
            nz_B_pinv, basic_leaving_index, entering_index, delta_y_steepest_edge);
        timers.se_entering_time += timers.stop_timer();

        iter++;

        // Clear delta_z
        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices,
                              delta_z);

        f_t now = toc(start_time);
        if ((iter - start_iter) < settings.first_iteration_log ||
            (iter % settings.iteration_log_frequency) == 0) {
            if (phase == 1 && iter == 1) {
                settings.log.printf(" Iter     Objective         "
                                    "  Num Inf.  Sum Inf.     "
                                    "Perturb  Time\n");
            }
            settings.log.printf("%5d %+.16e %7d %.8e %.2e %.2f\n", iter,
                                compute_user_objective(lp, obj), infeasibility_indices.size(),
                                primal_infeasibility, sum_perturb, now);
        }

        if (obj >= settings.cut_off) {
            settings.log.printf("Solve cutoff. Current objective "
                                "%e. Cutoff %e\n",
                                obj, settings.cut_off);
            return dual::status_t::CUTOFF;
        }

        if (now > settings.time_limit) {
            return dual::status_t::TIME_LIMIT;
        }

        if (settings.concurrent_halt != nullptr &&
            settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
            return dual::status_t::CONCURRENT_LIMIT;
        }

        // End of iteration cleanup
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_y_sparse_indices), "cudaFree d_delta_y_sparse_indices");
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_y_sparse_values), "cudaFree d_delta_y_sparse_values");
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_z_sparse_indices), "cudaFree d_delta_z_sparse_indices");
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_z_sparse_values), "cudaFree d_delta_z_sparse_values");
    }
    if (iter >= iter_limit) {
        status = dual::status_t::ITERATION_LIMIT;
    }

    if (phase == 2) {
        timers.print_timers(settings);
    }

    // Cleanup GPU resources
    // CUDA_CALL_AND_CHECK(cudaFree(d_A_col_ptr), "cudaFree d_A_col_ptr");
    // CUDA_CALL_AND_CHECK(cudaFree(d_A_row_ind), "cudaFree d_A_row_ind");
    // CUDA_CALL_AND_CHECK(cudaFree(d_A_values), "cudaFree d_A_values");
    // CUDA_CALL_AND_CHECK(cudaFree(d_B_col_ptr), "cudaFree d_B_col_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_values), "cudaFree d_B_values");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_col_ptr), "cudaFree d_B_pinv_col_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_row_ind), "cudaFree d_B_pinv_row_ind");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_values), "cudaFree d_B_pinv_values");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_col_ptr_buffer), "cudaFree d_B_pinv_col_ptr_buffer");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_row_ind_buffer), "cudaFree d_B_pinv_row_ind_buffer");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_values_buffer), "cudaFree d_B_pinv_values_buffer");
    // CUBLAS_CALL_AND_CHECK(cublasDestroy(cublas_handle), "cublasDestroy");
    // CUSPARSE_CALL_AND_CHECK(cusparseDestroy(cusparse_handle), "cusparseDestroy");
    // CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(B_pinv_cusparse),
    //                         "cusparseDestroyMatDescr B_pinv");
    // CUSPARSE_CALL_AND_CHECK(cusparseDestroy(cusparse_pinv_handle), "cusparseDestroy Pinv");
    // CUDSS_CALL_AND_CHECK(cudssConfigDestroy(cudss_config), "cudssConfigDestroy B");
    // CUDSS_CALL_AND_CHECK(cudssDestroy(cudss_handle), "cudssDestroyHandle B");

    return status;
}


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

    // Analyze matrix A
    problem_analyzer_t<i_t, f_t> analyzer(lp, settings);
    analyzer.analyze();
    analyzer.display_analysis();

    // SETUP GPU ALLOCATIONS AND HANDLES
    // Create all handles
    cublasHandle_t cublas_handle;
    CUBLAS_CALL_AND_CHECK(cublasCreate(&cublas_handle), "cublasCreate");
    cusparseHandle_t cusparse_handle;
    CUSPARSE_CALL_AND_CHECK(cusparseCreate(&cusparse_handle), "cusparseCreate");
    cusparseHandle_t cusparse_pinv_handle;
    CUSPARSE_CALL_AND_CHECK(cusparseCreate(&cusparse_pinv_handle), "cusparseCreate pinv");
    cudssHandle_t cudss_handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&cudss_handle), "cudssCreateHandle B");
    cudssConfig_t cudss_config;
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&cudss_config), "cudssCreateConfig B");
    i_t use_matching = 1;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(cudss_config, CUDSS_CONFIG_USE_MATCHING, &use_matching, sizeof(i_t)),
        "cudssConfigSetParameter ENABLE_MATCHINGS B");
    cudssAlgType_t matching_alg = CUDSS_ALG_5;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(cudss_config, CUDSS_CONFIG_MATCHING_ALG, &matching_alg,
                                        sizeof(cudssAlgType_t)),
                         "cudssConfigSetParameter MATCHING_ALG_TYPE B");

    // Create matrix representations
    cusparseSpMatDescr_t B_pinv_cusparse;

    // Move A to device
    i_t *d_A_col_ptr;
    i_t *d_A_row_ind;
    f_t *d_A_values;
    phase2_cu::move_A_to_device(lp.A, d_A_col_ptr, d_A_row_ind, d_A_values);

    // Create dense vectors for eta updates
    // TODO: remove once we have dense sparse mv
    // on device
    f_t *eta_b_old, *eta_b_new, *eta_v, *eta_c, *eta_d;
    CUDA_CALL_AND_CHECK(cudaMalloc(&eta_b_old, m * sizeof(f_t)), "cudaMalloc eta_b_old");
    CUDA_CALL_AND_CHECK(cudaMalloc(&eta_b_new, m * sizeof(f_t)), "cudaMalloc eta_b_new");
    CUDA_CALL_AND_CHECK(cudaMalloc(&eta_v, m * sizeof(f_t)), "cudaMalloc eta_v");
    CUDA_CALL_AND_CHECK(cudaMalloc(&eta_c, m * sizeof(f_t)), "cudaMalloc eta_c");
    CUDA_CALL_AND_CHECK(cudaMalloc(&eta_d, m * sizeof(f_t)), "cudaMalloc eta_d");

    // Compute Moore-Penrose pseudo-inverse of B
    i_t *d_B_col_ptr;
    i_t *d_B_row_ind;
    f_t *d_B_values;
    i_t nz_B;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_col_ptr");

    i_t len_B_pinv = 0;
    i_t *d_B_pinv_col_ptr;
    i_t *d_B_pinv_row_ind;
    f_t *d_B_pinv_values;
    i_t nz_B_pinv;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_col_ptr, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_B_pinv_col_ptr");
    phase2_cu::compute_inverse<i_t, f_t>(
        cusparse_handle, cudss_handle, cudss_config, m, n, d_A_col_ptr, d_A_row_ind, d_A_values,
        d_B_col_ptr, d_B_row_ind, d_B_values, basic_list, d_B_pinv_col_ptr, d_B_pinv_row_ind,
        d_B_pinv_values, nz_B, nz_B_pinv, len_B_pinv, settings);

    // We pre-allocate buffer for eta updates
    i_t len_B_pinv_buffer = static_cast<i_t>(settings.pinv_buffer_size_multiplier * nz_B_pinv);
    i_t *d_B_pinv_col_ptr_buffer;
    i_t *d_B_pinv_row_ind_buffer;
    f_t *d_B_pinv_values_buffer;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_col_ptr_buffer, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_B_pinv_col_ptr_buffer");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_row_ind_buffer, len_B_pinv_buffer * sizeof(i_t)),
                        "cudaMalloc d_B_pinv_row_ind_buffer");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv_values_buffer, len_B_pinv_buffer * sizeof(f_t)),
                        "cudaMalloc d_B_pinv_values_buffer");

    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&B_pinv_cusparse, m, m, nz_B_pinv, d_B_pinv_col_ptr,
                                              d_B_pinv_row_ind, d_B_pinv_values, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F),
                            "cusparseCreateCsc B_pinv_cusparse");

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    f_t *d_temp_vector_m; // Temporary vector of size m on device, so we dont have to keep
                          // allocating one, never make assumptions about its contents!!!
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_temp_vector_m, m * sizeof(f_t)),
                        "cudaMalloc d_temp_vector_m");

    i_t *d_basic_list, *d_nonbasic_list;
    f_t *d_c_basic, *d_objective;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy basic_list to device");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_nonbasic_list, (n - m) * sizeof(i_t)),
                        "cudaMalloc d_nonbasic_list");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_nonbasic_list, nonbasic_list.data(), (n - m) * sizeof(i_t),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy nonbasic_list to device");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_objective, n * sizeof(f_t)), "cudaMalloc d_objective");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_objective, objective.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy objective to device");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_c_basic, m * sizeof(f_t)), "cudaMalloc d_c_basic");
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    // for (i_t k = 0; k < m; ++k) {
    //     const i_t j = basic_list[k]; // j = col
    //     idx in A c_basic[k] = objective[j]; //
    //     costs of objective that wont be zeroed
    //     out
    // }
    phase2_cu::construct_c_basic_kernel<<<grid_size, block_size>>>(m, d_basic_list, d_objective,
                                                                   d_c_basic);
    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "construct_c_basic_kernel");
    std::vector<i_t> fake_indices(m);

    f_t *d_y;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_y, m * sizeof(f_t)), "cudaMalloc d_y");
    // Solve B'*y = cB
    phase2_cu::sparse_pinv_solve_gpu_dense_rhs(cusparse_handle, m, d_B_pinv_col_ptr,
                                               d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                                               d_c_basic, d_y, true);

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    f_t *d_z;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_z, n * sizeof(f_t)), "cudaMalloc d_z");
    phase2_cu::compute_reduced_costs(d_objective, d_A_col_ptr, d_A_row_ind, d_A_values, d_y,
                                     d_basic_list, d_nonbasic_list, d_z, m, n);

    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "compute_reduced_costs");

    // Copy z back to host
    CUDA_CALL_AND_CHECK(cudaMemcpy(z.data(), d_z, n * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy z to host");
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

    f_t *d_lp_rhs;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_lp_rhs, m * sizeof(f_t)), "cudaMalloc d_lp_rhs");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_lp_rhs, lp.rhs.data(), m * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy lp.rhs to device");

    f_t *d_x;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_x, n * sizeof(f_t)), "cudaMalloc d_x");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy x to device");

    phase2_cu::compute_primal_variables(cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                        d_B_pinv_values, nz_B_pinv, d_lp_rhs, d_A_col_ptr,
                                        d_A_row_ind, d_A_values, d_basic_list, d_nonbasic_list,
                                        settings.tight_tol, d_x, m, n);

    CUDA_CALL_AND_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy x to host");
    CUDA_CALL_AND_CHECK(cudaMemcpy(y.data(), d_y, m * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy y to host");
    CUDA_CALL_AND_CHECK(cudaFree(d_x), "cudaFree d_x");
    CUDA_CALL_AND_CHECK(cudaFree(d_lp_rhs), "cudaFree d_lp_rhs");

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    if (delta_y_steepest_edge.size() == 0) {
        delta_y_steepest_edge.resize(n);
        if (slack_basis) { // TODO: left off of GPU for now, bc i think is only called at the start,
                           // but needs to be confirmed
            phase2::initialize_steepest_edge_norms_from_slack_basis(basic_list, nonbasic_list,
                                                                    delta_y_steepest_edge);
        } else {
            std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
            if (phase2_cu::initialize_steepest_edge_norms(
                    lp, settings, start_time, d_basic_list, cusparse_handle, d_B_pinv_col_ptr,
                    d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, delta_y_steepest_edge,
                    m) == -1) { // TODO: opted against GPUifing delta_y_steepest_edge for
                                // now
                return dual::status_t::TIME_LIMIT;
            }
        }
    } else {
        settings.log.printf("using exisiting steepest edge %e\n",
                            vector_norm2<i_t, f_t>(delta_y_steepest_edge));
    }

    if (phase == 2) {
        settings.log.printf(" Iter     Objective           Num "
                            "Inf.  Sum Inf.     Perturb  Time\n");
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
    // since x is only read in the above func, d_x and host x are still in sync

    csc_matrix_t<i_t, f_t> A_transpose(1, 1, 0);
    lp.A.transpose(A_transpose);

    f_t obj = compute_objective(lp, x); // TODO: maybe use a reduction on GPU here
    const i_t start_iter = iter;

    i_t sparse_delta_z = 0;
    i_t dense_delta_z = 0;
    phase2::phase2_timers_t<i_t, f_t> timers(settings.profile && phase == 2);

    while (iter < iter_limit) {
        // Pricing
        i_t direction = 0;
        i_t basic_leaving_index = -1;
        i_t leaving_index = -1;
        f_t max_val = 0.0;
        timers.start_timer();
        if (settings.use_steepest_edge_pricing) {
            leaving_index = phase2::steepest_edge_pricing_with_infeasibilities(
                lp, settings, x, delta_y_steepest_edge, basic_mark, squared_infeasibilities,
                infeasibility_indices, direction, basic_leaving_index, max_val);
            // x is only read here, so d_x and host x are in sync
        } else {
            // Max infeasibility pricing
            leaving_index = phase2::phase2_pricing(lp, settings, x, basic_list, direction,
                                                   basic_leaving_index, primal_infeasibility);
        }
        timers.pricing_time += timers.stop_timer();
        if (leaving_index == -1) {
            phase2_cu::prepare_optimality(lp, settings, d_objective, d_A_col_ptr, d_A_row_ind,
                                          d_A_values, cusparse_handle, d_B_pinv_col_ptr,
                                          d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, objective,
                                          basic_list, d_basic_list, nonbasic_list, d_nonbasic_list,
                                          vstatus, phase, start_time, max_val, iter, x, y, z, sol);
            status = dual::status_t::OPTIMAL;
            break;
        }

        // BTran
        // BT*delta_y = -delta_zB = -sigma*ei
        timers.start_timer();
        // sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
        i_t *d_delta_y_sparse_indices;
        f_t *d_delta_y_sparse_values;
        i_t nz_delta_y_sparse = 0;

        i_t *d_delta_z_sparse_indices;
        f_t *d_delta_z_sparse_values;
        i_t nz_delta_z_sparse = 0;

        phase2_cu::compute_delta_y(cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                   d_B_pinv_values, nz_B_pinv, basic_leaving_index, direction,
                                   d_delta_y_sparse_indices, d_delta_y_sparse_values,
                                   nz_delta_y_sparse, m);
        timers.btran_time += timers.stop_timer();

        f_t steepest_edge_norm_check = phase2_cu::sparse_vector_squared_norm_gpu(
            nz_delta_y_sparse, d_delta_y_sparse_indices, d_delta_y_sparse_values);

        if (delta_y_steepest_edge[leaving_index] <
            settings.steepest_edge_ratio * steepest_edge_norm_check) {
            constexpr bool verbose = false;
            if constexpr (verbose) {
                settings.log.printf("iteration restart due to "
                                    "steepest edge. Leaving %d. "
                                    "Actual %.2e "
                                    "from update %.2e\n",
                                    leaving_index, steepest_edge_norm_check,
                                    delta_y_steepest_edge[leaving_index]);
            }
            delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
            continue;
        }

        timers.start_timer();
        // i_t delta_y_nz0 = 0;
        // const i_t nz_delta_y = delta_y_sparse.i.size();
        // for (i_t k = 0; k < nz_delta_y; k++) {
        //     if (std::abs(delta_y_sparse.x[k]) > 1e-12) {
        //         delta_y_nz0++;
        //     }
        // }
        const f_t delta_y_nz_percentage = nz_delta_y_sparse / static_cast<f_t>(m) * 100.0;
        const bool use_transpose = delta_y_nz_percentage <= 30.0;
        sparse_vector_t<i_t, f_t> delta_y_sparse(m, nz_delta_y_sparse);
        CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.i.data(), d_delta_y_sparse_indices,
                                       nz_delta_y_sparse * sizeof(i_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy delta_y_sparse indices to host");
        CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.x.data(), d_delta_y_sparse_values,
                                       nz_delta_y_sparse * sizeof(f_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy delta_y_sparse values to host");
        if (use_transpose) {
            sparse_delta_z++;

            phase2_cu::compute_delta_z(A_transpose, delta_y_sparse, leaving_index, direction,
                                       nonbasic_mark, delta_z_mark, delta_z_indices, delta_z);
        } else {
            dense_delta_z++;
            // delta_zB = sigma*ei
            delta_y_sparse.to_dense(delta_y);
            phase2::compute_reduced_cost_update(lp, basic_list, nonbasic_list, delta_y,
                                                leaving_index, direction, delta_z_mark,
                                                delta_z_indices, delta_z);
        }

        nz_delta_z_sparse = delta_z_indices.size();

        std::vector<f_t> delta_z_values_packed;
        delta_z_values_packed.reserve(nz_delta_z_sparse);
        for (i_t idx : delta_z_indices) {
            delta_z_values_packed.push_back(delta_z[idx]);
        }

        CUDA_CALL_AND_CHECK(cudaMalloc(&d_delta_z_sparse_indices, nz_delta_z_sparse * sizeof(i_t)),
                            "cudaMalloc d_delta_z_sparse_indices");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_delta_z_sparse_values, nz_delta_z_sparse * sizeof(f_t)),
                            "cudaMalloc d_delta_z_sparse_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_delta_z_sparse_indices, delta_z_indices.data(),
                                       nz_delta_z_sparse * sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy delta_z_indices to device");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_delta_z_sparse_values, delta_z.data(),
                                       nz_delta_z_sparse * sizeof(f_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy delta_z values to device");
        timers.delta_z_time += timers.stop_timer();

        // Ratio test
        f_t step_length;
        i_t entering_index = -1;
        i_t nonbasic_entering_index = -1;
        const bool harris_ratio = settings.use_harris_ratio;
        const bool bound_flip_ratio = settings.use_bound_flip_ratio;
        if (harris_ratio) {
            f_t max_step_length =
                phase2::first_stage_harris(lp, vstatus, nonbasic_list, z, delta_z);
            entering_index =
                phase2::second_stage_harris(lp, vstatus, nonbasic_list, z, delta_z, max_step_length,
                                            step_length, nonbasic_entering_index);
        } else if (bound_flip_ratio) {
            timers.start_timer();
            f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                                       : (x[leaving_index] - lp.upper[leaving_index]);
            bound_flipping_ratio_test_t<i_t, f_t> bfrt(
                settings, start_time, m, n, slope, lp.lower, lp.upper, bounded_variables, vstatus,
                nonbasic_list, z, delta_z, delta_z_indices, nonbasic_mark);
            entering_index = bfrt.compute_step_length(step_length, nonbasic_entering_index);
            timers.bfrt_time += timers.stop_timer();
        } else {
            entering_index =
                phase2::phase2_ratio_test(lp, settings, vstatus, nonbasic_list, z, delta_z,
                                          step_length, nonbasic_entering_index);
        }
        if (entering_index == -2) {
            return dual::status_t::TIME_LIMIT;
        }
        if (entering_index == -3) {
            return dual::status_t::CONCURRENT_LIMIT;
        }
        if (entering_index == -1) {
            settings.log.printf("No entering variable found. "
                                "Iter %d\n",
                                iter);
            settings.log.printf("Scaled infeasibility %e\n", max_val);
            f_t perturbation = phase2::amount_of_perturbation(lp, objective);

            if (perturbation > 0.0 && phase == 2) {
                // Try to remove perturbation
                std::vector<f_t> unperturbed_y(m);
                std::vector<f_t> unperturbed_z(n);
                f_t *d_unperturbed_y, *d_unperturbed_z;
                CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_y, m * sizeof(f_t)),
                                    "cudaMalloc unperturbed_y");
                CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_z, n * sizeof(f_t)),
                                    "cudaMalloc unperturbed_z");
                phase2_cu::compute_dual_solution_from_basis(
                    d_objective, d_A_col_ptr, d_A_row_ind, d_A_values, cusparse_handle,
                    d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, d_basic_list,
                    d_nonbasic_list, d_unperturbed_y, d_unperturbed_z, m, n);
                CUDA_CALL_AND_CHECK(cudaMemcpy(unperturbed_y.data(), d_unperturbed_y,
                                               m * sizeof(f_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy unperturbed_y to host");
                CUDA_CALL_AND_CHECK(cudaMemcpy(unperturbed_z.data(), d_unperturbed_z,
                                               n * sizeof(f_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy unperturbed_z to host");
                CUDA_CALL_AND_CHECK(cudaFree(d_unperturbed_y), "cudaFree d_unperturbed_y");
                CUDA_CALL_AND_CHECK(cudaFree(d_unperturbed_z), "cudaFree d_unperturbed_z");
                {
                    const f_t dual_infeas =
                        phase2::dual_infeasibility(lp, settings, vstatus, unperturbed_z,
                                                   settings.tight_tol, settings.dual_tol);
                    settings.log.printf("Dual infeasibility "
                                        "after removing "
                                        "perturbation %e\n",
                                        dual_infeas);
                    f_t *d_unperturbed_x;
                    CUDA_CALL_AND_CHECK(cudaMalloc(&d_unperturbed_x, n * sizeof(f_t)),
                                        "cudaMalloc unperturbed_x");
                    if (dual_infeas <= settings.dual_tol) {
                        settings.log.printf("Removed "
                                            "perturbation of "
                                            "%.2e.\n",
                                            perturbation);
                        z = unperturbed_z;
                        y = unperturbed_y;
                        perturbation = 0.0;

                        std::vector<f_t> unperturbed_x(n);

                        phase2_cu::compute_primal_solution_from_basis(
                            lp, d_lp_rhs, d_A_col_ptr, d_A_row_ind, d_A_values, cusparse_handle,
                            d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                            d_basic_list, d_nonbasic_list, vstatus, unperturbed_x, d_unperturbed_x,
                            m, n);
                        x = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                            lp, settings, basic_list, x, squared_infeasibilities,
                            infeasibility_indices);
                        settings.log.printf("Updated primal "
                                            "infeasibility: %e\n",
                                            primal_infeasibility);

                        objective = lp.objective;
                        // Need to reset the
                        // objective value, since
                        // we have recomputed x
                        obj = phase2::compute_perturbed_objective(objective, x);
                        if (dual_infeas <= settings.dual_tol &&
                            primal_infeasibility <= settings.primal_tol) {
                            phase2_cu::prepare_optimality(
                                lp, settings, d_objective, d_A_col_ptr, d_A_row_ind, d_A_values,
                                cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                d_B_pinv_values, nz_B_pinv, lp.objective, basic_list, d_basic_list,
                                nonbasic_list, d_nonbasic_list, vstatus, phase, start_time, max_val,
                                iter, x, y, z, sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf("Continuing with "
                                            "perturbation "
                                            "removed and "
                                            "steepest edge norms "
                                            "reset\n");
                        // Clear delta_z before
                        // restarting the
                        // iteration
                        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark,
                                              delta_z_indices, delta_z);
                        continue;
                    } else {
                        std::vector<f_t> unperturbed_x(n);
                        phase2_cu::compute_primal_solution_from_basis(
                            lp, d_lp_rhs, d_A_col_ptr, d_A_row_ind, d_A_values, cusparse_handle,
                            d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv,
                            d_basic_list, d_nonbasic_list, vstatus, unperturbed_x, d_unperturbed_x,
                            m, n);
                        x = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                            lp, settings, basic_list, x, squared_infeasibilities,
                            infeasibility_indices);

                        const f_t orig_dual_infeas = phase2::dual_infeasibility(
                            lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);

                        if (primal_infeasibility <= settings.primal_tol &&
                            orig_dual_infeas <= settings.dual_tol) {
                            phase2_cu::prepare_optimality(
                                lp, settings, d_objective, d_A_col_ptr, d_A_row_ind, d_A_values,
                                cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                d_B_pinv_values, nz_B_pinv, objective, basic_list, d_basic_list,
                                nonbasic_list, d_nonbasic_list, vstatus, phase, start_time, max_val,
                                iter, x, y, z, sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf("Failed to remove "
                                            "perturbation of "
                                            "%.2e.\n",
                                            perturbation);
                    }
                }
            }

            const f_t dual_infeas = phase2::dual_infeasibility(
                lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
            settings.log.printf("Dual infeasibility %e\n", dual_infeas);
            const f_t primal_inf = phase2::primal_infeasibility(lp, settings, vstatus, x);
            settings.log.printf("Primal infeasibility %e\n", primal_inf);
            settings.log.printf("Steepest edge %e\n", max_val);
            if (dual_infeas > settings.dual_tol) {
                settings.log.printf("Numerical issues "
                                    "encountered. No entering "
                                    "variable found with "
                                    "large infeasibility.\n");
                return dual::status_t::NUMERICAL;
            }
            return dual::status_t::DUAL_UNBOUNDED;
        }

        timers.start_timer();
        // Update dual variables
        // y <- y + steplength * delta_y
        // z <- z + steplength * delta_z
        // phase2_cu::update_dual_variables(d_delta_y_sparse_indices, d_delta_y_sparse_values,
        // nz_delta_y_sparse, d_delta_z_sparse_indices, d_delta_z_sparse_values, nz_delta_z_sparse,
        // step_length, leaving_index, d_y, d_z);
        phase2::update_dual_variables(delta_y_sparse, delta_z_indices, delta_z, step_length,
                                      leaving_index, y, z);

        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.i.data(), d_delta_y_sparse_indices,
        // nz_delta_y_sparse * sizeof(i_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_y_sparse.i");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_y_sparse.x.data(), d_delta_y_sparse_values,
        // nz_delta_y_sparse * sizeof(f_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_y_sparse.x");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_z_indices.data(), d_delta_z_sparse_indices,
        // nz_delta_z_sparse * sizeof(i_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_z_indices");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(delta_z.data(), d_delta_z_sparse_values, nz_delta_z_sparse
        // * sizeof(f_t), cudaMemcpyDeviceToHost), "cudaMemcpy delta_z");
        // CUDA_CALL_AND_CHECK(cudaMemcpy(y.data(), d_y, sizeof(y), cudaMemcpyDeviceToHost),
        // "cudaMemcpy y"); CUDA_CALL_AND_CHECK(cudaMemcpy(z.data(), d_z, sizeof(z),
        // cudaMemcpyDeviceToHost), "cudaMemcpy z");

        // z[leaving_index] += step_length * delta_z[leaving_index];
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // Update primal variable
        const i_t num_flipped = phase2::flip_bounds(
            lp, settings, bounded_variables, objective, z, delta_z_indices, nonbasic_list,
            entering_index, vstatus, delta_x_flip, atilde_mark, atilde, atilde_index);

        timers.flip_time += timers.stop_timer();

        sparse_vector_t<i_t, f_t> delta_xB_0_sparse(m, 0);
        if (num_flipped > 0) {
            timers.start_timer();
            phase2_cu::adjust_for_flips(cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                        d_B_pinv_values, nz_B_pinv, basic_list, delta_z_indices,
                                        atilde_index, atilde, atilde_mark, delta_xB_0_sparse,
                                        delta_x_flip, x);
            timers.ftran_time += timers.stop_timer();
        }

        timers.start_timer();
        sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
        sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
        if (phase2_cu::compute_delta_x(lp, cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                       d_B_pinv_values, nz_B_pinv, entering_index, leaving_index,
                                       basic_leaving_index, direction, basic_list, delta_x_flip,
                                       rhs_sparse, x, scaled_delta_xB_sparse, delta_x, m) == -1) {
            settings.log.printf("Failed to compute delta_x. Iter "
                                "%d\n",
                                iter);
            return dual::status_t::NUMERICAL;
        }

        timers.ftran_time += timers.stop_timer();

        timers.start_timer();
        const i_t steepest_edge_status = phase2_cu::update_steepest_edge_norms(
            settings, basic_list, cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind,
            d_B_pinv_values, nz_B_pinv, direction, d_delta_y_sparse_indices,
            d_delta_y_sparse_values, nz_delta_y_sparse, steepest_edge_norm_check,
            scaled_delta_xB_sparse, basic_leaving_index, entering_index, v, delta_y_steepest_edge,
            m);
        assert(steepest_edge_status == 0);
        timers.se_norms_time += timers.stop_timer();

        timers.start_timer();
        // x <- x + delta_x
        phase2::update_primal_variables(scaled_delta_xB_sparse, basic_list, delta_x, entering_index,
                                        x);
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // TODO(CMM): Do I also need to update the
        // objective due to the bound flips?
        // TODO(CMM): I'm using the unperturbed
        // objective here, should this be the
        // perturbed objective?
        phase2::update_objective(basic_list, scaled_delta_xB_sparse.i, lp.objective, delta_x,
                                 entering_index, obj);
        timers.objective_time += timers.stop_timer();

        timers.start_timer();
        // Update primal infeasibilities due to
        // changes in basic variables from
        // flipping bounds
        phase2::update_primal_infeasibilities(
            lp, settings, basic_list, x, entering_index, leaving_index, delta_xB_0_sparse.i,
            squared_infeasibilities, infeasibility_indices, primal_infeasibility);
        // Update primal infeasibilities due to
        // changes in basic variables from the
        // leaving and entering variables
        phase2::update_primal_infeasibilities(
            lp, settings, basic_list, x, entering_index, leaving_index, scaled_delta_xB_sparse.i,
            squared_infeasibilities, infeasibility_indices, primal_infeasibility);
        // Update the entering variable
        phase2::update_single_primal_infeasibility(lp.lower, lp.upper, x, settings.primal_tol,
                                                   squared_infeasibilities, infeasibility_indices,
                                                   entering_index, primal_infeasibility);

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

        CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "Before basis update");
        // Update basic and nonbasic lists and
        // marks

        basic_list[basic_leaving_index] = entering_index;
        nonbasic_list[nonbasic_entering_index] = leaving_index;
        nonbasic_mark[entering_index] = -1;
        nonbasic_mark[leaving_index] = nonbasic_entering_index;
        basic_mark[leaving_index] = -1;
        basic_mark[entering_index] = basic_leaving_index;
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_basic_list + basic_leaving_index, &entering_index,
                                       sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy entering_index to "
                            "d_basic_list");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_nonbasic_list + nonbasic_entering_index, &leaving_index,
                                       sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy leaving_index to "
                            "d_nonbasic_list");

        timers.start_timer();
        // Refactor or update the basis
        // factorization
        bool should_refactor = (iter + 1) % settings.refactor_frequency == 0;

        if (!should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Update 1");
            }

            // should_refactor = !phase2_cu::eta_update_inverse(
            //     cublas_handle, m, d_B_pinv, eta_b_old, eta_b_new, eta_v, eta_c, eta_d,
            //     d_A_col_ptr, d_A_row_ind, d_A_values, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values,
            //     basic_leaving_index, entering_index);
            should_refactor = !phase2_cu::eta_update_inverse(
                cublas_handle, m, B_pinv_cusparse, d_B_pinv_col_ptr, d_B_pinv_row_ind,
                d_B_pinv_values, d_B_pinv_col_ptr_buffer, d_B_pinv_row_ind_buffer, d_B_pinv_values_buffer, nz_B_pinv, len_B_pinv, len_B_pinv_buffer, eta_b_old, eta_b_new, eta_v, eta_c, eta_d, d_A_col_ptr,
                d_A_row_ind, d_A_values, d_B_col_ptr, d_B_row_ind, d_B_values, basic_leaving_index,
                entering_index, settings);

            if (settings.profile) {
                settings.timer.stop("Inverse Update 1");
            }
        }

        // Free old B and Bt and recompute
        CUDA_CALL_AND_CHECK(cudaFree(d_B_row_ind), "cudaFree d_B_col_ptr");
        CUDA_CALL_AND_CHECK(cudaFree(d_B_values), "cudaFree d_B_values");

        if (!should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Update 2");
            }

            // Move basic list to device
            // TODONE: as above consider keeping
            // this on device i_t *d_basic_list;
            // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list,
            // m * sizeof(i_t)),
            //                     "cudaMalloc
            //                     d_basic_list");
            // CUDA_CALL_AND_CHECK(cudaMemcpy(d_basic_list,
            // basic_list.data(), m * sizeof(i_t),
            //                                cudaMemcpyHostToDevice),
            //                     "cudaMemcpy to
            //                     d_basic_list");

            // TODO: It's probably not smart to rebuild B and Bt from scratch every time
            // Instead use a 95% threshold (or higher) to determine the size per row to
            // allocate s.t. we don't have to realloc every time
            phase2_cu::build_basis_on_device(m, d_A_col_ptr, d_A_row_ind, d_A_values, d_basic_list,
                                             d_B_col_ptr, d_B_row_ind, d_B_values, nz_B);

            if (settings.profile) {
                settings.timer.stop("Inverse Update 2");
            }
        }

        if (should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Refactorizaton");
            }

            // Recompute d_B_pinv
            phase2_cu::compute_inverse<i_t, f_t>(
                cusparse_handle, cudss_handle, cudss_config, m, n, d_A_col_ptr, d_A_row_ind,
                d_A_values, d_B_col_ptr, d_B_row_ind, d_B_values, basic_list, d_B_pinv_col_ptr,
                d_B_pinv_row_ind, d_B_pinv_values, nz_B, nz_B_pinv, len_B_pinv, settings);

            CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(B_pinv_cusparse),
                                    "cusparseDestroyMatDescr B_pinv");
            CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&B_pinv_cusparse, m, m, nz_B_pinv,
                                                            d_B_pinv_col_ptr, d_B_pinv_row_ind,
                                                            d_B_pinv_values,
                                                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                            CUSPARSE_INDEX_BASE_ZERO,
                                                            CUDA_R_64F),
                                    "cusparseCreateCsc B_pinv");

            phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);
            phase2::compute_initial_primal_infeasibilities(
                lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

            if (settings.profile) {
                settings.timer.stop("Inverse Refactorizaton");
            }
        }
        timers.lu_update_time += timers.stop_timer();

        timers.start_timer();
        phase2_cu::compute_steepest_edge_norm_entering(
            settings, m, cusparse_handle, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values,
            nz_B_pinv, basic_leaving_index, entering_index, delta_y_steepest_edge);
        timers.se_entering_time += timers.stop_timer();

        iter++;

        // Clear delta_z
        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices,
                              delta_z);

        f_t now = toc(start_time);
        if ((iter - start_iter) < settings.first_iteration_log ||
            (iter % settings.iteration_log_frequency) == 0) {
            if (phase == 1 && iter == 1) {
                settings.log.printf(" Iter     Objective         "
                                    "  Num Inf.  Sum Inf.     "
                                    "Perturb  Time\n");
            }
            settings.log.printf("%5d %+.16e %7d %.8e %.2e %.2f\n", iter,
                                compute_user_objective(lp, obj), infeasibility_indices.size(),
                                primal_infeasibility, sum_perturb, now);
        }

        if (obj >= settings.cut_off) {
            settings.log.printf("Solve cutoff. Current objective "
                                "%e. Cutoff %e\n",
                                obj, settings.cut_off);
            return dual::status_t::CUTOFF;
        }

        if (now > settings.time_limit) {
            return dual::status_t::TIME_LIMIT;
        }

        if (settings.concurrent_halt != nullptr &&
            settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
            return dual::status_t::CONCURRENT_LIMIT;
        }

        // End of iteration cleanup
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_y_sparse_indices), "cudaFree d_delta_y_sparse_indices");
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_y_sparse_values), "cudaFree d_delta_y_sparse_values");
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_z_sparse_indices), "cudaFree d_delta_z_sparse_indices");
        CUDA_CALL_AND_CHECK(cudaFree(d_delta_z_sparse_values), "cudaFree d_delta_z_sparse_values");
    }
    if (iter >= iter_limit) {
        status = dual::status_t::ITERATION_LIMIT;
    }

    if (phase == 2) {
        timers.print_timers(settings);
    }

    // Cleanup GPU resources
    CUDA_CALL_AND_CHECK(cudaFree(d_A_col_ptr), "cudaFree d_A_col_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_A_row_ind), "cudaFree d_A_row_ind");
    CUDA_CALL_AND_CHECK(cudaFree(d_A_values), "cudaFree d_A_values");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_col_ptr), "cudaFree d_B_col_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_values), "cudaFree d_B_values");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_col_ptr), "cudaFree d_B_pinv_col_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_row_ind), "cudaFree d_B_pinv_row_ind");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_values), "cudaFree d_B_pinv_values");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_col_ptr_buffer), "cudaFree d_B_pinv_col_ptr_buffer");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_row_ind_buffer), "cudaFree d_B_pinv_row_ind_buffer");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_values_buffer), "cudaFree d_B_pinv_values_buffer");
    CUBLAS_CALL_AND_CHECK(cublasDestroy(cublas_handle), "cublasDestroy");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroy(cusparse_handle), "cusparseDestroy");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(B_pinv_cusparse),
                            "cusparseDestroyMatDescr B_pinv");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroy(cusparse_pinv_handle), "cusparseDestroy Pinv");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(cudss_config), "cudssConfigDestroy B");
    CUDSS_CALL_AND_CHECK(cudssDestroy(cudss_handle), "cudssDestroyHandle B");

    return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template cuopt::linear_programming::dual_simplex::dual::status_t dual_phase2_cu<int, double>(
    int phase, int slack_basis, double start_time, const lp_problem_t<int, double> &lp,
    const simplex_solver_settings_t<int, double> &settings, std::vector<variable_status_t> &vstatus,
    lp_solution_t<int, double> &sol, int &iter, std::vector<double> &steepest_edge_norms);

template cuopt::linear_programming::dual_simplex::dual::status_t dual_phase2_cu_parallel_pivot<int, double>(
    int phase, int slack_basis, double start_time, const lp_problem_t<int, double> &lp,
    const simplex_solver_settings_t<int, double> &settings, std::vector<variable_status_t> &vstatus,
    lp_solution_t<int, double> &sol, int &iter, std::vector<double> &steepest_edge_norms);

#endif

} // namespace
  // cuopt::linear_programming::dual_simplex
