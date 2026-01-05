#include "cudss.h"
#include "cusparse.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <driver_types.h>
#include <dual_simplex/phase2_dual.cuh>
#include <dual_simplex/problem_analysis.hpp>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

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
                     i_t *&d_X_col_ptr, i_t *&d_X_row_ind, f_t *&d_X_values, i_t &nz_B, i_t &nz_X,
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

    // Factor (B^T B) = L*U using cuDSS and solve for (B^T B) X = B^T
    // => X = (B^T B)^(-1) B^T is the pseudo-inverse of B

    // We use slices to create X in CSC
    // Surprisingly, X is quite sparse for many LP problems
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

    CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_col_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_X_col_ptr");
    std::vector<i_t> X_nnz_per_slice(settings.pinv_slices, 0);
    std::vector<f_t *> d_X_values_slices(settings.pinv_slices, nullptr);
    std::vector<i_t *> d_X_row_ind_slices(settings.pinv_slices, nullptr);
    for (i_t slice_idx = 0; slice_idx < settings.pinv_slices; ++slice_idx) {
        // Determine the number of columns in this slice
        i_t cols_in_slice = slice_idx < m % settings.pinv_slices ? m / settings.pinv_slices + 1
                                                                 : m / settings.pinv_slices;
        i_t col_start = (m / settings.pinv_slices) * slice_idx +
                        std::min<i_t>(slice_idx, m % settings.pinv_slices);

        // Interpret rows of B in CSR as columns of B_T in CSC
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
        CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cuDSS solve");

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
        // Allocate row indices and values for this slice
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_row_ind_slices[slice_idx], nnz_X_slice * sizeof(i_t)),
                            "cudaMalloc d_X_row_ind for slice");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_values_slices[slice_idx], nnz_X_slice * sizeof(f_t)),
                            "cudaMalloc d_X_values for slice");
        // Fill CSC structure for this slice
        i_t *d_write_offsets;
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_write_offsets, cols_in_slice * sizeof(i_t)),
                            "cudaMalloc d_write_offsets for X slice");
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
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_row_ind, nz_B_pinv * sizeof(i_t)),
                        "cudaMalloc d_X_row_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_X_values, nz_B_pinv * sizeof(f_t)), "cudaMalloc d_X_values");
    i_t offset = 0;
    for (i_t slice_idx = 0; slice_idx < settings.pinv_slices; ++slice_idx) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_X_row_ind + offset, d_X_row_ind_slices[slice_idx],
                                       X_nnz_per_slice[slice_idx] * sizeof(i_t),
                                       cudaMemcpyDeviceToDevice),
                            "cudaMemcpy d_X_row_ind slice to final");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_X_values + offset, d_X_values_slices[slice_idx],
                                       X_nnz_per_slice[slice_idx] * sizeof(f_t),
                                       cudaMemcpyDeviceToDevice),
                            "cudaMemcpy d_X_values slice to final");
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
    // Note: d_X is not freed here - caller will free it after use
    // d_B and d_Bt are freed after inverse update later
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
                           const f_t *d_B_values, f_t *col_dense, cudaStream_t stream) {
    // Initialize to zero
    CUDA_CALL_AND_CHECK(cudaMemset(col_dense, 0, m * sizeof(f_t)), "cudaMemset col_dense");
    int block_size = 32;
    int grid_size = (m + block_size - 1) / block_size;
    fetch_column_as_dense_kernel<<<grid_size, block_size, 0, stream>>>(
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

    // Count new nnz due to U
    for (i_t u_idx = 0; u_idx < m; ++u_idx) {
        f_t u_val = d_vector_U_values[u_idx];
        if (abs(v_val) * abs(u_val) * abs(scale) > 1e-12) {
            i_t row_idx = u_idx;
            // Binary search to check if row_idx exists in B_pinv column
            i_t res = binary_search(num_existing_rows, &d_B_pinv_row_ind[col_start], row_idx);
            if (res == -1) {
                nnz_in_col += 1;
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
}

template <typename i_t, typename f_t>
bool eta_update_inverse(cublasHandle_t cublas_handle, cusparseHandle_t cusparse_handle, i_t m, cusparseSpMatDescr_t &B_pinv_cusparse,
                        i_t *&d_B_pinv_col_ptr, i_t *&d_B_pinv_row_ind, f_t *&d_B_pinv_values,
                        i_t &nz_B_pinv, f_t *eta_b_old, f_t *eta_b_new, f_t *eta_v, f_t *eta_c,
                        f_t *eta_d, const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
                        const f_t *d_A_values, const i_t *d_B_col_ptr, const i_t *d_B_row_ind,
                        const f_t *d_B_values, i_t basic_leaving_index, i_t entering_index) {
    const i_t j = basic_leaving_index; // Index of leaving variable in basis

    cudaStream_t stream1, stream2;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream1), "cudaStreamCreate stream1");
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream2), "cudaStreamCreate stream2");
    // Fetch b_old and b_new as dense vectors
    phase2_cu::fetch_column_as_dense(m, basic_leaving_index, d_B_col_ptr, d_B_row_ind, d_B_values,
                                     eta_b_old, stream1);
    phase2_cu::fetch_column_as_dense(m, entering_index, d_A_col_ptr, d_A_row_ind, d_A_values,
                                     eta_b_new, stream2);
    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream1), "cudaStreamDestroy stream1");
    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream2), "cudaStreamDestroy stream2");

    // Sherman-Morrison formula for updating B_inv when replacing column j:
    // B_new = B with column j replaced by b_new
    // B_new_inv = B_inv - (B_inv @ u @ e_j^T @ B_inv) / (1 + e_j^T @ B_inv @ u)
    // where u = b_new - b_old
    //
    // This simplifies to:
    // p = B_inv @ (b_new - b_old)
    // row_j = j-th row of B_inv
    // B_new_inv = B_inv - (p @ row_j) / (1 + p[j])

    f_t alpha, beta;

    // Compute u = b_new - b_old (store in eta_c temporarily)
    CUDA_CALL_AND_CHECK(cudaMemcpy(eta_c, eta_b_new, m * sizeof(f_t), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy eta_c = b_new");
    alpha = -1.0;
    CUBLAS_CALL_AND_CHECK(cublasDaxpy(cublas_handle, m, &alpha, eta_b_old, 1, eta_c, 1),
                          "cublasDaxpy compute u = b_new - b_old");

    // Compute p = B_inv @ u
    // Note: eta_p already contains B_inv @ b_old from the caller
    // So we need to recompute: p = B_inv @ (b_new - b_old)
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
    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "spmv_csc_kernel sync");

    // Get p[j] and compute denominator: 1 + p[j]
    f_t p_j;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&p_j, eta_v + j, sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy p_j from device");
    f_t denom = 1.0 + p_j;

    if (std::abs(denom) < 1e-10) {
        // Singular or near-singular update - should refactor instead
        return false;
    }

    // Kernel parameters
    block_size = 256;
    grid_size = (m + block_size - 1) / block_size;

    // Extract row j of B_inv (store in eta_d)
    // B_inv is column-major, so row j is at offsets j, j+m, j+2m, ...
    phase2_cu::fetch_row_as_dense(m, j, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, eta_d,
                                  0);

    // B_inv = B_inv - (p @ row_j) / denom
    f_t scale = -1.0 / denom;
    // Custom kernel to perform sparse rank-1 update
    // 1. Count the additional number of non-zeros introduced
    i_t *d_new_B_pinv_col_ptr;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_new_B_pinv_col_ptr, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_new_B_pinv_col_ptr");
    CUDA_CALL_AND_CHECK(cudaMemset(d_new_B_pinv_col_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_new_B_pinv_col_ptr");
    grid_size = (m + block_size - 1) / block_size;
    rank1_symbolic<<<grid_size, block_size>>>(m, d_B_pinv_col_ptr, d_B_pinv_row_ind, eta_v, eta_d,
                                              d_new_B_pinv_col_ptr, scale);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "get_rank1_delta_nnz_nnz kernel");
    // 2. Allocate new arrays for updated B_pinv
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_new_B_pinv_col_ptr);
    thrust::exclusive_scan(thrust::cuda::par, dev_ptr, dev_ptr + m + 1, dev_ptr, i_t(0));
    i_t nz_new_B_pinv = 0;
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(&nz_new_B_pinv, d_new_B_pinv_col_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy nz_new_B_pinv to host");
    i_t *d_new_B_pinv_row_ind;
    f_t *d_new_B_pinv_values;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_new_B_pinv_row_ind, nz_new_B_pinv * sizeof(i_t)),
                        "cudaMalloc d_new_B_pinv_row_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_new_B_pinv_values, nz_new_B_pinv * sizeof(f_t)),
                        "cudaMalloc d_new_B_pinv_values");
    // 3. Launch kernel to perform the rank-1 update and fill new arrays
    grid_size = (m + block_size - 1) / block_size;
    rank1_update<<<grid_size, block_size>>>(m, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values,
                                            d_new_B_pinv_col_ptr, d_new_B_pinv_row_ind,
                                            d_new_B_pinv_values, eta_v, eta_d, scale);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "rank1_csc_update kernel");
    // 4. Clean up and update pointers
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_col_ptr), "cudaFree old d_B_pinv_col_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_row_ind), "cudaFree old d_B_pinv_row_ind");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv_values), "cudaFree old d_B_pinv_values");
    d_B_pinv_col_ptr = d_new_B_pinv_col_ptr;
    d_B_pinv_row_ind = d_new_B_pinv_row_ind;
    d_B_pinv_values = d_new_B_pinv_values;
    nz_B_pinv = nz_new_B_pinv;
    // Update cusparse matrix descriptor
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(B_pinv_cusparse),
                            "cusparseDestroySpMat B_pinv_cusparse");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&B_pinv_cusparse, m, m, nz_B_pinv, d_B_pinv_col_ptr,
                                              d_B_pinv_row_ind, d_B_pinv_values, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F),
                            "cusparseCreateCSCMat B_pinv_cusparse");

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
            // We want (B_inv^T)_{idx, vec_idx} = (B_inv)_{vec_idx, idx}
            // B_inv is col-major, so element (row=vec_idx, col=idx) is at:
            // col * m + row = idx * m + vec_idx
            mat_idx = idx * m + vec_idx;
        } else {
            // We want (B_inv)_{idx, vec_idx}
            // B_inv is col-major, so element (row=idx, col=vec_idx) is at:
            // col * m + row = vec_idx * m + idx
            mat_idx = vec_idx * m + idx;
        }

        sum += vec_val * b_inv[mat_idx];
    }
    result[idx] = sum;
}

template <typename i_t, typename f_t>
void pinv_solve(cublasHandle_t &cublas_handle, f_t *d_B_pinv, const std::vector<f_t> &rhs,
                std::vector<f_t> &x, i_t m, bool transpose) {
    std::cout << "pinv dense solve gpu\n" << std::endl;
    // const f_t alpha = 1.0;
    // const f_t beta = 0.0;
    // cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    // CUBLAS_CALL_AND_CHECK(
    //     cublasDgemv(cublas_handle, op, m, m, &alpha, d_B_pinv, m, d_rhs, 1, &beta, d_x, 1),
    //     "cublasSgemv pinv_solve");
    // fake indices and values for dense vector
    std::vector<i_t> fake_indices(m);
    for (i_t i = 0; i < m; ++i) {
        fake_indices[i] = i;
    }
    i_t *d_rhs_indices;
    f_t *d_rhs_values;
    f_t *d_x;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_x, m * sizeof(f_t)), "cudaMalloc d_x");

    // Copy sparse rhs to dense d_rhs
    // std::vector<f_t> h_rhs_dense(m, 0.0);
    const i_t nz_rhs = rhs.size();
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs_indices, nz_rhs * sizeof(f_t)),
                        "cudaMalloc d_rhs indices");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_rhs_indices, fake_indices.data(), nz_rhs * sizeof(i_t),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy rhs indices to d_rhs_indices");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs_values, nz_rhs * sizeof(f_t)), "cudaMalloc d_rhs values");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_rhs_values, rhs.data(), nz_rhs * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy rhs values to d_rhs");
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    denseMatrixSparseVectorMulKernel<<<grid_size, block_size>>>(m, d_B_pinv, d_rhs_indices,
                                                                d_rhs_values, m, d_x, transpose);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "denseMatrixSparseVectorMulKernel");
    CUDA_CALL_AND_CHECK(cudaMemcpy(x.data(), d_x, m * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy d_x to x");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs_indices), "cudaFree d_rhs indices");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs_values), "cudaFree d_rhs values");
    CUDA_CALL_AND_CHECK(cudaFree(d_x), "cudaFree d_x");
}

template <typename i_t, typename f_t>
void pinv_solve(cublasHandle_t &cublas_handle, f_t *d_B_pinv, const sparse_vector_t<i_t, f_t> &rhs,
                sparse_vector_t<i_t, f_t> &x, i_t m, bool transpose) {
    std::cout << "pinv solve gpu\n";
    i_t *d_rhs_indices;
    f_t *d_rhs_values;
    f_t *d_x;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_x, m * sizeof(f_t)), "cudaMalloc d_x");

    // Copy sparse rhs to dense d_rhs
    // std::vector<f_t> h_rhs_dense(m, 0.0);
    const i_t nz_rhs = rhs.i.size();
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs_indices, nz_rhs * sizeof(f_t)),
                        "cudaMalloc d_rhs indices");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_rhs_indices, rhs.i.data(), nz_rhs * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy rhs indices to d_rhs_indices");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs_values, nz_rhs * sizeof(f_t)), "cudaMalloc d_rhs values");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_rhs_values, rhs.x.data(), nz_rhs * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy rhs values to d_rhs");

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    denseMatrixSparseVectorMulKernel<<<grid_size, block_size>>>(
        m, d_B_pinv, d_rhs_indices, d_rhs_values, nz_rhs, d_x, transpose);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "denseMatrixSparseVectorMulKernel");

    // Compute d_x = B_pinv * d_rhs

    // const f_t alpha = 1.0;
    // const f_t beta = 0.0;
    // cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    // CUBLAS_CALL_AND_CHECK(
    //     cublasDgemv(cublas_handle, op, m, m, &alpha, d_B_pinv, m, d_rhs, 1, &beta, d_x, 1),
    //     "cublasSgemv pinv_solve");
    //
    // // Copy dense d_x to sparse x
    std::vector<f_t> h_x_dense(m);
    CUDA_CALL_AND_CHECK(cudaMemcpy(h_x_dense.data(), d_x, m * sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy d_x to x");
    x.i.clear();
    x.x.clear();
    for (i_t i = 0; i < m; ++i) {
        if (std::abs(h_x_dense[i]) > 1e-12) {
            x.i.push_back(i);
            x.x.push_back(h_x_dense[i]);
        }
    }

    CUDA_CALL_AND_CHECK(cudaFree(d_rhs_indices), "cudaFree d_rhs indices");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs_values), "cudaFree d_rhs values");
    CUDA_CALL_AND_CHECK(cudaFree(d_x), "cudaFree d_x");
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

    // Analyze matrix A
    problem_analyzer_t<i_t, f_t> analyzer(lp, settings);
    analyzer.analyze();
    analyzer.display_analysis();

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
    // TODO: remove once we have dense sparse mv on device
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

    f_t *d_B_pinv;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv, m * m * sizeof(f_t)), "cudaMalloc d_B_pinv");
    i_t *d_B_pinv_col_ptr;
    i_t *d_B_pinv_row_ind;
    f_t *d_B_pinv_values;
    i_t nz_B_pinv;
    phase2_cu::compute_inverse<i_t, f_t>(
        cusparse_handle, cudss_handle, cudss_config, m, n, d_A_col_ptr, d_A_row_ind, d_A_values,
        d_B_col_ptr, d_B_row_ind, d_B_values, basic_list, d_B_pinv_col_ptr, d_B_pinv_row_ind,
        d_B_pinv_values, nz_B, nz_B_pinv, settings);

    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsc(&B_pinv_cusparse, m, m, nz_B_pinv, d_B_pinv_col_ptr,
                                              d_B_pinv_row_ind, d_B_pinv_values, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F),
                            "cusparseCreateCsc B_pinv_cusparse");
    {
        // Debug: copy B_pinv to host, construct as dense, then copy back to d_B_pinv
        std::vector<f_t> h_B_pinv_dense(m * m, 0.0);
        std::vector<i_t> h_B_pinv_col_ptr(m + 1);
        std::vector<i_t> h_B_pinv_row_ind(nz_B_pinv);
        std::vector<f_t> h_B_pinv_values(nz_B_pinv);
        CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_col_ptr.data(), d_B_pinv_col_ptr,
                                       (m + 1) * sizeof(i_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy d_B_pinv_col_ptr to host");
        CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_row_ind.data(), d_B_pinv_row_ind,
                                       nz_B_pinv * sizeof(i_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy d_B_pinv_row_ind to host");
        CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_values.data(), d_B_pinv_values,
                                       nz_B_pinv * sizeof(f_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy d_B_pinv_values to host");
        for (i_t col = 0; col < m; ++col) {
            i_t start = h_B_pinv_col_ptr[col];
            i_t end = h_B_pinv_col_ptr[col + 1];
            for (i_t idx = start; idx < end; ++idx) {
                i_t row = h_B_pinv_row_ind[idx];
                f_t val = h_B_pinv_values[idx];
                h_B_pinv_dense[col * m + row] = val; // Column-major
            }
        }
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_pinv, h_B_pinv_dense.data(), m * m * sizeof(f_t),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy h_B_pinv_dense to d_B_pinv");
    }

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }
    std::vector<f_t> c_basic(m);
    for (i_t k = 0; k < m; ++k) {
        const i_t j = basic_list[k];
        c_basic[k] = objective[j];
    }

    // Solve B'*y = cB
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, c_basic, y, m, true);

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

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

    phase2::compute_primal_variables(cublas_handle, d_B_pinv, lp.rhs, lp.A, basic_list,
                                     nonbasic_list, settings.tight_tol, x);

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    if (delta_y_steepest_edge.size() == 0) {
        delta_y_steepest_edge.resize(n);
        if (slack_basis) {
            phase2::initialize_steepest_edge_norms_from_slack_basis(basic_list, nonbasic_list,
                                                                    delta_y_steepest_edge);
        } else {
            std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
            if (phase2::initialize_steepest_edge_norms(lp, settings, start_time, basic_list,
                                                       cublas_handle, d_B_pinv,
                                                       delta_y_steepest_edge) == -1) {
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

    f_t obj = compute_objective(lp, x);
    const i_t start_iter = iter;

    i_t sparse_delta_z = 0;
    i_t dense_delta_z = 0;
    phase2::phase2_timers_t<i_t, f_t> timers(settings.profile && phase == 2);

    while (iter < iter_limit) {
        // Pricing
        i_t direction = 0;
        i_t basic_leaving_index = -1;
        i_t leaving_index = -1;
        f_t max_val;
        timers.start_timer();
        if (settings.use_steepest_edge_pricing) {
            leaving_index = phase2::steepest_edge_pricing_with_infeasibilities(
                lp, settings, x, delta_y_steepest_edge, basic_mark, squared_infeasibilities,
                infeasibility_indices, direction, basic_leaving_index, max_val);
        } else {
            // Max infeasibility pricing
            leaving_index = phase2::phase2_pricing(lp, settings, x, basic_list, direction,
                                                   basic_leaving_index, primal_infeasibility);
        }
        timers.pricing_time += timers.stop_timer();
        if (leaving_index == -1) {
            phase2::prepare_optimality(lp, settings, cublas_handle, d_B_pinv, objective, basic_list,
                                       nonbasic_list, vstatus, phase, start_time, max_val, iter, x,
                                       y, z, sol);
            status = dual::status_t::OPTIMAL;
            break;
        }

        // BTran
        // BT*delta_y = -delta_zB = -sigma*ei
        timers.start_timer();
        sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
        phase2::compute_delta_y(cublas_handle, d_B_pinv, basic_leaving_index, direction,
                                delta_y_sparse);
        timers.btran_time += timers.stop_timer();

        const f_t steepest_edge_norm_check = delta_y_sparse.norm2_squared();
        if (delta_y_steepest_edge[leaving_index] <
            settings.steepest_edge_ratio * steepest_edge_norm_check) {
            constexpr bool verbose = false;
            if constexpr (verbose) {
                settings.log.printf(
                    "iteration restart due to steepest edge. Leaving %d. Actual %.2e "
                    "from update %.2e\n",
                    leaving_index, steepest_edge_norm_check, delta_y_steepest_edge[leaving_index]);
            }
            delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
            continue;
        }

        timers.start_timer();
        i_t delta_y_nz0 = 0;
        const i_t nz_delta_y = delta_y_sparse.i.size();
        for (i_t k = 0; k < nz_delta_y; k++) {
            if (std::abs(delta_y_sparse.x[k]) > 1e-12) {
                delta_y_nz0++;
            }
        }
        const f_t delta_y_nz_percentage = delta_y_nz0 / static_cast<f_t>(m) * 100.0;
        const bool use_transpose = delta_y_nz_percentage <= 30.0;
        if (use_transpose) {
            sparse_delta_z++;
            phase2::compute_delta_z(A_transpose, delta_y_sparse, leaving_index, direction,
                                    nonbasic_mark, delta_z_mark, delta_z_indices, delta_z);
        } else {
            dense_delta_z++;
            // delta_zB = sigma*ei
            delta_y_sparse.to_dense(delta_y);
            phase2::compute_reduced_cost_update(lp, basic_list, nonbasic_list, delta_y,
                                                leaving_index, direction, delta_z_mark,
                                                delta_z_indices, delta_z);
        }
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
            settings.log.printf("No entering variable found. Iter %d\n", iter);
            settings.log.printf("Scaled infeasibility %e\n", max_val);
            f_t perturbation = phase2::amount_of_perturbation(lp, objective);

            if (perturbation > 0.0 && phase == 2) {
                // Try to remove perturbation
                std::vector<f_t> unperturbed_y(m);
                std::vector<f_t> unperturbed_z(n);
                phase2::compute_dual_solution_from_basis(lp, cublas_handle, d_B_pinv, basic_list,
                                                         nonbasic_list, unperturbed_y,
                                                         unperturbed_z);
                {
                    const f_t dual_infeas =
                        phase2::dual_infeasibility(lp, settings, vstatus, unperturbed_z,
                                                   settings.tight_tol, settings.dual_tol);
                    settings.log.printf("Dual infeasibility after removing perturbation %e\n",
                                        dual_infeas);
                    if (dual_infeas <= settings.dual_tol) {
                        settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
                        z = unperturbed_z;
                        y = unperturbed_y;
                        perturbation = 0.0;

                        std::vector<f_t> unperturbed_x(n);
                        phase2::compute_primal_solution_from_basis(lp, cublas_handle, d_B_pinv,
                                                                   basic_list, nonbasic_list,
                                                                   vstatus, unperturbed_x);
                        x = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                            lp, settings, basic_list, x, squared_infeasibilities,
                            infeasibility_indices);
                        settings.log.printf("Updated primal infeasibility: %e\n",
                                            primal_infeasibility);

                        objective = lp.objective;
                        // Need to reset the objective value, since we have recomputed x
                        obj = phase2::compute_perturbed_objective(objective, x);
                        if (dual_infeas <= settings.dual_tol &&
                            primal_infeasibility <= settings.primal_tol) {
                            phase2::prepare_optimality(lp, settings, cublas_handle, d_B_pinv,
                                                       objective, basic_list, nonbasic_list,
                                                       vstatus, phase, start_time, max_val, iter, x,
                                                       y, z, sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf(
                            "Continuing with perturbation removed and steepest edge norms reset\n");
                        // Clear delta_z before restarting the iteration
                        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark,
                                              delta_z_indices, delta_z);
                        continue;
                    } else {
                        std::vector<f_t> unperturbed_x(n);
                        phase2::compute_primal_solution_from_basis(lp, cublas_handle, d_B_pinv,
                                                                   basic_list, nonbasic_list,
                                                                   vstatus, unperturbed_x);
                        x = unperturbed_x;
                        primal_infeasibility = phase2::compute_initial_primal_infeasibilities(
                            lp, settings, basic_list, x, squared_infeasibilities,
                            infeasibility_indices);

                        const f_t orig_dual_infeas = phase2::dual_infeasibility(
                            lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);

                        if (primal_infeasibility <= settings.primal_tol &&
                            orig_dual_infeas <= settings.dual_tol) {
                            phase2::prepare_optimality(lp, settings, cublas_handle, d_B_pinv,
                                                       objective, basic_list, nonbasic_list,
                                                       vstatus, phase, start_time, max_val, iter, x,
                                                       y, z, sol);
                            status = dual::status_t::OPTIMAL;
                            break;
                        }
                        settings.log.printf("Failed to remove perturbation of %.2e.\n",
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
                settings.log.printf("Numerical issues encountered. No entering variable found with "
                                    "large infeasibility.\n");
                return dual::status_t::NUMERICAL;
            }
            return dual::status_t::DUAL_UNBOUNDED;
        }

        timers.start_timer();
        // Update dual variables
        // y <- y + steplength * delta_y
        // z <- z + steplength * delta_z
        phase2::update_dual_variables(delta_y_sparse, delta_z_indices, delta_z, step_length,
                                      leaving_index, y, z);
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
            phase2::adjust_for_flips(cublas_handle, d_B_pinv, basic_list, delta_z_indices,
                                     atilde_index, atilde, atilde_mark, delta_xB_0_sparse,
                                     delta_x_flip, x);
            timers.ftran_time += timers.stop_timer();
        }

        timers.start_timer();
        sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
        sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
        if (phase2::compute_delta_x(lp, cublas_handle, d_B_pinv, entering_index, leaving_index,
                                    basic_leaving_index, direction, basic_list, delta_x_flip,
                                    rhs_sparse, x, scaled_delta_xB_sparse, delta_x) == -1) {
            settings.log.printf("Failed to compute delta_x. Iter %d\n", iter);
            return dual::status_t::NUMERICAL;
        }

        timers.ftran_time += timers.stop_timer();

        timers.start_timer();
        const i_t steepest_edge_status = phase2::update_steepest_edge_norms(
            settings, basic_list, cublas_handle, d_B_pinv, direction, delta_y_sparse,
            steepest_edge_norm_check, scaled_delta_xB_sparse, basic_leaving_index, entering_index,
            v, delta_y_steepest_edge);
        assert(steepest_edge_status == 0);
        timers.se_norms_time += timers.stop_timer();

        timers.start_timer();
        // x <- x + delta_x
        phase2::update_primal_variables(scaled_delta_xB_sparse, basic_list, delta_x, entering_index,
                                        x);
        timers.vector_time += timers.stop_timer();

        timers.start_timer();
        // TODO(CMM): Do I also need to update the objective due to the bound flips?
        // TODO(CMM): I'm using the unperturbed objective here, should this be the perturbed
        // objective?
        phase2::update_objective(basic_list, scaled_delta_xB_sparse.i, lp.objective, delta_x,
                                 entering_index, obj);
        timers.objective_time += timers.stop_timer();

        timers.start_timer();
        // Update primal infeasibilities due to changes in basic variables
        // from flipping bounds
        phase2::update_primal_infeasibilities(
            lp, settings, basic_list, x, entering_index, leaving_index, delta_xB_0_sparse.i,
            squared_infeasibilities, infeasibility_indices, primal_infeasibility);
        // Update primal infeasibilities due to changes in basic variables
        // from the leaving and entering variables
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
        basic_list[basic_leaving_index] = entering_index;
        nonbasic_list[nonbasic_entering_index] = leaving_index;
        nonbasic_mark[entering_index] = -1;
        nonbasic_mark[leaving_index] = nonbasic_entering_index;
        basic_mark[leaving_index] = -1;
        basic_mark[entering_index] = basic_leaving_index;

        timers.start_timer();
        // Refactor or update the basis factorization
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
                cublas_handle, cusparse_pinv_handle, m, B_pinv_cusparse, d_B_pinv_col_ptr,
                d_B_pinv_row_ind, d_B_pinv_values, nz_B_pinv, eta_b_old, eta_b_new, eta_v, eta_c,
                eta_d, d_A_col_ptr, d_A_row_ind, d_A_values, d_B_col_ptr, d_B_row_ind, d_B_values,
                basic_leaving_index, entering_index);

            {
                // Debug: copy B_pinv to host, construct as dense, then copy back to d_B_pinv
                std::vector<f_t> h_B_pinv_dense(m * m, 0.0);
                std::vector<i_t> h_B_pinv_col_ptr(m + 1);
                std::vector<i_t> h_B_pinv_row_ind(nz_B_pinv);
                std::vector<f_t> h_B_pinv_values(nz_B_pinv);
                CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_col_ptr.data(), d_B_pinv_col_ptr,
                                               (m + 1) * sizeof(i_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy d_B_pinv_col_ptr to host");
                CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_row_ind.data(), d_B_pinv_row_ind,
                                               nz_B_pinv * sizeof(i_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy d_B_pinv_row_ind to host");
                CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_values.data(), d_B_pinv_values,
                                               nz_B_pinv * sizeof(f_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy d_B_pinv_values to host");
                for (i_t col = 0; col < m; ++col) {
                    i_t start = h_B_pinv_col_ptr[col];
                    i_t end = h_B_pinv_col_ptr[col + 1];
                    for (i_t idx = start; idx < end; ++idx) {
                        i_t row = h_B_pinv_row_ind[idx];
                        f_t val = h_B_pinv_values[idx];
                        h_B_pinv_dense[col * m + row] = val; // Column-major
                    }
                }
                CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_pinv, h_B_pinv_dense.data(), m * m * sizeof(f_t),
                                               cudaMemcpyHostToDevice),
                                    "cudaMemcpy h_B_pinv_dense to d_B_pinv");
            }

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
            // TODO: as above consider keeping this on device
            i_t *d_basic_list;
            CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)),
                                "cudaMalloc d_basic_list");
            CUDA_CALL_AND_CHECK(cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t),
                                           cudaMemcpyHostToDevice),
                                "cudaMemcpy to d_basic_list");

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
                d_B_pinv_row_ind, d_B_pinv_values, nz_B, nz_B_pinv, settings);

            {
                // Debug: copy B_pinv to host, construct as dense, then copy back to d_B_pinv
                std::vector<f_t> h_B_pinv_dense(m * m, 0.0);
                std::vector<i_t> h_B_pinv_col_ptr(m + 1);
                std::vector<i_t> h_B_pinv_row_ind(nz_B_pinv);
                std::vector<f_t> h_B_pinv_values(nz_B_pinv);
                CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_col_ptr.data(), d_B_pinv_col_ptr,
                                               (m + 1) * sizeof(i_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy d_B_pinv_col_ptr to host");
                CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_row_ind.data(), d_B_pinv_row_ind,
                                               nz_B_pinv * sizeof(i_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy d_B_pinv_row_ind to host");
                CUDA_CALL_AND_CHECK(cudaMemcpy(h_B_pinv_values.data(), d_B_pinv_values,
                                               nz_B_pinv * sizeof(f_t), cudaMemcpyDeviceToHost),
                                    "cudaMemcpy d_B_pinv_values to host");
                for (i_t col = 0; col < m; ++col) {
                    i_t start = h_B_pinv_col_ptr[col];
                    i_t end = h_B_pinv_col_ptr[col + 1];
                    for (i_t idx = start; idx < end; ++idx) {
                        i_t row = h_B_pinv_row_ind[idx];
                        f_t val = h_B_pinv_values[idx];
                        h_B_pinv_dense[col * m + row] = val; // Column-major
                    }
                }
                CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_pinv, h_B_pinv_dense.data(), m * m * sizeof(f_t),
                                               cudaMemcpyHostToDevice),
                                    "cudaMemcpy h_B_pinv_dense to d_B_pinv");
            }

            phase2::reset_basis_mark(basic_list, nonbasic_list, basic_mark, nonbasic_mark);
            phase2::compute_initial_primal_infeasibilities(
                lp, settings, basic_list, x, squared_infeasibilities, infeasibility_indices);

            if (settings.profile) {
                settings.timer.stop("Inverse Refactorizaton");
            }
        }
        timers.lu_update_time += timers.stop_timer();

        timers.start_timer();
        phase2::compute_steepest_edge_norm_entering(settings, m, cublas_handle, d_B_pinv,
                                                    basic_leaving_index, entering_index,
                                                    delta_y_steepest_edge);
        timers.se_entering_time += timers.stop_timer();

        iter++;

        // Clear delta_z
        phase2::clear_delta_z(entering_index, leaving_index, delta_z_mark, delta_z_indices,
                              delta_z);

        f_t now = toc(start_time);
        if ((iter - start_iter) < settings.first_iteration_log ||
            (iter % settings.iteration_log_frequency) == 0) {
            if (phase == 1 && iter == 1) {
                settings.log.printf(
                    " Iter     Objective           Num Inf.  Sum Inf.     Perturb  Time\n");
            }
            settings.log.printf("%5d %+.16e %7d %.8e %.2e %.2f\n", iter,
                                compute_user_objective(lp, obj), infeasibility_indices.size(),
                                primal_infeasibility, sum_perturb, now);
        }

        if (obj >= settings.cut_off) {
            settings.log.printf("Solve cutoff. Current objective %e. Cutoff %e\n", obj,
                                settings.cut_off);
            return dual::status_t::CUTOFF;
        }

        if (now > settings.time_limit) {
            return dual::status_t::TIME_LIMIT;
        }

        if (settings.concurrent_halt != nullptr &&
            settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
            return dual::status_t::CONCURRENT_LIMIT;
        }
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
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv), "cudaFree d_B_pinv");
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

template dual::status_t dual_phase2_cu<int, double>(
    int phase, int slack_basis, double start_time, const lp_problem_t<int, double> &lp,
    const simplex_solver_settings_t<int, double> &settings, std::vector<variable_status_t> &vstatus,
    lp_solution_t<int, double> &sol, int &iter, std::vector<double> &steepest_edge_norms);

#endif

} // namespace cuopt::linear_programming::dual_simplex
