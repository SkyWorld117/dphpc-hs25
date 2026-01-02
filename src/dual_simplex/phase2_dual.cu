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
                           i_t *&d_B_col_ind, f_t *&d_B_values, i_t &nz_B, cudaStream_t stream) {
    // 1. Allocate and compute row counts
    // Initialize row counts to 0
    CUDA_CALL_AND_CHECK(cudaMemset(d_B_row_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_B_row_ptr");
    assert(d_B_row_ptr != nullptr);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    count_basis_rows_kernel<<<grid_size, block_size, 0, stream>>>(m, d_A_col_start, d_A_row_ind,
                                                                  d_basic_list, d_B_row_ptr);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_basis_rows_kernel");

    // 2. Prefix sum to get row pointers
    // We use thrust for exclusive scan. d_B_row_ptr currently holds counts.
    // exclusive_scan on [0, m+1) will transform counts to offsets.
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_B_row_ptr);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + m + 1, dev_ptr, i_t(0));

    // Get total NNZ (stored in the last element after scan)
    i_t total_nz;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&total_nz, d_B_row_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy total_nz");
    nz_B = total_nz;

    // 3. Allocate B column indices and values TODO: could we allocate a bigger buffer at start?
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

    fill_basis_csr_kernel<<<grid_size, block_size, 0, stream>>>(
        m, d_A_col_start, d_A_row_ind, d_A_values, d_basic_list, d_write_offsets, d_B_col_ind,
        d_B_values);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_basis_csr_kernel");

    CUDA_CALL_AND_CHECK(cudaFree(d_write_offsets), "cudaFree d_write_offsets");
}

template <typename i_t>
__global__ void count_basis_transpose_rows_kernel(i_t m, const i_t *__restrict__ A_col_start,
                                                  const i_t *__restrict__ A_row_ind,
                                                  const i_t *__restrict__ basic_list,
                                                  i_t *__restrict__ row_counts) {
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
__global__ void
fill_basis_transpose_csr_kernel(i_t m, const i_t *__restrict__ A_col_start,
                                const i_t *__restrict__ A_row_ind, const f_t *__restrict__ A_values,
                                const i_t *__restrict__ basic_list, i_t *__restrict__ Bt_row_ptr,
                                i_t *__restrict__ Bt_col_ind, f_t *__restrict__ Bt_values) {
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
void build_basis_transpose_on_device(i_t m, const i_t *d_A_col_start, const i_t *d_A_row_ind,
                                     const f_t *d_A_values, const i_t *d_basic_list,
                                     i_t *&d_Bt_row_ptr, i_t *&d_Bt_col_ind, f_t *&d_Bt_values,
                                     i_t &nz_Bt, cudaStream_t stream) {
    // 1. Allocate and compute column counts for B_T
    CUDA_CALL_AND_CHECK(cudaMemset(d_Bt_row_ptr, 0, (m + 1) * sizeof(i_t)),
                        "cudaMemset d_Bt_row_ptr");
    assert(d_Bt_row_ptr != nullptr);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    count_basis_transpose_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        m, d_A_col_start, d_A_row_ind, d_basic_list, d_Bt_row_ptr);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_basis_transpose_rows_kernel");

    // 2. Prefix sum to get row pointers for B_T
    thrust::device_ptr<i_t> dev_ptr = thrust::device_pointer_cast(d_Bt_row_ptr);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + m + 1, dev_ptr, i_t(0));

    // Get total NNZ for B_T
    i_t total_nz;
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(&total_nz, d_Bt_row_ptr + m, sizeof(i_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy total_nz for B_T");
    nz_Bt = total_nz;

    // 3. Allocate B_T column indices and values
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_col_ind, total_nz * sizeof(i_t)),
                        "cudaMalloc d_Bt_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_values, total_nz * sizeof(f_t)), "cudaMalloc d_Bt_values");

    // 4. Fill CSR structure for B_T
    fill_basis_transpose_csr_kernel<<<grid_size, block_size, 0, stream>>>(
        m, d_A_col_start, d_A_row_ind, d_A_values, d_basic_list, d_Bt_row_ptr, d_Bt_col_ind,
        d_Bt_values);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_basis_transpose_csr_kernel");
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
void build_basis_and_basis_transpose_on_device(i_t m, const i_t *d_A_col_ptr,
                                               const i_t *d_A_row_ind, const f_t *d_A_values,
                                               const i_t *d_basic_list, i_t *&d_B_row_ptr,
                                               i_t *&d_B_col_ind, f_t *&d_B_values,
                                               i_t *&d_Bt_row_ptr, i_t *&d_Bt_col_ind,
                                               f_t *&d_Bt_values, i_t &nz_B, i_t &nz_Bt,
                                               cudaStream_t stream1, cudaStream_t stream2) {
    // Build B
    phase2_cu::build_basis_on_device<i_t, f_t>(m, d_A_col_ptr, d_A_row_ind, d_A_values,
                                               d_basic_list, d_B_row_ptr, d_B_col_ind, d_B_values,
                                               nz_B, stream1);

    // Build B_T
    phase2_cu::build_basis_transpose_on_device<i_t, f_t>(m, d_A_col_ptr, d_A_row_ind, d_A_values,
                                                         d_basic_list, d_Bt_row_ptr, d_Bt_col_ind,
                                                         d_Bt_values, nz_Bt, stream2);
}

template <typename i_t, typename f_t>
__global__ void densify_Bt_cols(i_t m, i_t num_cols, i_t col_start,
                                const i_t *__restrict__ B_row_ptr,
                                const i_t *__restrict__ B_col_ind, const f_t *__restrict__ B_values,
                                f_t *__restrict__ Bt_dense_slice) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    i_t start = B_row_ptr[col_start];
    i_t end = B_row_ptr[col_start + num_cols];
    if (idx >= end - start)
        return;

    i_t global_idx = start + idx; // Actual index into B arrays
    // Find which column this thread is working on
    // TODO: This can be optimized if we have constant nnz per column/row
    for (i_t col_offset = 0; col_offset < num_cols; ++col_offset) {
        i_t col_idx = col_start + col_offset;
        i_t col_nnz_start = B_row_ptr[col_idx];
        i_t col_nnz_end = B_row_ptr[col_idx + 1];
        if (global_idx >= col_nnz_start && global_idx < col_nnz_end) {
            i_t row = B_col_ind[global_idx];
            f_t val = B_values[global_idx];
            Bt_dense_slice[col_offset * m + row] = val; // Column-major
            break;
        }
    }
}

template <typename i_t, typename f_t>
__global__ void count_nnz_cols(i_t m, i_t num_cols, i_t col_start, const f_t *__restrict__ X_slice,
                               i_t *__restrict__ nnz_per_col) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cols * m)
        return;

    i_t col_idx = idx / m;
    i_t row_idx = idx % m;

    f_t val = X_slice[col_idx * m + row_idx]; // Column-major
    if (abs(val) > static_cast<f_t>(1e-12)) {
        atomicAdd(&nnz_per_col[col_idx], 1);
    }
}

template <typename i_t, typename f_t>
__global__ void fill_csc_cols(i_t m, i_t num_cols, i_t col_start, const f_t *__restrict__ X_slice,
                              i_t *__restrict__ write_offsets, i_t *__restrict__ X_row_ind,
                              f_t *__restrict__ X_values) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cols * m)
        return;
    i_t col_idx = idx / m;
    i_t row_idx = idx % m;
    f_t val = X_slice[col_idx * m + row_idx]; // Column-major
    if (abs(val) > static_cast<f_t>(1e-12)) {
        // Get position to write using atomic increment on the column offset
        i_t pos = atomicAdd(&write_offsets[col_idx], 1);
        X_row_ind[pos] = row_idx;
        X_values[pos] = val;
    }
}

template <typename i_t> __global__ void shift_col_ptrs(i_t num_elements, i_t *col_ptrs, i_t offset) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements)
        return;
    col_ptrs[idx] += offset;
}

template <typename i_t, typename f_t>
void compute_inverse(cusparseHandle_t &cusparse_handle, cudssHandle_t &cudss_handle,
                     cudssConfig_t &cudss_config, i_t m, i_t n, const i_t *d_A_col_ptr,
                     const i_t *d_A_row_ind, const f_t *d_A_values, i_t *&d_B_row_ptr,
                     i_t *&d_B_col_ind, f_t *&d_B_values, i_t *&d_Bt_row_ptr, i_t *&d_Bt_col_ind,
                     f_t *&d_Bt_values, const std::vector<i_t> &basic_list, i_t *&d_X_col_ptr,
                     i_t *&d_X_row_ind, f_t *&d_X_values, i_t &nz_B, i_t &nz_Bt, i_t &nz_X,
                     const simplex_solver_settings_t<i_t, f_t> &settings) {
    // Move basic list to device
    // TODO: Consider to keep basic list on device
    i_t *d_basic_list;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_basic_list");

    // Assemble the three matrices in parallel
    cudaStream_t stream1, stream2;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream1), "cudaStreamCreate stream1 (B)");
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream2), "cudaStreamCreate stream2 (B_T)");

    // Assemble B and B_T in CSR format
    phase2_cu::build_basis_and_basis_transpose_on_device<i_t, f_t>(
        m, d_A_col_ptr, d_A_row_ind, d_A_values, d_basic_list, d_B_row_ptr, d_B_col_ind, d_B_values,
        d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values, nz_B, nz_Bt, stream1, stream2);

    // Synchronize streams before proceeding
    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after matrix assembly");

    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream1), "cudaStreamDestroy stream1");
    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream2), "cudaStreamDestroy stream2");

    // Here we assume (B^T B) is also sparse
    i_t *d_BtB_row_ptr = nullptr;
    i_t *d_BtB_col_ind = nullptr;
    f_t *d_BtB_values = nullptr;
    cusparseSpMatDescr_t d_B_matrix, d_Bt_matrix, d_BtB_matrix;
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(&d_B_matrix, m, m, nz_B, d_B_row_ptr, d_B_col_ind,
                                              d_B_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                            "cusparseCreateCsr for B");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(&d_Bt_matrix, m, m, nz_Bt, d_Bt_row_ptr, d_Bt_col_ind,
                                              d_Bt_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                            "cusparseCreateCsr for B_T");
    CUSPARSE_CALL_AND_CHECK(cusparseCreateCsr(&d_BtB_matrix, m, m,
                                              0, // to be computed
                                              nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F),
                            "cusparseCreateCsr for B_T B");

    void *d_buffer1 = nullptr;
    void *d_buffer2 = nullptr;
    size_t buffer_size1 = 0;
    size_t buffer_size2 = 0;
    f_t alpha = 1.0;
    f_t beta = 0.0;
    cusparseSpGEMMDescr_t spgemm_desc;
    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_createDescr(&spgemm_desc), "cusparseSpGEMM_createDescr");

    CUSPARSE_CALL_AND_CHECK(
        cusparseSpGEMM_workEstimation(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, d_Bt_matrix,
                                      d_B_matrix, &beta, d_BtB_matrix, CUDA_R_64F,
                                      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size1, nullptr),
        "cusparseSpGEMM_workEstimation for B_T B");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_buffer1, buffer_size1),
                        "cudaMalloc d_buffer1 for B_T B"); // TODO: check if we can reuse buffer?
                                                           // malloc one big one at start?

    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_workEstimation(
                                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, d_Bt_matrix, d_B_matrix,
                                &beta, d_BtB_matrix, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_desc, &buffer_size1, d_buffer1),
                            "cusparseSpGEMM_workEstimation with alloced buffer for B_T B");

    CUSPARSE_CALL_AND_CHECK(
        cusparseSpGEMM_compute(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, d_Bt_matrix, d_B_matrix,
                               &beta, d_BtB_matrix, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                               spgemm_desc, &buffer_size2, nullptr),
        "cusparseSpGEMM_compute for B_T B");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_buffer2, buffer_size2), "cudaMalloc d_buffer2 for B_T B");

    CUSPARSE_CALL_AND_CHECK(
        cusparseSpGEMM_compute(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, d_Bt_matrix, d_B_matrix,
                               &beta, d_BtB_matrix, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                               spgemm_desc, &buffer_size2, d_buffer2),
        "cusparseSpGEMM_compute with alloced buffer for B_T B");

    // Get the size of the result matrix AFTER compute
    int64_t rows_BtB, cols_BtB, nnz_BtB;
    CUSPARSE_CALL_AND_CHECK(cusparseSpMatGetSize(d_BtB_matrix, &rows_BtB, &cols_BtB, &nnz_BtB),
                            "cusparseSpMatGetSize for B_T B");
    assert(rows_BtB == m);
    assert(cols_BtB == m);

    // Allocate memory for the result matrix
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_row_ptr, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_BtB_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_col_ind, nnz_BtB * sizeof(i_t)),
                        "cudaMalloc d_BtB_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_BtB_values, nnz_BtB * sizeof(f_t)),
                        "cudaMalloc d_BtB_values");

    // Update the matrix descriptor with the allocated pointers BEFORE copy
    CUSPARSE_CALL_AND_CHECK(
        cusparseCsrSetPointers(d_BtB_matrix, d_BtB_row_ptr, d_BtB_col_ind, d_BtB_values),
        "cusparseCsrSetPointers for B_T B");

    // Now copy the result into the allocated memory
    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_copy(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                                d_Bt_matrix, d_B_matrix, &beta, d_BtB_matrix,
                                                CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc),
                            "cusparseSpGEMM_copy for B_T B");

    // cuSPARSE cleanup
    CUSPARSE_CALL_AND_CHECK(cusparseSpGEMM_destroyDescr(spgemm_desc),
                            "cusparseSpGEMM_destroyDescr");
    CUDA_CALL_AND_CHECK(cudaFree(d_buffer1), "cudaFree d_buffer1");
    CUDA_CALL_AND_CHECK(cudaFree(d_buffer2), "cudaFree d_buffer2");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(d_B_matrix), "cusparseDestroySpMat B");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(d_Bt_matrix), "cusparseDestroySpMat B_T");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroySpMat(d_BtB_matrix), "cusparseDestroySpMat B_T B");

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
        // Get nnz in this slice from device
        i_t slice_start_nnz, slice_end_nnz;
        CUDA_CALL_AND_CHECK(cudaMemcpy(&slice_start_nnz, d_B_row_ptr + col_start, sizeof(i_t),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy slice_start_nnz");
        CUDA_CALL_AND_CHECK(cudaMemcpy(&slice_end_nnz, d_B_row_ptr + col_start + cols_in_slice,
                                       sizeof(i_t), cudaMemcpyDeviceToHost),
                            "cudaMemcpy slice_end_nnz");
        i_t nnz_slice_B = slice_end_nnz - slice_start_nnz;
        int grid_size = (nnz_slice_B + block_size - 1) / block_size;
        densify_Bt_cols<<<grid_size, block_size>>>(m, cols_in_slice, col_start, d_B_row_ptr,
                                                   d_B_col_ind, d_B_values, d_Bt_slice_dense);
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
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&d_X_matrix_slice_cudss, m, cols_in_slice, m, d_X_slice,
                                                 CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                             "cudssMatrixCreateDn for X");

        CUDSS_CALL_AND_CHECK(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_config, solverData,
                                          d_BtB_matrix_cudss, d_X_matrix_slice_cudss,
                                          d_Bt_matrix_slice_cudss),
                             "cudssExecute Solve for B");
        CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize after cuDSS solve");

        // Count non-zeros in d_X_slice
        // For slice > 0, we need to preserve d_X_col_ptr[col_start] which is the end of previous slice
        i_t prev_slice_end = 0;
        if (slice_idx > 0) {
            CUDA_CALL_AND_CHECK(
                cudaMemcpy(&prev_slice_end, d_X_col_ptr + col_start, sizeof(i_t),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy prev_slice_end");
        }
        
        CUDA_CALL_AND_CHECK(cudaMemset(d_X_col_ptr + col_start, 0, (cols_in_slice + 1) * sizeof(i_t)),
                            "cudaMemset d_X_col_ptr for slice");
        grid_size = (cols_in_slice * m + block_size - 1) / block_size;
        count_nnz_cols<<<grid_size, block_size>>>(m, cols_in_slice, col_start, d_X_slice,
                                                  d_X_col_ptr + col_start);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "count_nnz_cols kernel for X slice");
        // Prefix sum to get column pointers for this slice
        thrust::device_ptr<i_t> dev_ptr =
            thrust::device_pointer_cast(d_X_col_ptr + col_start);
        thrust::exclusive_scan(thrust::cuda::par, dev_ptr, dev_ptr + cols_in_slice + 1, dev_ptr,
                               i_t(0));
        // Get nnz for this slice
        i_t nnz_X_slice;
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(&nnz_X_slice,
                       d_X_col_ptr + col_start + cols_in_slice,
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
        grid_size = (cols_in_slice * m + block_size - 1) / block_size;
        fill_csc_cols<<<grid_size, block_size>>>(m, cols_in_slice, col_start, d_X_slice,
                                                 d_write_offsets, d_X_row_ind_slices[slice_idx],
                                                 d_X_values_slices[slice_idx]);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "fill_csc_cols kernel for X slice");
        CUDA_CALL_AND_CHECK(cudaFree(d_write_offsets), "cudaFree d_write_offsets for X slice");
        // Shift column pointers to account for previous slices
        i_t offset = prev_slice_end;  // Use the accumulated offset
        grid_size = (cols_in_slice + 1 + block_size - 1) / block_size;
        shift_col_ptrs<<<grid_size, block_size>>>(
            cols_in_slice + 1, d_X_col_ptr + col_start, offset);
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
__global__ void fetch_row_as_dense_kernel(i_t m, i_t row_idx, const i_t *__restrict__ Bt_row_ptr,
                                          const i_t *__restrict__ Bt_col_ind,
                                          const f_t *__restrict__ Bt_values,
                                          f_t *__restrict__ row_dense) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    i_t start = Bt_row_ptr[row_idx];
    i_t end = Bt_row_ptr[row_idx + 1];

    if (idx >= end - start)
        return;

    i_t col = Bt_col_ind[start + idx];
    f_t val = Bt_values[start + idx];
    row_dense[col] = val;
}

template <typename i_t, typename f_t>
void fetch_row_as_dense(i_t m, i_t row_idx, const i_t *d_Bt_row_ptr, const i_t *d_Bt_col_ind,
                        const f_t *d_Bt_values, f_t *h_row_dense, cudaStream_t stream) {
    // Initialize to zero
    CUDA_CALL_AND_CHECK(cudaMemset(h_row_dense, 0, m * sizeof(f_t)), "cudaMemset h_row_dense");
    int block_size = 32;
    int grid_size = (m + block_size - 1) / block_size;
    fetch_row_as_dense_kernel<<<grid_size, block_size, 0, stream>>>(
        m, row_idx, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values, h_row_dense);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fetch_row_as_dense_kernel");
}

template <typename i_t, typename f_t>
__global__ void fetch_column_as_dense_kernel(i_t m, i_t col_idx, const i_t *__restrict__ B_row_ptr,
                                             const i_t *__restrict__ B_col_ind,
                                             const f_t *__restrict__ B_values,
                                             f_t *__restrict__ col_dense) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    i_t start = B_row_ptr[col_idx];
    i_t end = B_row_ptr[col_idx + 1];

    if (idx >= end - start)
        return;

    i_t row = B_col_ind[start + idx];
    f_t val = B_values[start + idx];
    col_dense[row] = val;
}

template <typename i_t, typename f_t>
void fetch_column_as_dense(i_t m, i_t col_idx, const i_t *d_B_row_ptr, const i_t *d_B_col_ind,
                           const f_t *d_B_values, f_t *h_col_dense, cudaStream_t stream) {
    // Initialize to zero
    CUDA_CALL_AND_CHECK(cudaMemset(h_col_dense, 0, m * sizeof(f_t)), "cudaMemset h_col_dense");
    int block_size = 32;
    int grid_size = (m + block_size - 1) / block_size;
    fetch_column_as_dense_kernel<<<grid_size, block_size, 0, stream>>>(
        m, col_idx, d_B_row_ptr, d_B_col_ind, d_B_values, h_col_dense);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fetch_column_as_dense_kernel");
}

template <typename i_t, typename f_t>
__global__ void extract_row_kernel(i_t m, const f_t *d_B_pinv, i_t row_idx, f_t *row_data) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    row_data[idx] = d_B_pinv[idx * m + row_idx]; // Column-major: row_idx + idx*m
}

template <typename i_t, typename f_t>
bool eta_update_inverse(cublasHandle_t cublas_handle, i_t m, f_t *d_B_pinv, f_t *eta_b_old,
                        f_t *eta_b_new, f_t *eta_v, f_t *eta_c, f_t *eta_d, const i_t *d_A_col_ptr,
                        const i_t *d_A_row_ind, const f_t *d_A_values, const i_t *d_Bt_row_ptr,
                        const i_t *d_Bt_col_ind, const f_t *d_Bt_values, i_t basic_leaving_index,
                        i_t entering_index) {
    const i_t j = basic_leaving_index; // Index of leaving variable in basis

    cudaStream_t stream1, stream2;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream1), "cudaStreamCreate stream1");
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream2), "cudaStreamCreate stream2");
    // Fetch b_old and b_new as dense vectors
    phase2_cu::fetch_row_as_dense(m, basic_leaving_index, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values,
                                  eta_b_old, stream1);
    phase2_cu::fetch_column_as_dense(m, entering_index, d_A_col_ptr, d_A_row_ind, d_A_values,
                                     eta_b_new, stream2);
    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream1), "cudaStreamDestroy stream1");
    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream2), "cudaStreamDestroy stream2");

    CUDA_CALL_AND_CHECK(cudaMemset(eta_v, 0, m * sizeof(f_t)), "cudaMemset eta_v");
    CUDA_CALL_AND_CHECK(cudaMemset(eta_c, 0, m * sizeof(f_t)), "cudaMemset eta_c");
    CUDA_CALL_AND_CHECK(cudaMemset(eta_d, 0, m * sizeof(f_t)), "cudaMemset eta_d");

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
    CUBLAS_CALL_AND_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, m, m, &alpha, d_B_pinv, m, eta_c,
                                      1,              // eta_c contains u = b_new - b_old
                                      &beta, eta_v, 1 // eta_v will contain p = B_inv @ u
                                      ),
                          "cublasDgemv compute p = B_inv @ (b_new - b_old)");

    // Get p[j] and compute denominator: 1 + p[j]
    f_t p_j;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&p_j, eta_v + j, sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy p_j from device");
    f_t denom = 1.0 + p_j;

    if (std::abs(denom) < 1e-10) {
        // Singular or near-singular update - should refactor instead
        return false;
    }

    // Extract row j of B_inv (store in eta_d)
    // B_inv is column-major, so row j is at offsets j, j+m, j+2m, ...
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    extract_row_kernel<<<grid_size, block_size>>>(m, d_B_pinv, j, eta_d);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "extract_row_kernel");

    // B_inv = B_inv - (p @ row_j) / denom
    alpha = -1.0 / denom;
    CUBLAS_CALL_AND_CHECK(cublasDger(cublas_handle, m, m, &alpha, eta_v, 1, // p
                                     eta_d, 1,                              // row_j
                                     d_B_pinv, m),
                          "cublasDger Sherman-Morrison update");

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
__global__ void construct_c_basic_kernel(i_t m, const i_t *__restrict__ basic_list,
                                         const f_t *__restrict__ objective,
                                         f_t *__restrict__ c_basic) {
    i_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m)
        return;

    i_t col_idx = basic_list[idx]; // Column index in original matrix
    c_basic[idx] = objective[col_idx];
}

template <typename i_t, typename f_t>
void pinv_solve(cublasHandle_t &cublas_handle, f_t *d_B_pinv, const i_t *d_rhs_indices,
                const f_t *d_rhs_values, f_t *&d_x, const i_t m, const i_t nz_rhs, bool transpose) {
    std::cout << "pinv dense solve gpu\n" << std::endl;
    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;
    denseMatrixSparseVectorMulKernel<<<grid_size, block_size>>>(m, d_B_pinv, d_rhs_indices,
                                                                d_rhs_values, m, d_x, transpose);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "denseMatrixSparseVectorMulKernel");
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
    //     for (i_t p = col_start; p < col_end; ++p) {
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
i_t compute_delta_x(const lp_problem_t<i_t, f_t> &lp, cublasHandle_t &cublas_handle, f_t *d_B_pinv,
                    i_t entering_index, i_t leaving_index, i_t basic_leaving_index, i_t direction,
                    const std::vector<i_t> &basic_list, const std::vector<f_t> &delta_x_flip,
                    const sparse_vector_t<i_t, f_t> &rhs_sparse, const std::vector<f_t> &x,
                    sparse_vector_t<i_t, f_t> &scaled_delta_xB_sparse, std::vector<f_t> &delta_x) {
    f_t delta_x_leaving = direction == 1 ? lp.lower[leaving_index] - x[leaving_index]
                                         : lp.upper[leaving_index] - x[leaving_index];
    // B*w = -A(:, entering)
    //   ft.b_solve(rhs_sparse, scaled_delta_xB_sparse, utilde_sparse);
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, rhs_sparse, scaled_delta_xB_sparse,
                          (i_t) (basic_list.size()), false);
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
                settings.log.printf("Bsolve diff %d %e rhs %e residual %e\n", k, err, rhs[k],
                                    residual_B[k]);
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
        // We couldn't find a coefficient for the basic leaving index.
        // The coefficient might be very small. Switch to a regular solve and try to recover.
        std::vector<f_t> rhs;
        rhs_sparse.to_dense(rhs);
        const i_t m = basic_list.size();
        std::vector<f_t> scaled_delta_xB(m);
        // ft.b_solve(rhs, scaled_delta_xB);
        phase2::pinv_solve(cublas_handle, d_B_pinv, rhs, scaled_delta_xB, m, false);
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

template <typename i_t, typename f_t>
void compute_delta_y(cublasHandle_t &cublas_handle, f_t *d_B_pinv, i_t basic_leaving_index,
                     i_t direction, sparse_vector_t<i_t, f_t> &delta_y_sparse) {
    const i_t m = delta_y_sparse.n;
    // BT*delta_y = -delta_zB = -sigma*ei
    sparse_vector_t<i_t, f_t> ei_sparse(m, 1);
    ei_sparse.i[0] = basic_leaving_index;
    ei_sparse.x[0] = -direction;
    //   ft.b_transpose_solve(ei_sparse, delta_y_sparse, UTsol_sparse);
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, ei_sparse, delta_y_sparse, m, true);
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
                                   cublasHandle_t &cublas_handle, f_t *d_B_pinv,
                                   std::vector<f_t> &delta_y_steepest_edge, i_t m) {
    // Start of GPU code, but we fall back to CPU for now
    // i_t *d_row_degree, *d_mapping;
    // f_t *d_coeff;
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_row_degree, m * sizeof(i_t)), "cudaMalloc d_row_degree");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_mapping, m * sizeof(i_t)), "cudaMalloc d_mapping");
    // CUDA_CALL_AND_CHECK(cudaMalloc(&d_coeff, m * sizeof(f_t)), "cudaMalloc d_coeff");
    // CUDA_CALL_AND_CHECK(cudaMemset(d_row_degree, 0, m * sizeof(i_t)), "cudaMemset d_row_degree");
    // CUDA_CALL_AND_CHECK(cudaMemset(d_mapping, -1, m * sizeof(i_t)), "cudaMemset d_mapping");
    // CUDA_CALL_AND_CHECK(cudaMemset(d_coeff, 0, m * sizeof(f_t)), "cudaMemset d_coeff");
    //
    // int block_size = 256;
    // int grid_size = (m + block_size - 1) / block_size;
    // initialize_steepest_edge_norms_find_row_degree_kernel<<<grid_size, block_size>>>(
    //     m, d_basic_list, lp.A.d_col_ptr, lp.A.d_row_ind, lp.A.d_values, d_row_degree, d_mapping,
    //     d_coeff);
    // CUDA_CALL_AND_CHECK(cudaGetLastError(),
    //                     "initialize_steepest_edge_norms_find_row_degree_kernel");

    // We want to compute B^T delta_y_i = -e_i
    // If there is a column u of B^T such that B^T(:, u) = alpha * e_i than the
    // solve delta_y_i = -1/alpha * e_u
    // So we need to find columns of B^T (or rows of B) with only a single non-zero entry
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
        settings.log.printf("Found %d singleton rows for steepest edge norms in %.2fs\n",
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
            //   ft.b_transpose_solve(sparse_ei, sparse_dy);
            phase2_cu::pinv_solve(cublas_handle, d_B_pinv, sparse_ei, sparse_dy, m, true);
            f_t my_init = 0.0;
            for (i_t p = 0; p < (i_t) (sparse_dy.x.size()); ++p) {
                my_init += sparse_dy.x[p] * sparse_dy.x[p];
            }
            init = my_init;
        }
        // ei[k]          = 0.0;
        // init = vector_norm2_squared<i_t, f_t>(dy);
        assert(init > 0);
        delta_y_steepest_edge[j] = init;

        f_t now = toc(start_time);
        f_t time_since_log = toc(last_log);
        if (time_since_log > 10) {
            last_log = tic();
            settings.log.printf("Initialized %d of %d steepest edge norms in %.2fs\n", k, m, now);
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

template <typename i_t, typename f_t>
i_t update_steepest_edge_norms(const simplex_solver_settings_t<i_t, f_t> &settings,
                               const std::vector<i_t> &basic_list, cublasHandle_t &cublas_handle,
                               f_t *d_B_pinv, i_t direction,
                               const sparse_vector_t<i_t, f_t> &delta_y_sparse, f_t dy_norm_squared,
                               const sparse_vector_t<i_t, f_t> &scaled_delta_xB,
                               i_t basic_leaving_index, i_t entering_index, std::vector<f_t> &v,
                               std::vector<f_t> &delta_y_steepest_edge) {
    i_t m = basic_list.size();
    const i_t delta_y_nz = delta_y_sparse.i.size();
    sparse_vector_t<i_t, f_t> v_sparse(m, 0);
    // B^T delta_y = - direction * e_basic_leaving_index
    // We want B v =  - B^{-T} e_basic_leaving_index
    //   ft.b_solve(delta_y_sparse, v_sparse);
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, delta_y_sparse, v_sparse, m, false);
    if (direction == -1) {
        v_sparse.negate();
    }
    v_sparse.scatter(v);

    const i_t leaving_index = basic_list[basic_leaving_index];
    const f_t prev_dy_norm_squared = delta_y_steepest_edge[leaving_index];
#ifdef STEEPEST_EDGE_DEBUG
    const f_t err = std::abs(dy_norm_squared - prev_dy_norm_squared) / (1.0 + dy_norm_squared);
    if (err > 1e-3) {
        settings.log.printf("i %d j %d leaving norm error %e computed %e previous estimate %e\n",
                            basic_leaving_index, leaving_index, err, dy_norm_squared,
                            prev_dy_norm_squared);
    }
#endif

    // B*w = A(:, leaving_index)
    // B*scaled_delta_xB = -A(:, leaving_index) so w = -scaled_delta_xB
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
            std::cout << "New val before max " << new_val << " for j " << j << std::endl;
            new_val = std::max(new_val, 1e-4);
#ifdef STEEPEST_EDGE_DEBUG
            if (!(new_val >= 0)) {
                settings.log.printf("new val %e\n", new_val);
                settings.log.printf("k %d j %d norm old %e wk %e vk %e wr %e omegar %e\n", k, j,
                                    delta_y_steepest_edge[j], wk, v[k], wr, omegar);
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
// Compute steepest edge info for entering variable
template <typename i_t, typename f_t>
i_t compute_steepest_edge_norm_entering(const simplex_solver_settings_t<i_t, f_t> &settings, i_t m,
                                        cublasHandle_t &cublas_handle, f_t *d_B_pinv,
                                        i_t basic_leaving_index, i_t entering_index,
                                        std::vector<f_t> &steepest_edge_norms) {
    sparse_vector_t<i_t, f_t> es_sparse(m, 1);
    es_sparse.i[0] = basic_leaving_index;
    es_sparse.x[0] = -1.0;
    sparse_vector_t<i_t, f_t> delta_ys_sparse(m, 0);
    //   // ft.b_transpose_solve(es_sparse, delta_ys_sparse);
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, es_sparse, delta_ys_sparse, m, true);
    steepest_edge_norms[entering_index] = delta_ys_sparse.norm2_squared();

#ifdef STEEPEST_EDGE_DEBUG
    settings.log.printf("Steepest edge norm %e for entering j %d at i %d\n",
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

    i_t col_idx = d_basic_list[idx]; // Column index in original matrix
    d_x[col_idx] = d_xB[idx];
}

template <typename i_t, typename f_t>
void compute_primal_variables(cublasHandle_t &cublas_handle, f_t *d_B_pinv, const i_t *d_dummy_idx,
                              const f_t *lp_rhs, const i_t *d_A_col_ptr, const i_t *d_A_row_ind,
                              const f_t *d_A_values, const i_t *d_basic_list,
                              const i_t *d_nonbasic_list, f_t tight_tol, f_t *&d_x, const i_t m,
                              const i_t n) {
    f_t *d_rhs;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_rhs, m * sizeof(f_t)), "cudaMalloc d_rhs");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_rhs, lp_rhs, m * sizeof(f_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy lp_rhs");
    // rhs = b - sum_{j : x_j = l_j} A(:, j) * l(j)
    //         - sum_{j : x_j = u_j} A(:, j) * u(j)
    int block_size = 256;
    int grid_size_nonbasic = ((n - m) + block_size - 1) / block_size;
    compute_primal_variables_nonbasiclist_kernel<<<grid_size_nonbasic, block_size>>>(
        m, n, d_A_col_ptr, d_A_row_ind, d_A_values, d_nonbasic_list, d_x, tight_tol, d_rhs);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_primal_variables_nonbasiclist_kernel");

    f_t *d_xB;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_xB, m * sizeof(f_t)), "cudaMalloc d_xB");
    //   ft.b_solve(rhs, xB);
    phase2_cu::pinv_solve<i_t, f_t>(cublas_handle, d_B_pinv, d_dummy_idx, d_rhs, d_xB, m, m, false);

    int grid_size_basic = (m + block_size - 1) / block_size;
    compute_primal_variables_basiclist_kernel<<<grid_size_basic, block_size>>>(m, d_basic_list,
                                                                               d_xB, d_x);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "compute_primal_variables_basiclist_kernel");
    CUDA_CALL_AND_CHECK(cudaFree(d_rhs), "cudaFree d_rhs");
    CUDA_CALL_AND_CHECK(cudaFree(d_xB), "cudaFree d_xB");
}

template <typename i_t, typename f_t>
i_t compute_primal_solution_from_basis(const lp_problem_t<i_t, f_t> &lp,
                                       cublasHandle_t &cublas_handle, f_t *d_B_pinv,
                                       const std::vector<i_t> &basic_list,
                                       const std::vector<i_t> &nonbasic_list,
                                       const std::vector<variable_status_t> &vstatus,
                                       std::vector<f_t> &x) {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;
    std::vector<f_t> rhs = lp.rhs;

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

    // rhs = b - sum_{j : x_j = l_j} A(:, j) l(j) - sum_{j : x_j = u_j} A(:, j) *
    // u(j)
    for (i_t k = 0; k < n - m; ++k) {
        const i_t j = nonbasic_list[k];
        const i_t col_start = lp.A.col_start[j];
        const i_t col_end = lp.A.col_start[j + 1];
        const f_t xj = x[j];
        for (i_t p = col_start; p < col_end; ++p) {
            rhs[lp.A.i[p]] -= xj * lp.A.x[p];
        }
    }

    std::vector<f_t> xB(m);
    //   ft.b_solve(rhs, xB);
    phase2_cu::pinv_solve<i_t, f_t>(cublas_handle, d_B_pinv, rhs, xB, m, false);

    for (i_t k = 0; k < m; ++k) {
        const i_t j = basic_list[k];
        x[j] = xB[k];
    }
    return 0;
}

template <typename i_t, typename f_t>
void compute_dual_solution_from_basis(const lp_problem_t<i_t, f_t> &lp,
                                      cublasHandle_t &cublas_handle, f_t *d_B_pinv,
                                      const std::vector<i_t> &basic_list,
                                      const std::vector<i_t> &nonbasic_list, std::vector<f_t> &y,
                                      std::vector<f_t> &z) {
    const i_t m = lp.num_rows;
    const i_t n = lp.num_cols;

    y.resize(m);
    std::vector<f_t> cB(m);
    for (i_t k = 0; k < m; ++k) {
        const i_t j = basic_list[k];
        cB[k] = lp.objective[j];
    }
    //   ft.b_transpose_solve(cB, y);
    phase2_cu::pinv_solve<i_t, f_t>(cublas_handle, d_B_pinv, cB, y, m, true);

    // We want A'y + z = c
    // A = [ B N ]
    // B' y = c_B, z_B = 0
    // N' y + z_N = c_N
    z.resize(n);
    // zN = cN - N'*y
    for (i_t k = 0; k < n - m; k++) {
        const i_t j = nonbasic_list[k];
        // z_j <- c_j
        z[j] = lp.objective[j];

        // z_j <- z_j - A(:, j)'*y
        const i_t col_start = lp.A.col_start[j];
        const i_t col_end = lp.A.col_start[j + 1];
        f_t dot = 0.0;
        for (i_t p = col_start; p < col_end; ++p) {
            dot += lp.A.x[p] * y[lp.A.i[p]];
        }
        z[j] -= dot;
    }
    // zB = 0
    for (i_t k = 0; k < m; ++k) {
        z[basic_list[k]] = 0.0;
    }
}
template <typename i_t, typename f_t>
void prepare_optimality(const lp_problem_t<i_t, f_t> &lp,
                        const simplex_solver_settings_t<i_t, f_t> &settings, cublasHandle_t &handle,
                        f_t *d_B_pinv, const std::vector<f_t> &objective,
                        const std::vector<i_t> &basic_list, const std::vector<i_t> &nonbasic_list,
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
        phase2_cu::compute_dual_solution_from_basis(lp, handle, d_B_pinv, basic_list, nonbasic_list,
                                                    unperturbed_y, unperturbed_z);
        {
            const f_t dual_infeas = phase2::dual_infeasibility(
                lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
            if (dual_infeas <= settings.dual_tol) {
                settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
                z = unperturbed_z;
                y = unperturbed_y;
                perturbation = 0.0;
            } else {
                settings.log.printf("Failed to remove perturbation of %.2e.\n", perturbation);
            }
        }
    }

    sol.l2_primal_residual = phase2::l2_primal_residual(lp, sol);
    sol.l2_dual_residual = phase2::l2_dual_residual(lp, sol);
    const f_t dual_infeas = phase2::dual_infeasibility(lp, settings, vstatus, z, 0.0, 0.0);
    const f_t primal_infeas = phase2::primal_infeasibility(lp, settings, vstatus, x);
    if (phase == 1 && iter > 0) {
        settings.log.printf("Dual phase I complete. Iterations %d. Time %.2f\n", iter,
                            toc(start_time));
    }
    if (phase == 2) {
        if (!settings.inside_mip) {
            settings.log.printf("\n");
            settings.log.printf("Optimal solution found in %d iterations and %.2fs\n", iter,
                                toc(start_time));
            settings.log.printf("Objective %+.8e\n", sol.user_objective);
            settings.log.printf("\n");
            settings.log.printf("Primal infeasibility (abs): %.2e\n", primal_infeas);
            settings.log.printf("Dual infeasibility (abs):   %.2e\n", dual_infeas);
            settings.log.printf("Perturbation:               %.2e\n", perturbation);
        } else {
            settings.log.printf("\n");
            settings.log.printf("Root relaxation solution found in %d iterations and %.2fs\n", iter,
                                toc(start_time));
            settings.log.printf("Root relaxation objective %+.8e\n", sol.user_objective);
            settings.log.printf("\n");
        }
    }
}

template <typename i_t, typename f_t>
void adjust_for_flips(cublasHandle_t &cublas_handle, f_t *d_B_pinv,
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
    //   ft.b_solve(atilde_sparse, delta_xB_0_sparse);
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, atilde_sparse, delta_xB_0_sparse, m, false);
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

// TODO: end kernelify

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

    // SETUP GPU ALLOCATIONS AND HANDLES
    // create all handles
    cublasHandle_t cublas_handle;
    CUBLAS_CALL_AND_CHECK(cublasCreate(&cublas_handle), "cublasCreate");
    cusparseHandle_t cusparse_handle;
    CUSPARSE_CALL_AND_CHECK(cusparseCreate(&cusparse_handle), "cusparseCreate");
    cudssHandle_t cudss_handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&cudss_handle), "cudssCreateHandle B");
    cudssConfig_t cudss_config;
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&cudss_config), "cudssCreateConfig B");

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
    i_t *d_B_row_ptr;
    i_t *d_B_col_ind;
    f_t *d_B_values;
    i_t nz_B;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_row_ptr");

    i_t *d_Bt_row_ptr;
    i_t *d_Bt_col_ind;
    f_t *d_Bt_values;
    i_t nz_Bt;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_Bt_row_ptr, (m + 1) * sizeof(i_t)),
                        "cudaMalloc d_Bt_row_ptr");

    f_t *d_B_pinv;
    i_t *d_B_pinv_col_ptr;
    i_t *d_B_pinv_row_ind;
    f_t *d_B_pinv_values;
    i_t nz_B_pinv;
    phase2_cu::compute_inverse<i_t, f_t>(
        cusparse_handle, cudss_handle, cudss_config, m, n, d_A_col_ptr, d_A_row_ind, d_A_values,
        d_B_row_ptr, d_B_col_ind, d_B_values, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values, basic_list,
        d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B, nz_Bt, nz_B_pinv, settings);

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
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv, m * m * sizeof(f_t)), "cudaMalloc d_B_pinv");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_pinv, h_B_pinv_dense.data(), m * m * sizeof(f_t),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy h_B_pinv_dense to d_B_pinv");
    }

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

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
    //     const i_t j = basic_list[k]; // j = col idx in A
    //     c_basic[k] = objective[j];   // costs of objective that wont be zeroed out
    // }
    phase2_cu::construct_c_basic_kernel<<<grid_size, block_size>>>(m, d_basic_list, d_objective,
                                                                   d_c_basic);
    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "construct_c_basic_kernel");
    std::vector<i_t> fake_indices(m);
    for (i_t i = 0; i < m; ++i) {
        fake_indices[i] = i;
    }
    i_t *d_dummy_idx_m; // dummy indices for dense vec of size m
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_dummy_idx_m, m * sizeof(i_t)), "cudaMalloc d_dummy_idx");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_dummy_idx_m, fake_indices.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy fake_indices to device");

    f_t *d_y;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_y, m * sizeof(f_t)), "cudaMalloc d_y");
    // Solve B'*y = cB
    phase2_cu::pinv_solve(cublas_handle, d_B_pinv, d_dummy_idx_m, d_c_basic, d_y, m, m, true);

    if (toc(start_time) > settings.time_limit) {
        return dual::status_t::TIME_LIMIT;
    }

    // phase2_cu::compute_reduced_costs(d_objective, d_A_col_ptr, d_A_row_ind, d_A_values, m, n,
    // d_y,
    //                                  basic_list, nonbasic_list, d_z);
    f_t *d_z;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_z, n * sizeof(f_t)), "cudaMalloc d_z");
    phase2_cu::compute_reduced_costs(d_objective, d_A_col_ptr, d_A_row_ind, d_A_values, d_y,
                                     d_basic_list, d_nonbasic_list, d_z, m, n);

    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "compute_reduced_costs");

    // Copy z and y back to host
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

    phase2_cu::compute_primal_variables(cublas_handle, d_B_pinv, d_dummy_idx_m, d_lp_rhs,
                                        d_A_col_ptr, d_A_row_ind, d_A_values, d_basic_list,
                                        d_nonbasic_list, settings.tight_tol, d_x, m, n);

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
        if (slack_basis) {
            phase2::initialize_steepest_edge_norms_from_slack_basis(
                basic_list, nonbasic_list,
                delta_y_steepest_edge); // TODO: left off of GPU for now, bc i think is only called
                                        // at the start, but needs to be confirmed
        } else {
            std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
            if (phase2_cu::initialize_steepest_edge_norms(lp, settings, start_time, d_basic_list,
                                                          cublas_handle, d_B_pinv,
                                                          delta_y_steepest_edge, m) ==
                -1) { // TODO: opted against GPUifing delta_y_steepest_edge for now
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
            phase2_cu::prepare_optimality(lp, settings, cublas_handle, d_B_pinv, objective,
                                          basic_list, nonbasic_list, vstatus, phase, start_time,
                                          max_val, iter, x, y, z, sol);
            status = dual::status_t::OPTIMAL;
            break;
        }

        // BTran
        // BT*delta_y = -delta_zB = -sigma*ei
        timers.start_timer();
        sparse_vector_t<i_t, f_t> delta_y_sparse(m, 0);
        phase2_cu::compute_delta_y(cublas_handle, d_B_pinv, basic_leaving_index, direction,
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
                phase2_cu::compute_dual_solution_from_basis(lp, cublas_handle, d_B_pinv, basic_list,
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
                        phase2_cu::compute_primal_solution_from_basis(lp, cublas_handle, d_B_pinv,
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
                            phase2_cu::prepare_optimality(lp, settings, cublas_handle, d_B_pinv,
                                                          objective, basic_list, nonbasic_list,
                                                          vstatus, phase, start_time, max_val, iter,
                                                          x, y, z, sol);
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
                        phase2_cu::compute_primal_solution_from_basis(lp, cublas_handle, d_B_pinv,
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
                            phase2_cu::prepare_optimality(lp, settings, cublas_handle, d_B_pinv,
                                                          objective, basic_list, nonbasic_list,
                                                          vstatus, phase, start_time, max_val, iter,
                                                          x, y, z, sol);
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
            phase2_cu::adjust_for_flips(cublas_handle, d_B_pinv, basic_list, delta_z_indices,
                                        atilde_index, atilde, atilde_mark, delta_xB_0_sparse,
                                        delta_x_flip, x);
            timers.ftran_time += timers.stop_timer();
        }

        timers.start_timer();
        sparse_vector_t<i_t, f_t> scaled_delta_xB_sparse(m, 0);
        sparse_vector_t<i_t, f_t> rhs_sparse(lp.A, entering_index);
        if (phase2_cu::compute_delta_x(lp, cublas_handle, d_B_pinv, entering_index, leaving_index,
                                       basic_leaving_index, direction, basic_list, delta_x_flip,
                                       rhs_sparse, x, scaled_delta_xB_sparse, delta_x) == -1) {
            settings.log.printf("Failed to compute delta_x. Iter %d\n", iter);
            return dual::status_t::NUMERICAL;
        }

        timers.ftran_time += timers.stop_timer();

        timers.start_timer();
        const i_t steepest_edge_status = phase2_cu::update_steepest_edge_norms(
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

        CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "Before basis update");
        // Update basic and nonbasic lists and marks

        basic_list[basic_leaving_index] = entering_index;
        nonbasic_list[nonbasic_entering_index] = leaving_index;
        nonbasic_mark[entering_index] = -1;
        nonbasic_mark[leaving_index] = nonbasic_entering_index;
        basic_mark[leaving_index] = -1;
        basic_mark[entering_index] = basic_leaving_index;
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_basic_list + basic_leaving_index, &entering_index,
                                       sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy entering_index to d_basic_list");
        CUDA_CALL_AND_CHECK(cudaMemcpy(d_nonbasic_list + nonbasic_entering_index, &leaving_index,
                                       sizeof(i_t), cudaMemcpyHostToDevice),
                            "cudaMemcpy leaving_index to d_nonbasic_list");

        timers.start_timer();
        // Refactor or update the basis factorization
        bool should_refactor = (iter + 1) % settings.refactor_frequency == 0;

        if (!should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Update 1");
            }

            should_refactor = !phase2_cu::eta_update_inverse(
                cublas_handle, m, d_B_pinv, eta_b_old, eta_b_new, eta_v, eta_c, eta_d, d_A_col_ptr,
                d_A_row_ind, d_A_values, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values,
                basic_leaving_index, entering_index);

            if (settings.profile) {
                settings.timer.stop("Inverse Update 1");
            }
        }

        // Free old B and Bt and recompute
        // CUDA_CALL_AND_CHECK(cudaFree(d_B_col_ind), "cudaFree d_B_col_ind");
        // CUDA_CALL_AND_CHECK(cudaFree(d_B_values), "cudaFree d_B_values");
        // CUDA_CALL_AND_CHECK(cudaFree(d_Bt_col_ind), "cudaFree d_Bt_col_ind");
        // CUDA_CALL_AND_CHECK(cudaFree(d_Bt_values), "cudaFree d_Bt_values");

        if (!should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Update 2");
            }

            // Move basic list to device
            // TODONE: as above consider keeping this on device
            // i_t *d_basic_list;
            // CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)),
            //                     "cudaMalloc d_basic_list");
            // CUDA_CALL_AND_CHECK(cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t),
            //                                cudaMemcpyHostToDevice),
            //                     "cudaMemcpy to d_basic_list");

            // TODO: It's probably not smart to rebuild B and Bt from scratch every time
            // Instead use a 95% threshold (or higher) to determine the size per row to
            // allocate s.t. we don't have to realloc every time
            cudaStream_t stream1, stream2;
            CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream1), "cudaStreamCreate stream1");
            CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream2), "cudaStreamCreate stream2");
            phase2_cu::build_basis_and_basis_transpose_on_device<i_t, f_t>(
                m, d_A_col_ptr, d_A_row_ind, d_A_values, d_basic_list, d_B_row_ptr, d_B_col_ind,
                d_B_values, d_Bt_row_ptr, d_Bt_col_ind, d_Bt_values, nz_B, nz_Bt, stream1, stream2);

            CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(),
                                "cudaDeviceSynchronize after build B and Bt");
            CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream1), "cudaStreamDestroy stream1");
            CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream2), "cudaStreamDestroy stream2");

            if (settings.profile) {
                settings.timer.stop("Inverse Update 2");
            }
        }

        if (should_refactor) {
            if (settings.profile) {
                settings.timer.start("Inverse Refactorizaton");
            }

            // Free old B and Bt
            CUDA_CALL_AND_CHECK(cudaFree(d_B_col_ind), "cudaFree d_B_col_ind");
            CUDA_CALL_AND_CHECK(cudaFree(d_B_values), "cudaFree d_B_values");
            CUDA_CALL_AND_CHECK(cudaFree(d_Bt_col_ind), "cudaFree d_Bt_col_ind");
            CUDA_CALL_AND_CHECK(cudaFree(d_Bt_values), "cudaFree d_Bt_values");

            // Recompute d_B_pinv
            phase2_cu::compute_inverse<i_t, f_t>(
                cusparse_handle, cudss_handle, cudss_config, m, n, d_A_col_ptr, d_A_row_ind,
                d_A_values, d_B_row_ptr, d_B_col_ind, d_B_values, d_Bt_row_ptr, d_Bt_col_ind,
                d_Bt_values, basic_list, d_B_pinv_col_ptr, d_B_pinv_row_ind, d_B_pinv_values, nz_B,
                nz_Bt, nz_B_pinv, settings);

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
                CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_pinv, m * m * sizeof(f_t)),
                                    "cudaMalloc d_B_pinv");
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
        phase2_cu::compute_steepest_edge_norm_entering(settings, m, cublas_handle, d_B_pinv,
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
    CUDA_CALL_AND_CHECK(cudaFree(d_B_row_ptr), "cudaFree d_B_row_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_Bt_row_ptr), "cudaFree d_Bt_row_ptr");
    CUDA_CALL_AND_CHECK(cudaFree(d_B_pinv), "cudaFree d_D_pinv");
    CUBLAS_CALL_AND_CHECK(cublasDestroy(cublas_handle), "cublasDestroy");
    CUSPARSE_CALL_AND_CHECK(cusparseDestroy(cusparse_handle), "cusparseDestroy");
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
