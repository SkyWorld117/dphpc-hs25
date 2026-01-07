#include <dual_simplex/sparse_matrix.cuh>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


// Begin: Kernels

template <typename i_t, typename f_t>
__global__ void fetch_row_kernel(i_t row_idx, i_t n_cols, const i_t *d_col_ptrs,
                                 const i_t *d_row_indices, const f_t *d_values, i_t *d_out_idx,
                                 f_t *d_out_val, i_t *d_count, i_t max_out) {
    i_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) {
        return;
    }

    const i_t start = d_col_ptrs[col];
    const i_t end = d_col_ptrs[col + 1];
    for (i_t k = start; k < end; ++k) {
        if (d_row_indices[k] == row_idx) {
            i_t pos = atomicAdd(d_count, 1);
            if (pos < max_out) {
                d_out_idx[pos] = col;      // column index becomes sparse index for the row
                d_out_val[pos] = d_values[k];
            }
            return; // each column contributes at most one entry for this row
        }
    }
}

template <typename i_t, typename f_t>
__global__ void spmv_csc_sparse_vec_kernel(i_t m, const i_t *d_col_ptrs,
                                           const i_t *d_row_indices, const f_t *d_values,
                                           const i_t *d_vec_indices, const f_t *d_vec_values,
                                           i_t nz_vec, f_t *d_out_dense) {
    i_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nz_vec) {
        return;
    }

    const i_t col = d_vec_indices[tid];
    const f_t xj = d_vec_values[tid];

    const i_t start = d_col_ptrs[col];
    const i_t end = d_col_ptrs[col + 1];
    for (i_t k = start; k < end; ++k) {
        const i_t row = d_row_indices[k];
        const f_t val = d_values[k];
        atomicAdd(&d_out_dense[row], xj * val);
    }
}



// End: Kernels


namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void csc_cu_matrix<i_t, f_t>::fetch_column(i_t col_idx, cu_vector<i_t, f_t> &column) const {
    if (col_idx < 0 || col_idx >= n) {
        printf("csc_cu_matrix::fetch_column: column index out of bounds (%lld)\n", static_cast<long long>(col_idx));
        return;
    }

    i_t col_start = 0;
    i_t col_end = 0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&col_start, d_col_ptrs + col_idx, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "csc_cu_matrix::fetch_column: cudaMemcpy col_ptrs");
    CUDA_CALL_AND_CHECK(cudaMemcpy(&col_end, d_col_ptrs + col_idx + 1, sizeof(i_t), cudaMemcpyDeviceToHost),
                        "csc_cu_matrix::fetch_column: cudaMemcpy col_ptrs");

    i_t col_nnz = col_end - col_start;
    if (column.max_nz < col_nnz) {
        printf("csc_cu_matrix::fetch_column: insufficient capacity in column (max_nz=%lld, needed %lld)\n",
               static_cast<long long>(column.max_nz), static_cast<long long>(col_nnz));
        return;
    }

    column.m = m;
    column.nnz = col_nnz;
    
    // Only copy if there's data to copy
    if (col_nnz > 0) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(column.d_indices, d_row_indices + col_start, col_nnz * sizeof(i_t),
                                       cudaMemcpyDeviceToDevice),
                            "csc_cu_matrix::fetch_column: cudaMemcpy row_indices");
        CUDA_CALL_AND_CHECK(cudaMemcpy(column.d_values, d_values + col_start, col_nnz * sizeof(f_t),
                                       cudaMemcpyDeviceToDevice),
                            "csc_cu_matrix::fetch_column: cudaMemcpy values");
    }
}

template <typename i_t, typename f_t>
void csc_cu_matrix<i_t, f_t>::fetch_row(i_t row_idx, cu_vector<i_t, f_t> &row) const {
    if (row_idx < 0 || row_idx >= m) {
        printf("csc_cu_matrix::fetch_row: row index out of bounds (%lld)\n",
               static_cast<long long>(row_idx));
        return;
    }

    // Clear output count on device
    i_t *d_nnz = nullptr;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_nnz, sizeof(i_t)), "cudaMalloc d_nnz");
    CUDA_CALL_AND_CHECK(cudaMemset(d_nnz, 0, sizeof(i_t)), "cudaMemset d_nnz");

    const i_t block_size = 128;
    const i_t grid_size = (n + block_size - 1) / block_size; // n columns to scan

    fetch_row_kernel<<<grid_size, block_size>>>(row_idx, n, d_col_ptrs, d_row_indices, d_values,
                                               row.d_indices, row.d_values, d_nnz, row.max_nz);
    CUDA_CALL_AND_CHECK(cudaGetLastError(), "fetch_row_kernel");

    // Copy nnz back
    i_t h_nnz = 0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&h_nnz, d_nnz, sizeof(i_t), cudaMemcpyDeviceToHost),
                       "cudaMemcpy d_nnz to host");
    CUDA_CALL_AND_CHECK(cudaFree(d_nnz), "cudaFree d_nnz");

    if (h_nnz > row.max_nz) {
        printf(
            "csc_cu_matrix::fetch_row: truncated result (capacity %lld, needed %lld)\n",
            static_cast<long long>(row.max_nz), static_cast<long long>(h_nnz));
        h_nnz = row.max_nz;
    }

    row.m = n;
    row.nnz = h_nnz;

    // Sort by column index to keep sparsity ordered
    if (h_nnz > 1) {
        thrust::device_ptr<i_t> idx_begin(row.d_indices);
        thrust::device_ptr<f_t> val_begin(row.d_values);
        thrust::sort_by_key(idx_begin, idx_begin + h_nnz, val_begin);
    }
}

template <typename i_t, typename f_t>
void csc_cu_matrix<i_t, f_t>::spmv_sparse(const cu_vector<i_t, f_t> &vec, cu_vector<i_t, f_t> &result) const {
    // Assumes square CSC (m rows, m cols) and 0-based indices
    if (result.max_nz < m) {
        printf("csc_cu_matrix::spmv_sparse: insufficient capacity (max_nz=%lld, need up to %lld)\n",
               static_cast<long long>(result.max_nz), static_cast<long long>(m));
        return;
    }
    
    // Handle empty input vector
    if (vec.nnz == 0) {
        result.m = m;
        result.nnz = 0;
        return;
    }

    // Temporary dense accumulation buffer
    f_t *d_dense = nullptr;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_dense, m * sizeof(f_t)), "cudaMalloc d_dense");
    CUDA_CALL_AND_CHECK(cudaMemset(d_dense, 0, m * sizeof(f_t)), "cudaMemset d_dense");

    const i_t block_size = 256;
    const i_t grid_size = (vec.nnz + block_size - 1) / block_size;
    if (grid_size > 0) {
        spmv_csc_sparse_vec_kernel<<<grid_size, block_size>>>(m, d_col_ptrs, d_row_indices,
                                                              d_values, vec.d_indices, vec.d_values,
                                                              vec.nnz, d_dense);
        CUDA_CALL_AND_CHECK(cudaGetLastError(), "spmv_csc_sparse_vec_kernel");
    }

    // Compact dense result into sparse format
    thrust::device_ptr<f_t> val_begin(d_dense);
    thrust::device_ptr<i_t> out_idx_begin(result.d_indices);
    thrust::device_ptr<f_t> out_val_begin(result.d_values);

    auto counting_begin = thrust::make_counting_iterator<i_t>(0);
    auto counting_end = counting_begin + m;

    auto zip_in_begin = thrust::make_zip_iterator(thrust::make_tuple(counting_begin, val_begin));
    auto zip_in_end = thrust::make_zip_iterator(thrust::make_tuple(counting_end, val_begin + m));
    auto zip_out_begin = thrust::make_zip_iterator(thrust::make_tuple(out_idx_begin, out_val_begin));

    auto zip_out_end = thrust::copy_if(
        thrust::cuda::par, zip_in_begin, zip_in_end, zip_out_begin,
        [] __device__(const thrust::tuple<i_t, f_t> &item) { return thrust::get<1>(item) != f_t(0); });

    i_t nnz_out = static_cast<i_t>(zip_out_end - zip_out_begin);
    result.m = m;
    result.nnz = nnz_out;

    CUDA_CALL_AND_CHECK(cudaFree(d_dense), "cudaFree d_dense");
}

template class csc_cu_matrix<int, double>;

}