#pragma once

#include <cuda_runtime.h>
#include <dual_simplex/sparse_vector.cuh>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class csc_cu_matrix {
  public:
    csc_cu_matrix(i_t m, i_t n, i_t nnz, i_t max_nz, i_t* d_col_ptrs, i_t* d_row_indices, f_t* d_values) : 
        m(m), n(n), nnz(nnz), max_nz(max_nz), d_col_ptrs(d_col_ptrs), d_row_indices(d_row_indices), d_values(d_values) {}

    // ~csc_cu_matrix() {
    //     if (max_nz > 0) {
    //         cudaFree(d_row_indices);
    //         cudaFree(d_col_ptrs);
    //         cudaFree(d_values);
    //     }
    // }

    void fetch_column(i_t col_idx, cu_vector<i_t, f_t>& column) const;
    void fetch_row(i_t row_idx, cu_vector<i_t, f_t>& row) const;
    void spmv_sparse(const cu_vector<i_t, f_t>& vec, cu_vector<i_t, f_t>& result) const;
    void spmv_sparse_transpose(const cu_vector<i_t, f_t>& vec, cu_vector<i_t, f_t>& result) const;
    void spmv_dense(const f_t* d_vec, f_t* d_result) const;
    void spmv_dense_transpose(const f_t* d_vec, f_t* d_result) const;

    i_t m;
    i_t n;
    i_t nnz;
    i_t max_nz;
    i_t* d_col_ptrs;
    i_t* d_row_indices;
    f_t* d_values;
};

}