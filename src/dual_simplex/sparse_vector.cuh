#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CALL_AND_CHECK(call, msg)                                                             \
    do {                                                                                           \
        cudaError_t cuda_error = call;                                                             \
        if (cuda_error != cudaSuccess) {                                                           \
            printf("CUDA API returned error = %d from call " #msg ", details: %s\n", cuda_error,   \
                   cudaGetErrorString(cuda_error));                                                \
        }                                                                                          \
    } while (0);

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class cu_vector {
  public:
    cu_vector(i_t m, i_t max_nz) : m(m), nnz(0), max_nz(max_nz) {
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_indices, max_nz * sizeof(i_t)), "cu_vector constructor d_indices");
        CUDA_CALL_AND_CHECK(cudaMalloc(&d_values, max_nz * sizeof(f_t)), "cu_vector constructor d_values");
    }
    cu_vector(i_t m, i_t nnz, i_t max_nz, i_t* d_indices, f_t* d_values) : 
        m(m), nnz(nnz), max_nz(max_nz), d_indices(d_indices), d_values(d_values) {}

    // ~cu_vector() {
    //     if (max_nz > 0) {
    //         cudaFree(d_indices);
    //         cudaFree(d_values);
    //     }
    // }

    // subtract: self - other = result
    void subtract(const cu_vector<i_t, f_t>& other, cu_vector<i_t, f_t>& result);
    
    // squared_norm: result = ||self||^2
    void squared_norm(f_t& result);

    f_t get(i_t index);
    
    // reset: clear the vector (set nnz to 0)
    void reset() {
        nnz = 0;
        CUDA_CALL_AND_CHECK(cudaMemset(d_indices, 0, max_nz * sizeof(i_t)), "cu_vector reset memset d_indices");
        CUDA_CALL_AND_CHECK(cudaMemset(d_values, 0, max_nz * sizeof(f_t)), "cu_vector reset memset d_values");
    }

    i_t m;
    i_t nnz;
    i_t max_nz;
    i_t* d_indices;
    f_t* d_values;
};

}