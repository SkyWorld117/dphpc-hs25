#include <dual_simplex/sparse_vector.cuh>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <vector>

// Begin: Kernels

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

// End: Kernels

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
void cu_vector<i_t, f_t>::subtract(const cu_vector<i_t, f_t> &other, cu_vector<i_t, f_t> &result) {
    if (m != other.m) {
        printf("cu_vector::substract: dimension mismatch (%lld vs %lld)\n",
               static_cast<long long>(m), static_cast<long long>(other.m));
        return;
    }

    const i_t nnz_a = nnz;
    const i_t nnz_b = other.nnz;
    const i_t merged_cap = nnz_a + nnz_b;
    
    // The result only needs to hold at most m elements (after reduction),
    // not merged_cap elements. Temporary buffers will hold the merged data.
    if (result.max_nz < m) {
        printf(
            "cu_vector::substract: insufficient capacity in result (max_nz=%lld, needed at least %lld)\n",
            static_cast<long long>(result.max_nz), static_cast<long long>(m));
        return;
    }

    // Trivial empty cases
    if (merged_cap == 0) {
        result.m = m;
        result.nnz = 0;
        return;
    }

    // Temporary storage for merged indices/values prior to reduction by key
    i_t *d_idx_tmp = nullptr;
    f_t *d_val_tmp = nullptr;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_idx_tmp, merged_cap * sizeof(i_t)), "cudaMalloc d_idx_tmp");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_val_tmp, merged_cap * sizeof(f_t)), "cudaMalloc d_val_tmp");

    auto policy = thrust::cuda::par.on(0);

    // Wrap raw pointers
    thrust::device_ptr<const i_t> ia_begin(d_indices);
    thrust::device_ptr<const i_t> ib_begin(other.d_indices);
    thrust::device_ptr<const f_t> va_begin(d_values);
    thrust::device_ptr<const f_t> vb_begin(other.d_values);

    thrust::device_ptr<i_t> idx_tmp_begin(d_idx_tmp);
    thrust::device_ptr<f_t> val_tmp_begin(d_val_tmp);

    // Merge indices while negating values from the second vector to form a - b
    auto neg = thrust::negate<f_t>();
    thrust::merge_by_key(policy, ia_begin, ia_begin + nnz_a, ib_begin, ib_begin + nnz_b,
                         va_begin,
                         thrust::make_transform_iterator(vb_begin, neg), idx_tmp_begin,
                         val_tmp_begin);

    // Reduce by key to combine duplicates (union of sparsity patterns)
    thrust::device_ptr<i_t> res_idx_begin(result.d_indices);
    thrust::device_ptr<f_t> res_val_begin(result.d_values);

    auto reduce_out = thrust::reduce_by_key(policy, idx_tmp_begin, idx_tmp_begin + merged_cap,
                                            val_tmp_begin, res_idx_begin, res_val_begin);
    const i_t reduced_nnz = static_cast<i_t>(reduce_out.first - res_idx_begin);

    // Drop structural zeros produced by cancellation
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(res_idx_begin, res_val_begin));
    auto zip_end = zip_begin + reduced_nnz;
    auto zip_out = thrust::remove_if(
        policy, zip_begin, zip_end,
        [] __device__(const thrust::tuple<i_t, f_t> &item) { return thrust::get<1>(item) == f_t(0); });

    const i_t out_nnz = static_cast<i_t>(zip_out - zip_begin);
    result.m = m;
    result.nnz = out_nnz;

    CUDA_CALL_AND_CHECK(cudaFree(d_idx_tmp), "cudaFree d_idx_tmp");
    CUDA_CALL_AND_CHECK(cudaFree(d_val_tmp), "cudaFree d_val_tmp");
}

template <typename i_t, typename f_t> void cu_vector<i_t, f_t>::squared_norm(f_t &result) {
    // Handle empty vector
    if (nnz == 0) {
        result = 0.0;
        return;
    }
    
    const i_t block_size = 256;
    i_t grid_size = (nnz + block_size - 1) / block_size;
    f_t *d_partial_sums;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_partial_sums, grid_size * sizeof(f_t)),
                        "cudaMalloc d_partial_sums");
    sparse_vector_squared_norm_kernel<<<grid_size, block_size, block_size * sizeof(f_t)>>>(
        nnz, d_indices, d_values, d_partial_sums);
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
    result = total_sum;
}

template <typename i_t, typename f_t> f_t cu_vector<i_t, f_t>::get(i_t index) {
    // Early exit when empty or out-of-bounds
    if (nnz == 0 || index < 0 || index >= m) {
        return 0.0;
    }

    auto policy = thrust::cuda::par.on(0);
    thrust::device_ptr<const i_t> idx_begin(d_indices);
    thrust::device_ptr<const i_t> idx_end = idx_begin + nnz;

    // Binary search on device to locate index (indices assumed sorted)
    auto it = thrust::lower_bound(policy, idx_begin, idx_end, index);
    const i_t pos = static_cast<i_t>(it - idx_begin);
    if (pos == nnz) {
        return 0.0;
    }

    i_t found_index = -1;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&found_index, d_indices + pos, sizeof(i_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy found_index in cu_vector::get");
    if (found_index != index) {
        return 0.0;
    }

    f_t value = 0.0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&value, d_values + pos, sizeof(f_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy value in cu_vector::get");
    return value;
}

template class cu_vector<int, double>;

} // namespace cuopt::linear_programming::dual_simplex