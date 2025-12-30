#include <cstdio>
#include <dual_simplex/phase2_dual.cuh>

#include <cuda_runtime.h>
#include <cudss.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define CUDA_CALL_AND_CHECK(call, msg)                                                             \
    do {                                                                                           \
        cudaError_t cuda_error = call;                                                             \
        if (cuda_error != cudaSuccess) {                                                           \
            printf("CUDA API returned error = %d, details: " #msg "\n", cuda_error);               \
        }                                                                                          \
    } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                                    \
    do {                                                                                           \
        status = call;                                                                             \
        if (status != CUDSS_STATUS_SUCCESS) {                                                      \
            printf("CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n",        \
                   status);                                                                        \
        }                                                                                          \
    } while (0);

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

// template <typename i_t, typename f_t>
// i_t factorize_and_repair(i_t m, i_t n, const csc_matrix_t<i_t, f_t> &A,
//                          const std::vector<i_t> &basic_list, cudssHandle_t &dss_handle,
//                          cudssConfig_t &solverConfig, cudssData_t &solverData) {
//     dual::status_t status = dual::status_t::UNSET;
//
//     // 1. Assemble B = A(:, basic_list) as CSR
//     // 2. Move B to device
//     // 3. Factor B = L*U using cuDSS
//     // 4. If fails, repair the basis only once and try again. If still fails,
//     // return NUMERICAL status
//     //    Notice this might happen only if we eliminate singletons which we do not
//     //    do for now. Seems like the cleanup for deficient columns is not
//     //    necessary without singleton handling.
//     // TODO: Implement singleton handling and basis repair (and potentially
//     // cleanup of deficient columns)
//
//     // Assemble B in CSR format
//     csr_matrix_t<i_t, f_t> B_csr;
//     B_csr.m = m;
//     B_csr.n = m;
//     B_csr.nz_max = 0;
//     B_csr.row_start.resize(m + 1, 0);
//
//     std::vector<i_t> elements_per_row(m);
//     for (i_t col_idx = 0; col_idx < m; ++col_idx) {
//         i_t j = basic_list[col_idx];
//         for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
//             i_t row = A.i[idx];
//             elements_per_row[row]++;
//         }
//     }
//     for (i_t row = 0; row < m; ++row) {
//         B_csr.row_start[row + 1] = B_csr.row_start[row] + elements_per_row[row];
//         B_csr.nz_max += elements_per_row[row];
//     }
//     B_csr.j.resize(B_csr.nz_max);
//     B_csr.x.resize(B_csr.nz_max);
//     std::vector<i_t> current_position_in_row(m);
//     for (i_t col_idx = 0; col_idx < m; ++col_idx) {
//         i_t j = basic_list[col_idx];
//         for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
//             i_t row = A.i[idx];
//             i_t dest_pos = B_csr.row_start[row] + current_position_in_row[row];
//             B_csr.j[dest_pos] = col_idx;
//             B_csr.x[dest_pos] = A.x[idx];
//             current_position_in_row[row]++;
//         }
//     }
//
//     // Move B to device
//     i_t *d_B_row_ptr;
//     i_t *d_B_col_ind;
//     f_t *d_B_values;
//
//     CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc
//     d_B_row_ptr"); CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ind, B_csr.nz_max * sizeof(i_t)),
//                         "cudaMalloc d_B_col_ind");
//     CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_values, B_csr.nz_max * sizeof(f_t)),
//                         "cudaMalloc d_B_values");
//
//     CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_row_ptr, B_csr.row_start.data(), (m + 1) * sizeof(i_t),
//                                    cudaMemcpyHostToDevice),
//                         "cudaMemcpy d_B_row_ptr");
//     CUDA_CALL_AND_CHECK(
//         cudaMemcpy(d_B_col_ind, B_csr.j.data(), B_csr.nz_max * sizeof(i_t),
//         cudaMemcpyHostToDevice), "cudaMemcpy d_B_col_ind");
//     CUDA_CALL_AND_CHECK(
//         cudaMemcpy(d_B_values, B_csr.x.data(), B_csr.nz_max * sizeof(f_t),
//         cudaMemcpyHostToDevice), "cudaMemcpy d_B_values");
//
//     // Factor B = L*U using cuDSS
//     cudssStatus_t dss_status = CUDSS_STATUS_SUCCESS;
//
//     cudssMatrix_t d_B_matrix;
//     CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&d_B_matrix, m, m, B_csr.nz_max, d_B_row_ptr, NULL,
//                                               d_B_col_ind, d_B_values, CUDA_R_32I, CUDA_R_64F,
//                                               CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
//                                               CUDSS_BASE_ZERO),
//                          dss_status, "cudssMatrixCreateCsr for B");
//
//     CUDSS_CALL_AND_CHECK(cudssExecute(dss_handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
//                                       d_B_matrix, NULL, NULL),
//                          dss_status, "cudssExecute Analysis for B");
//
//     CUDSS_CALL_AND_CHECK(cudssExecute(dss_handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
//                                       solverData, d_B_matrix, NULL, NULL),
//                          dss_status, "cudssExecute Factorization for B");
//
//     CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_B_matrix), dss_status, "cudssMatrixDestroy for B");
//
//     return 0;
// }

// template <typename i_t, typename f_t>
// i_t move_basis_to_device(cudssMatrix_t *d_B_matrix, const csc_matrix_t<i_t, f_t> &A,
//                          const std::vector<i_t> &basic_list, cudssHandle_t &dss_handle) {
//     // Assemble B in CSR format
//     const i_t m = A.m;
//     csr_matrix_t<i_t, f_t> B_csr;
//     B_csr.m = m;
//     B_csr.n = m;
//     B_csr.nz_max = 0;
//     B_csr.row_start.resize(m + 1, 0);
//
//     std::vector<i_t> elements_per_row(m);
//     for (i_t col_idx = 0; col_idx < m; ++col_idx) {
//         i_t j = basic_list[col_idx];
//         for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
//             i_t row = A.i[idx];
//             elements_per_row[row]++;
//         }
//     }
//     for (i_t row = 0; row < m; ++row) {
//         B_csr.row_start[row + 1] = B_csr.row_start[row] + elements_per_row[row];
//         B_csr.nz_max += elements_per_row[row];
//     }
//     B_csr.j.resize(B_csr.nz_max);
//     B_csr.x.resize(B_csr.nz_max);
//     std::vector<i_t> current_position_in_row(m);
//     for (i_t col_idx = 0; col_idx < m; ++col_idx) {
//         i_t j = basic_list[col_idx];
//         for (i_t idx = A.col_start[j]; idx < A.col_start[j + 1]; ++idx) {
//             i_t row = A.i[idx];
//             i_t dest_pos = B_csr.row_start[row] + current_position_in_row[row];
//             B_csr.j[dest_pos] = col_idx;
//             B_csr.x[dest_pos] = A.x[idx];
//             current_position_in_row[row]++;
//         }
//     }
//
//     // Move B to device
//     i_t *d_B_row_ptr;
//     i_t *d_B_col_ind;
//     f_t *d_B_values;
//
//     CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc
//     d_B_row_ptr"); CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ind, B_csr.nz_max * sizeof(i_t)),
//                         "cudaMalloc d_B_col_ind");
//     CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_values, B_csr.nz_max * sizeof(f_t)),
//                         "cudaMalloc d_B_values");
//
//     CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_row_ptr, B_csr.row_start.data(), (m + 1) * sizeof(i_t),
//                                    cudaMemcpyHostToDevice),
//                         "cudaMemcpy d_B_row_ptr");
//     CUDA_CALL_AND_CHECK(
//         cudaMemcpy(d_B_col_ind, B_csr.j.data(), B_csr.nz_max * sizeof(i_t),
//         cudaMemcpyHostToDevice), "cudaMemcpy d_B_col_ind");
//     CUDA_CALL_AND_CHECK(
//         cudaMemcpy(d_B_values, B_csr.x.data(), B_csr.nz_max * sizeof(f_t),
//         cudaMemcpyHostToDevice), "cudaMemcpy d_B_values");
//
//     // Factor B = L*U using cuDSS
//     cudssStatus_t dss_status = CUDSS_STATUS_SUCCESS;
//
//     CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(d_B_matrix, m, m, B_csr.nz_max, d_B_row_ptr, NULL,
//                                               d_B_col_ind, d_B_values, CUDA_R_32I, CUDA_R_64F,
//                                               CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
//                                               CUDSS_BASE_ZERO),
//                          dss_status, "cudssMatrixCreateCsr for B");
// }

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
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_A_row_ind, A.nz_max * sizeof(i_t)), "cudaMalloc d_A_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_A_values, A.nz_max * sizeof(f_t)), "cudaMalloc d_A_values");

    // Copy data to device
    // Column pointers
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_A_col_ptr, A.col_start.data(), (A.n + 1) * sizeof(i_t),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy d_A_row_ptr");
    // Row indices
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_A_row_ind, A.i.data(), A.nz_max * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_A_col_ind");
    // Non-zero values
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_A_values, A.x.data(), A.nz_max * sizeof(f_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_A_values");
}

// Unfortunately, CUDSS does not support accessing L and U factors directly
// template <typename i_t, typename f_t> void
// factorize_basis(cudssHandle_t &dss_handle, cudssConfig_t &solverConfig,
//                      cudssData_t &solverData, cudssMatrix_t &d_B_matrix,
//                      cudssMatrix_t &d_y_matrix, cudssMatrix_t &d_z_matrix) {
//   cudssStatus_t dss_status = CUDSS_STATUS_SUCCESS;
//
//   CUDSS_CALL_AND_CHECK(cudssExecute(dss_handle, CUDSS_PHASE_ANALYSIS,
//                                     solverConfig, solverData, d_B_matrix,
//                                     NULL, NULL),
//                        dss_status, "cudssExecute Analysis for B");
//
//   CUDSS_CALL_AND_CHECK(cudssExecute(dss_handle, CUDSS_PHASE_FACTORIZATION,
//                                     solverConfig, solverData, d_B_matrix,
//                                     NULL, NULL),
//                        dss_status, "cudssExecute Factorization for B");
// }
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

    // CUDSS Initialization
    cudssStatus_t dss_status = CUDSS_STATUS_SUCCESS;

    cudssHandle_t dss_handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&dss_handle), dss_status, "cudssCreateHandle");

    cudssConfig_t solverConfig;
    cudssData_t solverData;
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), dss_status, "cudssCreateConfig");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(dss_handle, &solverData), dss_status, "cudssCreateData");

    // Move A to device
    i_t *d_A_col_ptr = nullptr;
    i_t *d_A_row_ind = nullptr;
    f_t *d_A_values = nullptr;
    phase2_cu::move_A_to_device<i_t, f_t>(lp.A, d_A_col_ptr, d_A_row_ind, d_A_values);
    assert(d_A_col_ptr != nullptr);
    assert(d_A_row_ind != nullptr);
    assert(d_A_values != nullptr);

    i_t *d_basic_list;
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_basic_list, m * sizeof(i_t)), "cudaMalloc d_basic_list");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(d_basic_list, basic_list.data(), m * sizeof(i_t), cudaMemcpyHostToDevice),
        "cudaMemcpy d_basic_list");
    // Move B to device
    i_t *d_B_row_ptr = nullptr;
    i_t *d_B_col_ind = nullptr;
    f_t *d_B_values = nullptr;
    i_t nz_B;
    phase2_cu::build_basis_on_device<i_t, f_t>(m, d_A_col_ptr, d_A_row_ind, d_A_values,
                                               d_basic_list, d_B_row_ptr, d_B_col_ind, d_B_values,
                                               &nz_B);
    assert(d_B_row_ptr != nullptr);
    assert(d_B_col_ind != nullptr);
    assert(d_B_values != nullptr);
    // move_basis_to_device(&d_B_matrix, lp.A, basic_list, dss_handle);

    // Prepare cuDSS matrices for B, y, z
    cudssMatrix_t d_B_matrix;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&d_B_matrix, m, m, nz_B, d_B_row_ptr, NULL,
                                              d_B_col_ind, d_B_values, CUDA_R_32I, CUDA_R_64F,
                                              CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                              CUDSS_BASE_ZERO),
                         dss_status, "cudssMatrixCreateCsr for B");
    cudssMatrix_t d_y_matrix;
    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateDn(&d_y_matrix, m, 1, m, NULL, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        dss_status, "cudssMatrixCreateDense for y");

    // Compute L*U = A(p, basic_list) = B(p,:)
    // phase2_cu::factorize_and_repair<i_t, f_t>(m, n, lp.A, basic_list,
    // dss_handle, solverConfig, solverData);
    //

    status = dual::status_t::NUMERICAL; // Placeholder until full implementation

    // CUDSS Cleanup
    // CUDSS_CALL_AND_CHECK(cudssDataDestroy(dss_handle, solverData), dss_status,
    //                      "cudssDataDestroy for solverData");
    // CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), dss_status,
    //                      "cudssConfigDestroy for solverConfig");
    // CUDSS_CALL_AND_CHECK(cudssDestroy(dss_handle), dss_status, "cudssDestroy for dss_handle");

    return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template dual::status_t dual_phase2_cu<int, double>(
    int phase, int slack_basis, double start_time, const lp_problem_t<int, double> &lp,
    const simplex_solver_settings_t<int, double> &settings, std::vector<variable_status_t> &vstatus,
    lp_solution_t<int, double> &sol, int &iter, std::vector<double> &steepest_edge_norms);

#endif

} // namespace cuopt::linear_programming::dual_simplex
