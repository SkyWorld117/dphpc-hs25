#include <dual_simplex/phase2_dual.cuh>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cudss.h>

#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cudaError_t cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
        } \
    } while(0);

namespace cuopt::linear_programming::dual_simplex {

namespace phase2_cu {

template <typename i_t, typename f_t>
i_t factorize_and_repair(
    i_t m,
    i_t n,
    const csc_matrix_t<i_t, f_t>& A,
    const std::vector<i_t>& basic_list,
    cudssHandle_t& dss_handle,
    cudssConfig_t& solverConfig,
    cudssData_t& solverData
) {
    dual::status_t status = dual::status_t::UNSET;

    // 1. Assemble B = A(:, basic_list) as CSR
    // 2. Move B to device
    // 3. Factor B = L*U using cuDSS
    // 4. If fails, repair the basis only once and try again. If still fails, return NUMERICAL status
    //    Notice this might happen only if we eliminate singletons which we do not do for now.
    //    Seems like the cleanup for deficient columns is not necessary without singleton handling.
    // TODO: Implement singleton handling and basis repair (and potentially cleanup of deficient columns)

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

    // Move B to device
    i_t* d_B_row_ptr;
    i_t* d_B_col_ind;
    f_t* d_B_values;

    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_row_ptr, (m + 1) * sizeof(i_t)), "cudaMalloc d_B_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_col_ind, B_csr.nz_max * sizeof(i_t)), "cudaMalloc d_B_col_ind");
    CUDA_CALL_AND_CHECK(cudaMalloc(&d_B_values, B_csr.nz_max * sizeof(f_t)), "cudaMalloc d_B_values");

    CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_row_ptr, B_csr.row_start.data(), (m + 1) * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy d_B_row_ptr");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_col_ind, B_csr.j.data(), B_csr.nz_max * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy d_B_col_ind");
    CUDA_CALL_AND_CHECK(cudaMemcpy(d_B_values, B_csr.x.data(), B_csr.nz_max * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy d_B_values");

    // Factor B = L*U using cuDSS
    cudssStatus_t dss_status = CUDSS_STATUS_SUCCESS;

    cudssMatrix_t d_B_matrix;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(
        &d_B_matrix,
        m,
        m,
        B_csr.nz_max,
        d_B_row_ptr,
        NULL,
        d_B_col_ind,
        d_B_values,
        CUDA_R_32I,
        CUDA_R_64F,
        CUDSS_MTYPE_GENERAL,
        CUDSS_MVIEW_FULL,
        CUDSS_BASE_ZERO
    ), dss_status, "cudssMatrixCreateCsr for B");

    CUDSS_CALL_AND_CHECK(cudssExecute(
        dss_handle,
        CUDSS_PHASE_ANALYSIS,
        solverConfig,
        solverData,
        d_B_matrix,
        NULL,
        NULL
    ), dss_status, "cudssExecute Analysis for B");
    
    CUDSS_CALL_AND_CHECK(cudssExecute(
        dss_handle,
        CUDSS_PHASE_FACTORIZATION,
        solverConfig,
        solverData,
        d_B_matrix,
        NULL,
        NULL
    ), dss_status, "cudssExecute Factorization for B");

    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(d_B_matrix), dss_status, "cudssMatrixDestroy for B");

    return 0;
}


}  // namespace phase2_cu

template <typename i_t, typename f_t>
dual::status_t dual_phase2_cu(
    i_t phase,
    i_t slack_basis,
    f_t start_time,
    const lp_problem_t<i_t, f_t>& lp,
    const simplex_solver_settings_t<i_t, f_t>& settings,
    std::vector<variable_status_t>& vstatus,
    lp_solution_t<i_t, f_t>& sol,
    i_t& iter,
    std::vector<f_t>& delta_y_steepest_edge
) {
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

    std::vector<f_t>& x = sol.x;
    std::vector<f_t>& y = sol.y;
    std::vector<f_t>& z = sol.z;

    dual::status_t status = dual::status_t::UNSET;

    // Perturbed objective
    std::vector<f_t> objective = lp.objective;

    settings.log.printf("Dual Simplex Phase %d\n", phase);
    std::vector<variable_status_t> vstatus_old = vstatus;
    std::vector<f_t> z_old                     = z;

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

    // Compute L*U = A(p, basic_list)
    phase2_cu::factorize_and_repair<i_t, f_t>(
        m, n, lp.A, basic_list,
        dss_handle, solverConfig, solverData
    );

    status = dual::status_t::NUMERICAL;  // Placeholder until full implementation

    // CUDSS Cleanup
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(dss_handle, solverData), dss_status, "cudssDataDestroy for solverData");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), dss_status, "cudssConfigDestroy for solverConfig");
    CUDSS_CALL_AND_CHECK(cudssDestroy(dss_handle), dss_status, "cudssDestroy for dss_handle");


    return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template dual::status_t dual_phase2_cu<int, double>(
    int phase,
    int slack_basis,
    double start_time,
    const lp_problem_t<int, double>& lp,
    const simplex_solver_settings_t<int, double>& settings,
    std::vector<variable_status_t>& vstatus,
    lp_solution_t<int, double>& sol,
    int& iter,
    std::vector<double>& steepest_edge_norms);

#endif

}  // namespace cuopt::linear_programming::dual_simplex