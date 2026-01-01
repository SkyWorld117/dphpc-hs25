#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_matrix.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t> class problem_analyzer_t {
  public:
    problem_analyzer_t(const lp_problem_t<i_t, f_t>& problem,
                       const simplex_solver_settings_t<i_t, f_t>& settings)
        : lp_problem(problem), solver_settings(settings), preprocessed_A(0, 0, 0),
          preprocessed_AT(0, 0, 0) {}

    void analyze();
    void display_analysis() const;

    csc_matrix_t<i_t, f_t> preprocessed_A;
    csc_matrix_t<i_t, f_t> preprocessed_AT;

  private:
    const lp_problem_t<i_t, f_t>& lp_problem;
    const simplex_solver_settings_t<i_t, f_t>& solver_settings;

    i_t max_nnz_per_row = 0;
    i_t min_nnz_per_row = 0;
    i_t max_nnz_per_col = 0;
    i_t min_nnz_per_col = 0;
    f_t avg_nnz_per_row = 0.0;
    f_t avg_nnz_per_col = 0.0;

    // Block size statistics:
    // smallest block size that is greater than 95%/99% of the rows/columns
    i_t block_size_per_row_95 = 0;
    i_t block_size_per_col_95 = 0;
    i_t block_size_per_row_99 = 0;
    i_t block_size_per_col_99 = 0;
};

} // namespace cuopt::linear_programming::dual_simplex