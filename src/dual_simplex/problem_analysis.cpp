#include <dual_simplex/problem_analysis.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t> void problem_analyzer_t<i_t, f_t>::analyze() {
    // Calulate statistics on the original problem matrix
    const csc_matrix_t<i_t, f_t>& A = lp_problem.A;
    i_t m = A.m;
    i_t n = A.n;
    i_t total_nnz = A.col_start[n];

    max_nnz_per_col = 0;
    min_nnz_per_col = m + 1; // Initialize to max possible + 1
    std::vector<i_t> col_counts(n, 0);
    for (i_t j = 0; j < n; ++j) {
        i_t col_nnz = A.col_start[j + 1] - A.col_start[j];
        col_counts[j] = col_nnz;
        avg_nnz_per_col += static_cast<f_t>(col_nnz);
        if (col_nnz > max_nnz_per_col) {
            max_nnz_per_col = col_nnz;
        }
        if (col_nnz < min_nnz_per_col) {
            min_nnz_per_col = col_nnz;
        }
    }

    max_nnz_per_row = 0;
    min_nnz_per_row = n + 1; // Initialize to max possible + 1
    std::vector<i_t> row_counts(m, 0);
    for (i_t j = 0; j < n; ++j) {
        i_t col_start = A.col_start[j];
        i_t col_end = A.col_start[j + 1];
        for (i_t p = col_start; p < col_end; ++p) {
            i_t i = A.i[p];
            row_counts[i]++;
        }
    }
    for (i_t i = 0; i < m; ++i) {
        avg_nnz_per_row += static_cast<f_t>(row_counts[i]);
        i_t row_nnz = row_counts[i];
        if (row_nnz > max_nnz_per_row) {
            max_nnz_per_row = row_nnz;
        }
        if (row_nnz < min_nnz_per_row) {
            min_nnz_per_row = row_nnz;
        }
    }

    // Compute block size statistics
    std::vector<i_t> sorted_row_counts = row_counts;
    std::sort(sorted_row_counts.begin(), sorted_row_counts.end());
    block_size_per_row_95 = sorted_row_counts[static_cast<i_t>(0.95 * m)];
    block_size_per_row_99 = sorted_row_counts[static_cast<i_t>(0.99 * m)];

    std::vector<i_t> sorted_col_counts = col_counts;
    std::sort(sorted_col_counts.begin(), sorted_col_counts.end());
    block_size_per_col_95 = sorted_col_counts[static_cast<i_t>(0.95 * n)];
    block_size_per_col_99 = sorted_col_counts[static_cast<i_t>(0.99 * n)];
}

template <typename i_t, typename f_t> void problem_analyzer_t<i_t, f_t>::display_analysis() const {
    printf("Problem Analysis:\n");
    printf("Rows: %d, Columns: %d\n", lp_problem.num_rows, lp_problem.num_cols);
    printf("Max NNZ per Row: %d\n", max_nnz_per_row);
    printf("Min NNZ per Row: %d\n", min_nnz_per_row);
    printf("Max NNZ per Column: %d\n", max_nnz_per_col);
    printf("Min NNZ per Column: %d\n", min_nnz_per_col);
    printf("Average NNZ per Row: %.2f\n", avg_nnz_per_row / static_cast<f_t>(lp_problem.num_rows));
    printf("Average NNZ per Column: %.2f\n",
           avg_nnz_per_col / static_cast<f_t>(lp_problem.num_cols));

    printf("95th Percentile Block Size per Row: %d\n", block_size_per_row_95);
    printf("99th Percentile Block Size per Row: %d\n", block_size_per_row_99);
    printf("95th Percentile Block Size per Column: %d\n", block_size_per_col_95);
    printf("99th Percentile Block Size per Column: %d\n", block_size_per_col_99);
}

// Explicit template instantiations
template class problem_analyzer_t<int, double>;

} // namespace cuopt::linear_programming::dual_simplex