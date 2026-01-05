#include <dual_simplex/problem_analysis.hpp>
#include <vector>

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

    // Compute block size statistics
    std::vector<i_t> sorted_col_counts = col_counts;
    std::sort(sorted_col_counts.begin(), sorted_col_counts.end());
    block_size_per_col_95 = sorted_col_counts[static_cast<i_t>(0.95 * n)];
    block_size_per_col_99 = sorted_col_counts[static_cast<i_t>(0.99 * n)];

    // Compute median nnz per row and column
    if (n % 2 == 0) {
        median_nnz_per_col = (sorted_col_counts[n / 2 - 1] + sorted_col_counts[n / 2]) / 2;
    } else {
        median_nnz_per_col = sorted_col_counts[n / 2];
    }

    // Check if indices are sorted
    sorted_indices = true;
    for (i_t j = 0; j < n; ++j) {
        i_t col_start = A.col_start[j];
        i_t col_end = A.col_start[j + 1];
        for (i_t p = col_start + 1; p < col_end; ++p) {
            if (A.i[p] < A.i[p - 1]) {
                sorted_indices = false;
                break;
            }
        }
        if (!sorted_indices) {
            break;
        }
    }
}

template <typename i_t, typename f_t> void problem_analyzer_t<i_t, f_t>::display_analysis() const {
    printf("Problem Analysis:\n");
    printf("Rows: %d, Columns: %d\n", lp_problem.num_rows, lp_problem.num_cols);
    printf("Max NNZ per Column: %d\n", max_nnz_per_col);
    printf("Min NNZ per Column: %d\n", min_nnz_per_col);
    printf("Average NNZ per Column: %.2f\n",
           avg_nnz_per_col / static_cast<f_t>(lp_problem.num_cols));
    printf("Median NNZ per Column: %d\n", median_nnz_per_col);

    printf("95th Percentile Block Size per Column: %d\n", block_size_per_col_95);
    printf("99th Percentile Block Size per Column: %d\n", block_size_per_col_99);
    printf("Indices Sorted: %s\n", sorted_indices ? "Yes" : "No");
}

template <typename i_t, typename f_t>
void reconstruct_column(const csc_matrix_t<i_t, f_t>& A, std::vector<i_t>& row_indices,
                        std::vector<f_t>& values, i_t col, i_t max_nnz_per_col) {
    i_t m = A.m;
    i_t col_start = A.col_start[col];
    i_t col_end = A.col_start[col + 1];
    i_t existing_nnz = col_end - col_start;

    // Extract existing row indices and values (already sorted in CSC format)
    std::vector<std::pair<i_t, f_t>> existing_entries;
    for (i_t p = col_start; p < col_end; ++p) {
        existing_entries.push_back({ A.i[p], A.x[p] });
    }

    // If existing entries already exceed or equal max_nnz_per_col, just copy them
    if (existing_nnz >= max_nnz_per_col) {
        row_indices.resize(max_nnz_per_col);
        values.resize(max_nnz_per_col);
        for (i_t i = 0; i < max_nnz_per_col; ++i) {
            row_indices[i] = existing_entries[i].first;
            values[i] = existing_entries[i].second;
        }
        return;
    }

    // Determine the range spanned by existing non-zeros
    i_t min_row = existing_entries.empty() ? 0 : existing_entries.front().first;
    i_t max_row = existing_entries.empty() ? 0 : existing_entries.back().first;
    i_t span = max_row - min_row + 1;

    // Padding needed
    i_t padding_needed = max_nnz_per_col - existing_nnz;

    row_indices.clear();
    values.clear();
    row_indices.reserve(max_nnz_per_col);
    values.reserve(max_nnz_per_col);

    // Copy all existing entries first
    for (const auto& entry : existing_entries) {
        row_indices.push_back(entry.first);
        values.push_back(entry.second);
    }

    // Create a set of used rows for quick lookup
    std::vector<bool> used_rows(m, false);
    for (const auto& entry : existing_entries) {
        used_rows[entry.first] = true;
    }

    // Add padding: try to fill gaps within [min_row, max_row] first, then extend
    i_t padding_added = 0;

    // Fill gaps within existing range
    for (i_t row = min_row; row <= max_row && padding_added < padding_needed; ++row) {
        if (!used_rows[row]) {
            row_indices.push_back(row);
            values.push_back(0.0);
            used_rows[row] = true;
            padding_added++;
        }
    }

    // Extend downward if needed
    for (i_t row = min_row - 1; row >= 0 && padding_added < padding_needed; --row) {
        if (!used_rows[row]) {
            row_indices.push_back(row);
            values.push_back(0.0);
            used_rows[row] = true;
            padding_added++;
        }
    }

    // Extend upward if needed
    for (i_t row = max_row + 1; row < m && padding_added < padding_needed; ++row) {
        if (!used_rows[row]) {
            row_indices.push_back(row);
            values.push_back(0.0);
            used_rows[row] = true;
            padding_added++;
        }
    }

    // Sort by row indices to maintain CSC format
    std::vector<std::pair<i_t, f_t>> pairs(row_indices.size());
    for (size_t i = 0; i < row_indices.size(); ++i) {
        pairs[i] = { row_indices[i], values[i] };
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<i_t, f_t>& a, const std::pair<i_t, f_t>& b) {
                  return a.first < b.first;
              });

    for (size_t i = 0; i < pairs.size(); ++i) {
        row_indices[i] = pairs[i].first;
        values[i] = pairs[i].second;
    }
}

template <typename i_t, typename f_t>
void problem_analyzer_t<i_t, f_t>::construct_preprocessed_A() {
    // Create a copy of the original matrix A in CSC format
    // every column should have the same number of non-zeros where nnz = max_nnz_per_col
    preprocessed_A = csc_matrix_t<i_t, f_t>(lp_problem.A.m, lp_problem.A.n,
                                            lp_problem.A.col_start[lp_problem.A.n]);
    const csc_matrix_t<i_t, f_t>& A = lp_problem.A;
    preprocessed_A.i.resize(A.n * max_nnz_per_col);
    preprocessed_A.x.resize(A.n * max_nnz_per_col);

    for (i_t j = 0; j < A.n; ++j) {
        i_t col_start = A.col_start[j];
        i_t col_end = A.col_start[j + 1];
        i_t nnz_in_col = col_end - col_start;

        std::vector<i_t> col_indices(max_nnz_per_col);
        std::vector<f_t> col_values(max_nnz_per_col);

        reconstruct_column(A, col_indices, col_values, j, max_nnz_per_col);

        // Fill in the preprocessed_A matrix
        i_t pre_col_start = j * max_nnz_per_col;
        preprocessed_A.col_start[j] = pre_col_start;
        for (i_t p = 0; p < max_nnz_per_col; ++p) {
            preprocessed_A.i[pre_col_start + p] = col_indices[p];
            preprocessed_A.x[pre_col_start + p] = col_values[p];
        }
    }

    preprocessed_A.col_start[A.n] = A.n * max_nnz_per_col;
}
// Explicit template instantiations
template class problem_analyzer_t<int, double>;

} // namespace cuopt::linear_programming::dual_simplex