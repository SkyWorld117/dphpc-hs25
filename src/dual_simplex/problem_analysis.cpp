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
                        std::vector<f_t>& values, i_t col, i_t target_nnz_per_col) {
    i_t m = A.m;
    i_t col_start = A.col_start[col];
    i_t col_end = A.col_start[col + 1];
    i_t existing_nnz = col_end - col_start;

    // Extract existing row indices and values (already sorted in CSC format)
    std::vector<std::pair<i_t, f_t>> existing_entries;
    for (i_t p = col_start; p < col_end; ++p) {
        existing_entries.push_back({ A.i[p], A.x[p] });
    }

    // If existing entries already exceed or equal target_nnz_per_col, keep all of them (no
    // truncation)
    if (existing_nnz >= target_nnz_per_col) {
        row_indices.resize(existing_nnz);
        values.resize(existing_nnz);
        for (i_t i = 0; i < existing_nnz; ++i) {
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
    i_t padding_needed = target_nnz_per_col - existing_nnz;

    row_indices.clear();
    values.clear();
    row_indices.reserve(target_nnz_per_col);
    values.reserve(target_nnz_per_col);

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
void problem_analyzer_t<i_t, f_t>::construct_preprocessed_A(const f_t coverage_ratio) {
    const csc_matrix_t<i_t, f_t>& A = lp_problem.A;
    i_t n = A.n;
    i_t m = A.m;

    // Compute the target nnz based on coverage ratio
    std::vector<i_t> col_counts(n);
    #pragma omp parallel for
    for (i_t j = 0; j < n; ++j) {
        col_counts[j] = A.col_start[j + 1] - A.col_start[j];
    }

    std::vector<i_t> sorted_col_counts = col_counts;
    std::sort(sorted_col_counts.begin(), sorted_col_counts.end());
    i_t target_nnz = sorted_col_counts[static_cast<i_t>(coverage_ratio * n)];

    // First pass: determine actual nnz for each column after padding
    nnz_per_col.resize(n);
    #pragma omp parallel for
    for (i_t j = 0; j < n; ++j) {
        i_t existing_nnz = col_counts[j];
        nnz_per_col[j] = (existing_nnz >= target_nnz) ? existing_nnz : target_nnz;
    }

    // Compute col_start array based on actual nnz per column
    preprocessed_A.col_start.resize(n + 1);
    preprocessed_A.col_start[0] = 0;
    for (i_t j = 0; j < n; ++j) {
        preprocessed_A.col_start[j + 1] = preprocessed_A.col_start[j] + nnz_per_col[j];
    }

    i_t total_nnz = preprocessed_A.col_start[n];
    preprocessed_A.m = m;
    preprocessed_A.n = n;
    preprocessed_A.i.resize(total_nnz);
    preprocessed_A.x.resize(total_nnz);

    // Second pass: fill in the matrix data in parallel
    #pragma omp parallel for
    for (i_t j = 0; j < n; ++j) {
        std::vector<i_t> col_indices;
        std::vector<f_t> col_values;

        reconstruct_column(A, col_indices, col_values, j, target_nnz);

        // Fill in the preprocessed_A matrix
        i_t pre_col_start = preprocessed_A.col_start[j];
        for (size_t p = 0; p < col_indices.size(); ++p) {
            preprocessed_A.i[pre_col_start + p] = col_indices[p];
            preprocessed_A.x[pre_col_start + p] = col_values[p];
        }
    }
}
// Explicit template instantiations
template class problem_analyzer_t<int, double>;

} // namespace cuopt::linear_programming::dual_simplex