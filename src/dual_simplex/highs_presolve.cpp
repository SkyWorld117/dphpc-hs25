#include <dual_simplex/highs_presolve.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t highs_presolve_t<i_t, f_t>::presolve_from_file(const std::string& filename,
                                            user_problem_t<i_t, f_t>& presolved)
{
    HighsStatus status = highs.readModel(filename);
    if (status != HighsStatus::kOk) {
        return -1;
    }

    status = highs.presolve();
    if (status != HighsStatus::kOk) {
        return -1;
    }

    HighsModel presolved_model = highs.getPresolvedModel();
    presolved.num_rows = presolved_model.lp_.num_row_;
    presolved.num_cols = presolved_model.lp_.num_col_;
    presolved.objective = presolved_model.lp_.col_cost_;
    presolved.lower = presolved_model.lp_.col_lower_;
    presolved.upper = presolved_model.lp_.col_upper_;
    presolved.rhs = presolved_model.lp_.row_lower_; // assuming row_lower_ == row_upper_
    // Populate row senses and range info
    presolved.row_sense.resize(presolved.num_rows);
    presolved.range_rows.clear();
    presolved.range_value.clear();
    const auto &row_lower = presolved_model.lp_.row_lower_;
    const auto &row_upper = presolved_model.lp_.row_upper_;
    for (i_t i = 0; i < presolved.num_rows; ++i) {
        const f_t lb = row_lower[i];
    presolved.A.m = presolved.num_rows;
    presolved.A.n = presolved.num_cols;
    presolved.A.nz_max = static_cast<i_t>(presolved.A.i.size());
        const f_t ub = row_upper[i];
        if (lb == ub) {
            presolved.row_sense[i] = 'E';
            presolved.rhs[i] = lb;
        } else if (ub == std::numeric_limits<f_t>::infinity()) {
            presolved.row_sense[i] = 'G';
            presolved.rhs[i] = lb;
        } else if (lb == -std::numeric_limits<f_t>::infinity()) {
            presolved.row_sense[i] = 'L';
            presolved.rhs[i] = ub;
        } else {
            presolved.row_sense[i] = 'E';
            presolved.rhs[i] = lb;
            presolved.range_rows.push_back(i);
            presolved.range_value.push_back(ub - lb);
        }
    }
    presolved.num_range_rows = static_cast<i_t>(presolved.range_rows.size());
    // Convert HiGHS internal sparse matrix back to our CSC matrix format
    presolved.A.col_start.assign(presolved_model.lp_.a_matrix_.start_.begin(),
                                 presolved_model.lp_.a_matrix_.start_.end());
    presolved.A.i.assign(presolved_model.lp_.a_matrix_.index_.begin(),
                         presolved_model.lp_.a_matrix_.index_.end());
    presolved.A.x.assign(presolved_model.lp_.a_matrix_.value_.begin(),
                         presolved_model.lp_.a_matrix_.value_.end());
    presolved.obj_constant = presolved_model.lp_.offset_;
    presolved.obj_scale = (presolved_model.lp_.sense_ == ObjSense::kMinimize) ? 1.0 : -1.0;

    return 0;
}

// Explicit instantiation for the commonly used types to ensure the symbol
// is emitted in this translation unit.
template class highs_presolve_t<int, double>;


}