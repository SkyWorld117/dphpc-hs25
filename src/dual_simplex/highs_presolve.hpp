#pragma once

#include <dual_simplex/user_problem.hpp>
#include <dual_simplex/presolve.hpp>
#include <highs/Highs.h>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class highs_presolve_t {
public:
    highs_presolve_t() = default;

    i_t presolve_from_file(const std::string& filename,
                        user_problem_t<i_t, f_t>& presolved_problem);

private:
    Highs highs;
};

}  // namespace cuopt::linear_programming::dual_simplex