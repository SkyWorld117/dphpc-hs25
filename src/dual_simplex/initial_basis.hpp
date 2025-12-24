/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {

enum class variable_status_t : int8_t {
  BASIC          = 0,
  NONBASIC_LOWER = -1,
  NONBASIC_UPPER = 1,
  NONBASIC_FREE  = 2,
  NONBASIC_FIXED = 3,
  SUPERBASIC     = 4
};

template <typename i_t, typename f_t>
i_t initial_basis_selection(const lp_problem_t<i_t, f_t>& problem,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<i_t>& candidate_columns,
                            f_t start_time,
                            std::vector<variable_status_t>& vstatus,
                            std::vector<i_t>& dependent_rows);

}  // namespace cuopt::linear_programming::dual_simplex
