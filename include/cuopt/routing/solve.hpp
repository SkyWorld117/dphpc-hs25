/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>
namespace cuopt {
namespace routing {

/**
 * @brief Routing solve function
 *
 * @tparam i_t
 * @tparam f_t
 * @param[in] data_model  input data model of type data_model_view_type
 * @param[in] settings    solver settings of type solver_settings_t
 * @return assignment_t<i_t> owning container for the solver output
 */
template <typename i_t, typename f_t>
assignment_t<i_t> solve(
  data_model_view_t<i_t, f_t> const& data_model,
  solver_settings_t<i_t, f_t> const& settings = solver_settings_t<i_t, f_t>{});
}  // namespace routing
}  // namespace cuopt
