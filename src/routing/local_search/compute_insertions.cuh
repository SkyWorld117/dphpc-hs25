/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../solution/solution.cuh"
#include "move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, request_t REQUEST>
void find_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                     move_candidates_t<i_t, f_t>& move_candidates,
                     search_type_t search_type = search_type_t::IMPROVE);

template <typename i_t, typename f_t, request_t REQUEST>
void find_unserviced_insertions(solution_t<i_t, f_t, REQUEST>& sol,
                                move_candidates_t<i_t, f_t>& move_candidates);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
