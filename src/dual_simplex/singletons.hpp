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

#include <dual_simplex/sparse_matrix.hpp>

#include <numeric>
#include <queue>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
struct row_col_graph_t {
 public:
  typename std::vector<i_t>::iterator Xdeg;
  typename std::vector<i_t>::iterator Xperm;
  typename std::vector<i_t>::const_iterator Xp;
  typename std::vector<i_t>::const_iterator Xi;
  typename std::vector<i_t>::iterator Ydeg;
  typename std::vector<i_t>::iterator Yperm;
  typename std::vector<i_t>::const_iterator Yp;
  typename std::vector<i_t>::const_iterator Yi;
};

template <typename i_t>
i_t order_singletons(std::queue<i_t>& singleton_queue,
                     i_t& singletons_found,
                     row_col_graph_t<i_t>& G);

// \param [in,out]  workspace - size m
template <typename i_t, typename f_t>
void create_row_representationon(const csc_matrix_t<i_t, f_t>& A,
                                 std::vector<i_t>& row_start,
                                 std::vector<i_t>& col_index,
                                 std::vector<i_t>& workspace);
// Complete the permuation
template <typename i_t, typename f_t>
i_t complete_permutationn(i_t singletons, std::vector<i_t>& Xdeg, std::vector<i_t>& Xperm);

template <typename i_t, typename f_t>
i_t find_singletons(const csc_matrix_t<i_t, f_t>& A,
                    i_t& row_singletons,
                    std::vector<i_t>& row_perm,
                    i_t& col_singleton,
                    std::vector<i_t>& col_perm);

}  // namespace cuopt::linear_programming::dual_simplex
