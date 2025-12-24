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

#include "../structures.hpp"

namespace cuopt::routing::detail {

template <typename i_t, typename f_t>
struct adapted_node_t {
  //! Index in route
  size_t r_index{0};
  //! Route id
  size_t r_id{0};

  //! vehicle id
  size_t v_id{0};

  //! Global id of the node (as defined in adapted_problem_t<i_t, f_t>)
  NodeInfo<i_t> node_info;

  bool is_depot() const { return node_info.is_depot(); }

  int node_id() const { return node_info.node(); }
};

template <typename i_t, typename f_t>
bool operator==(const adapted_node_t<i_t, f_t>& lhs, const adapted_node_t<i_t, f_t>& rhs)
{
  bool equal = true;
  equal      = equal && (lhs.r_index == rhs.r_index);
  equal      = equal && (lhs.r_id == rhs.r_id);
  equal      = equal && (lhs.v_id == rhs.v_id);
  equal      = equal && (lhs.node_info == rhs.node_info);
  return equal;
}

template struct adapted_node_t<int, float>;
}  // namespace cuopt::routing::detail
