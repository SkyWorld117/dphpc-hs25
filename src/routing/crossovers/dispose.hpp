/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../diversity/helpers.hpp"

namespace cuopt {
namespace routing {

template <class Solution>
struct dispose {
  /*! \brief { Remove a random route from solution
   * } */
  using i_t = typename Solution::i_type;

  std::vector<i_t> removed_nodes;

  bool recombine(Solution& sol)
  {
    raft::common::nvtx::range fun_scope("dispose");
    if (sol.sol.get_n_routes() <= sol.sol.problem_ptr->data_view_ptr->get_min_vehicles()) {
      return false;
    }

    // In case of prize collection, we can have unserviced nodes, we should not dispose route when
    // we still have unserviced nodes unless we have an accompanying add route operator.
    // FIXME:: Implement add routes operator
    if (sol.has_unserviced_nodes && sol.sol.problem_ptr->has_prize_collection()) { return false; }

    std::vector<i_t> random_route{static_cast<i_t>(next_random() % sol.sol.get_n_routes())};
    removed_nodes = sol.get_nodes_of_routes(random_route);
    sol.remove_routes(random_route);
    return true;
  }
};

}  // namespace routing
}  // namespace cuopt
