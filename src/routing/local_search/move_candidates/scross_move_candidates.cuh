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

#include <rmm/device_uvector.hpp>
#include "../../solution/solution_handle.cuh"
#include "cand.cuh"

namespace cuopt {
namespace routing {
namespace detail {

constexpr int max_cross_cand = 8;
constexpr int min_cross_cand = 1;

template <typename i_t, typename f_t>
class scross_move_candidates_t {
 public:
  scross_move_candidates_t(solution_handle_t<i_t, f_t> const* sol_handle_)
    : scross_best_cand_list(0, sol_handle_->get_stream()),
      route_pair_locks(0, sol_handle_->get_stream())
  {
  }

  void reset(solution_handle_t<i_t, f_t> const* sol_handle)
  {
    async_fill(scross_best_cand_list,
               cross_cand_t{0, 0, std::numeric_limits<double>::max(), 0, 0},
               sol_handle->get_stream());
    async_fill(route_pair_locks, 0, sol_handle->get_stream());
  }

  struct view_t {
    DI void insert_best_scross_candidate(i_t route_pair_idx, cross_cand_t cand)
    {
      if (cand < scross_best_cand_list[route_pair_idx]) {
        scross_best_cand_list[route_pair_idx] = cand;
      }
    }

    raft::device_span<cross_cand_t> scross_best_cand_list;
    raft::device_span<i_t> route_pair_locks;
  };

  view_t view()
  {
    view_t v;
    v.scross_best_cand_list =
      raft::device_span<cross_cand_t>{scross_best_cand_list.data(), scross_best_cand_list.size()};
    v.route_pair_locks = raft::device_span<i_t>{route_pair_locks.data(), route_pair_locks.size()};
    return v;
  }

  rmm::device_uvector<cross_cand_t> scross_best_cand_list;
  rmm::device_uvector<i_t> route_pair_locks;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
