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

#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"

#include <routing/arc_value.hpp>
#include <routing/fleet_info.hpp>
#include <routing/routing_details.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class prize_node_t {
 public:
  double prize          = 0.0;
  double prize_forward  = 0.0;
  double prize_backward = 0.0;

  /*! \brief { Calculate next node forward time data based on actual node} */
  void HDI calculate_forward(prize_node_t& next, [[maybe_unused]] double dummy = 0.) const noexcept
  {
    next.prize_forward = prize_forward + next.prize;
  }

  /*! \brief { Calculate prev node time backward data based on actual node} */
  void HDI calculate_backward(prize_node_t& prev, [[maybe_unused]] double dummy = 0.) const noexcept
  {
    prev.prize_backward = prize_backward + prize;
  }

  /*! \brief  { Combine information from begining and ending fragments.}
      \return { Time excess of route represented by nodes prev and next }*/
  static HDI double combine([[maybe_unused]] const prize_node_t& prev,
                            [[maybe_unused]] const prize_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double time_between) noexcept
  {
    return 0.;
  }

  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double weight       = 1.0,
                            [[maybe_unused]] double excess_limit = 0.) const
  {
    return true;
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             [[maybe_unused]] double weight       = 1.0,
                             [[maybe_unused]] double excess_limit = 0.) const
  {
    return true;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const prize_node_t& prev_node,
                    [[maybe_unused]] const VehicleInfo<f_t, is_device>& vehicle_info,
                    [[maybe_unused]] const prize_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
  {
    obj_cost[objective_t::PRIZE] = -(prize_forward + prize_backward);
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
