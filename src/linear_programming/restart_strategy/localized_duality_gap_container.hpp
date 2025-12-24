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

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
struct localized_duality_gap_container_t {
 public:
  localized_duality_gap_container_t(raft::handle_t const* handle_ptr,
                                    i_t primal_size,
                                    i_t dual_size);

  struct view_t {
    /** size of primal problem */
    i_t primal_size;
    /** size of dual problem */
    i_t dual_size;

    f_t* lagrangian_value;
    f_t* lower_bound_value;
    f_t* upper_bound_value;
    f_t* distance_traveled;
    f_t* primal_distance_traveled;
    f_t* dual_distance_traveled;
    f_t* normalized_gap;

    f_t* primal_solution;
    f_t* dual_solution;
    f_t* primal_gradient;
    f_t* dual_gradient;
    f_t* primal_solution_tr;
    f_t* dual_solution_tr;
  };

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */
  view_t view();

  i_t primal_size_h_;
  i_t dual_size_h_;
  rmm::device_scalar<f_t> lagrangian_value_;
  rmm::device_scalar<f_t> lower_bound_value_;
  rmm::device_scalar<f_t> upper_bound_value_;
  rmm::device_scalar<f_t> distance_traveled_;
  rmm::device_scalar<f_t> primal_distance_traveled_;
  rmm::device_scalar<f_t> dual_distance_traveled_;
  rmm::device_scalar<f_t> normalized_gap_;

  rmm::device_uvector<f_t> primal_solution_;
  rmm::device_uvector<f_t> dual_solution_;
  rmm::device_uvector<f_t> primal_gradient_;
  rmm::device_uvector<f_t> dual_gradient_;
  rmm::device_uvector<f_t> primal_solution_tr_;
  rmm::device_uvector<f_t> dual_solution_tr_;
};
}  // namespace cuopt::linear_programming::detail
