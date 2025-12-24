/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <routing/routing_details.hpp>
#include <routing/structures.hpp>

#include <cuda_fp16.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename f_t>
constexpr f_t distance(const f_t px1, const f_t py1, const f_t px2, const f_t py2)
{
  f_t diff_x = (px1 - px2);
  f_t diff_y = (py1 - py2);
  return sqrtf(diff_x * diff_x + diff_y * diff_y);
}

template <typename i_t, typename f_t>
constexpr f_t euclidean_dist(const f_t* px, const f_t* py, const i_t a, const i_t b)
{
  f_t px_a   = static_cast<f_t>(px[a]);
  f_t px_b   = static_cast<f_t>(px[b]);
  f_t py_a   = static_cast<f_t>(py[a]);
  f_t py_b   = static_cast<f_t>(py[b]);
  f_t diff_x = (px_a - px_b);
  f_t diff_y = (py_a - py_b);
  return sqrt(diff_x * diff_x + diff_y * diff_y);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
