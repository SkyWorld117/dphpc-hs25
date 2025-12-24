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

#include <routing/fleet_info.hpp>
#include <routing/order_info.hpp>
namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
void populate_demand_container(data_model_view_t<i_t, f_t> const& data_model,
                               fleet_info_t<i_t, f_t>& fleet_info,
                               order_info_t<i_t, f_t>& order_info);
}
}  // namespace routing
}  // namespace cuopt
