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

#include <routing/hyper_params.hpp>

#include <routing/utilities/constants.hpp>

#include <string>

namespace cuopt {
namespace routing {
namespace detail {

// string overload
void inline set_if_env_set(std::string& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = std::string(str); }
}

// float overload
void inline set_if_env_set(float& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = std::stoi(str) / 1000.f; }
}

// i_t overload
void inline set_if_env_set(int& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = std::stoi(str); }
}

// bool overload
void inline set_if_env_set(bool& val, const char* env_var)
{
  const char* str = std::getenv(env_var);
  if (str != NULL) { val = bool(std::stoi(str)); }
}

hyper_params_t inline get_hyper_parameters_from_env()
{
  hyper_params_t hyper_params{};
  return hyper_params;
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
