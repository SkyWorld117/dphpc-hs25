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

#include <chrono>
#include <string>

namespace cuopt {

// TODO extend this for the whole solver.
// we are currently using this for diversity and adapters
class timer_t {
  using steady_clock = std::chrono::steady_clock;

 public:
  timer_t()               = delete;
  timer_t(const timer_t&) = default;
  timer_t(double time_limit_)
  {
    time_limit = time_limit_;
    begin      = steady_clock::now();
  }

  void print_debug(std::string msg) const
  {
    printf("%s time_limit: %f remaining_time: %f elapsed_time: %f \n",
           msg.c_str(),
           time_limit,
           remaining_time(),
           elapsed_time());
  }

  bool check_time_limit() const noexcept { return elapsed_time() >= time_limit; }

  bool check_half_time() const noexcept { return elapsed_time() >= time_limit / 2; }

  double elapsed_time() const noexcept
  {
    return std::chrono::duration<double>(steady_clock::now() - begin).count();
  }

  double remaining_time() const noexcept
  {
    return std::max<double>(0.0, time_limit - elapsed_time());
  }

  double clamp_remaining_time(double desired_time) const noexcept
  {
    return std::min<double>(desired_time, remaining_time());
  }

  double get_time_limit() const noexcept { return time_limit; }

 private:
  double time_limit;
  steady_clock::time_point begin;
};

}  // namespace cuopt
