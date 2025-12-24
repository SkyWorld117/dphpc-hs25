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

#include <utilities/cuda_helpers.cuh>

namespace cuopt {
namespace routing {
namespace detail {

struct _nullopt {
  explicit constexpr _nullopt() {}
} const nullopt;

template <typename T>
class optional {
 public:
  HD optional(const T& value) : is_set_(true), value_(value) {}

  HD optional(_nullopt) noexcept : is_set_(false) {}

  HD optional() noexcept : optional(nullopt) {}

  HD optional(const optional& other)
    : is_set_(other.is_set_), value_(other.is_set_ ? other.value_ : T{})
  {
  }

  HD optional& operator=(const optional& other)
  {
    is_set_ = other.is_set_;
    value_  = is_set_ ? other.value_ : T{};
    return *this;
  }

  HD explicit operator bool() const noexcept { return is_set_; }

  HD bool has_value() const noexcept { return is_set_; };

  HD T& value()
  {
    cuopt_assert(has_value(), "Calling optional value without value");
    return value_;
  }

  HD const T& value() const
  {
    cuopt_assert(has_value(), "Calling optional value without value");
    return value_;
  }

  HD const T& operator*() const { return value_; }
  HD T& operator*() { return value_; }

 private:
  bool is_set_;
  T value_;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
