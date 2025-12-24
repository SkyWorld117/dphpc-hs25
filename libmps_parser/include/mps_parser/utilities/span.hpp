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

#include <cstddef>

namespace cuopt::mps_parser {

template <typename T>
class span {
 public:
  span() = default;
  span(T* ptr, std::size_t size) : ptr_(ptr), size_(size) {}
  std::size_t size() const noexcept { return size_; }
  const T* data() const noexcept { return ptr_; }

 private:
  T* ptr_{nullptr};
  std::size_t size_{0};
};

}  // namespace cuopt::mps_parser
