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

#include "../routing_helpers.cuh"

#include <array>
#include <csignal>
#include <cstdio>
#define DEPO 0

// Array indexes for evaluation of distinct features
// distance
constexpr int DIST = 0;
// time
constexpr int TIME = 1;
// pdp capacity ( positive/negative supply )
constexpr int CAP = 2;

constexpr int PRIZE = 3;

constexpr int TASKS = 4;

constexpr int SERVICE_TIME = 5;

constexpr int MISMATCH = 6;

constexpr int BREAK = 7;

constexpr int VEHICLE_FIXED_COST = 8;

constexpr int NDIM = 9;

#define MACHINE_EPSILON 0.000001
#define MOVE_EPSILON    0.0001

//! A type storing all dimensions data of a solution
using costs = std::array<double, NDIM>;

inline double apply_costs(const costs& in, const costs& weights) noexcept
{
  double cost = 0.;
  cuopt::routing::detail::constexpr_for<0, NDIM, 1>([&](auto I) { cost += in[I] * weights[I]; });
  return cost;
}

// #define RUNTIME_RUNTIME_TEST

#ifdef RUNTIME_RUNTIME_TEST
#define INVARIANT(__) (__);
#define RUNTIME_TEST(__)                                               \
  if (!(__)) {                                                         \
    printf("test invalid: %s, %s:%d\n", __func__, __FILE__, __LINE__); \
    std::raise(SIGTRAP);                                               \
  }
#else
#define RUNTIME_TEST(_)
#define INVARIANT(_)
#endif
