/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuda_runtime.h>
#include <raft/util/cuda_utils.cuh>
#include <rmm/cuda_stream_view.hpp>

namespace cuopt {

class event_handler_t {
 public:
  event_handler_t() { RAFT_CUDA_TRY(cudaEventCreate(&event_)); }
  event_handler_t(unsigned int flags) { RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event_, flags)); }
  ~event_handler_t() { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_)); }

  event_handler_t(const event_handler_t&)            = delete;
  event_handler_t& operator=(const event_handler_t&) = delete;

  void record(rmm::cuda_stream_view stream_view)
  {
    RAFT_CUDA_TRY(cudaEventRecord(event_, stream_view));
  }

  void record_with_flags(rmm::cuda_stream_view stream_view, int flags)
  {
    RAFT_CUDA_TRY(cudaEventRecordWithFlags(event_, stream_view, flags));
  }

  void stream_wait(rmm::cuda_stream_view stream_view)
  {
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream_view, event_));
  }

  float elapsed_time_since_ms(const event_handler_t& start)
  {
    float ms;
    // TODO: use cudaEventElapsedTime_v2 with CUDA 12.8?
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
  }

  void synchronize() { RAFT_CUDA_TRY(cudaEventSynchronize(event_)); }

 private:
  cudaEvent_t event_;
};
}  // namespace cuopt
