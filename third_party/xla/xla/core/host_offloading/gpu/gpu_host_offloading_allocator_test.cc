/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/core/host_offloading/gpu/gpu_host_offloading_allocator.h"

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

// Helper function to get memory types from cuda.
absl::StatusOr<cudaMemoryType> GetCudaMemoryType(void* ptr) {
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  // cudaErrorInvalidValue is expected for pageable host memory
  if (err != cudaSuccess) {
    return Internal("Failed to get CUDA memory type: %s",
                    cudaGetErrorString(err));
  }

  return attributes.type;
}

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

TEST(GpuHostOffloadingAllocatorTest, AllocateTransferBuffer) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  auto allocator = CreateGpuHostOffloadingAllocator(stream_executor);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, allocator->AllocateTransferBuffer(1024));
  EXPECT_EQ(buffer->size_bytes(), 1024);
  TF_ASSERT_OK_AND_ASSIGN(auto memory_type,
                          GetCudaMemoryType(buffer->untyped_data()));
  EXPECT_EQ(memory_type, cudaMemoryTypeHost);
}

TEST(GpuHostOffloadingAllocatorTest, AllocateStagingBuffer) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  auto allocator = CreateGpuHostOffloadingAllocator(stream_executor);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer, allocator->AllocateStagingBuffer(1024));
  EXPECT_EQ(buffer->size_bytes(), 1024);

  TF_ASSERT_OK_AND_ASSIGN(auto memory_type,
                          GetCudaMemoryType(buffer->untyped_data()));
  // Staging buffers are not registered with CUDA because they are allocated
  // with operator new.
  EXPECT_EQ(memory_type, cudaMemoryTypeUnregistered);
}

}  // namespace

}  // namespace xla::gpu
