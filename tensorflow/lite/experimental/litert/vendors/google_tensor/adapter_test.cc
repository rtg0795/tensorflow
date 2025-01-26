// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/adapter.h"

#include <sys/types.h>

#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"  // TODO(abhirs): remove this and use subgraphserialize
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert {
namespace google_tensor {

TEST(AdapterTest, CreateSuccess) {
  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }
  ASSERT_TRUE(adapter_result.HasValue());
}

TEST(AdapterTest, CreateFailure) {
  auto kLibDarwinnCompilerNoLib = "libcompiler_api_wrapper_no_lib.so";
  auto adapter_result = Adapter::Create(kLibDarwinnCompilerNoLib);
  ASSERT_FALSE(adapter_result.HasValue());
}

TEST(AdapterTest, CompileSuccess) {
  auto adapter_result = Adapter::Create(/*shared_library_dir=*/
                                        std::nullopt);
  if (!adapter_result.HasValue()) {
    LITERT_LOG(LITERT_ERROR, "Failed to create Adapter: %s",
               adapter_result.Error().Message().c_str());
  }
  ASSERT_TRUE(adapter_result.HasValue());
  auto adapter = std::move(adapter_result.Value());
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");
  LiteRtModel litert_model = model.Get();
  auto serialized = litert::internal::SerializeModel(std::move(*litert_model));
  ASSERT_TRUE(serialized);
  absl::string_view buffer(reinterpret_cast<const char*>(serialized->Data()),
                           serialized->Size());
  const char* soc_model = "P25";
  auto compile_result = adapter->api().compile(buffer, soc_model);
  ASSERT_TRUE(compile_result.ok());
  ASSERT_FALSE(compile_result->empty());
}

}  // namespace google_tensor
}  // namespace litert
