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

#include <dlfcn.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/dynamic_loading.h"

namespace litert {
namespace google_tensor {

Adapter::Adapter() : api_(new Api) {}

Adapter::~Adapter() {
  if (dlib_handle_) {
    litert::internal::CloseLib(dlib_handle_);
  }
}

litert::Expected<Adapter::Ptr> Adapter::Create(
    std::optional<std::string> shared_library_dir) {
  Ptr adapter(new Adapter);  // Owned by the caller.
  if (auto status = adapter->LoadSymbols(shared_library_dir); !status) {
    return status.Error();
  }
  return adapter;
}

litert::Expected<void> Adapter::LoadSymbols(
    std::optional<std::string> shared_library_dir) {
  constexpr auto kLibTensorTPUCompiler = "libcompiler_api_wrapper.so";

  const std::vector<std::string> so_paths = {
      shared_library_dir.has_value()
          ? absl::StrCat(*shared_library_dir, "/", kLibTensorTPUCompiler)
          : kLibTensorTPUCompiler};

  if (litert::internal::OpenLib(so_paths, &dlib_handle_) != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to load Tensor TPU compiler library");
  }

  api_->compile =
      reinterpret_cast<Compile>(dlsym(dlib_handle_, "CompileFlatbuffer"));
  if (!api_->compile) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to load Tensor TPU compiler API");
  }

  LITERT_LOG(LITERT_INFO, "Tensor TPU compiler API symbols loaded");
  return {};
}

}  // namespace google_tensor
}  // namespace litert
