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

#include "xla/codegen/math_lib.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "xla/codegen/math/intrinsic.h"

namespace xla::codegen {
namespace {

class FakeMathFunction : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return "fake"; }
  std::vector<std::string> TargetFunctions() const override {
    return {"xla.fake"};
  }
  std::vector<std::vector<Intrinsic::Type>> SupportedVectorTypes()
      const override {
    return {
        {Intrinsic::S(xla::F64)},
        {Intrinsic::V(xla::F64, 2)},
        {Intrinsic::V(xla::F64, 4)},
    };
  }
  llvm::Function* CreateDefinition(
      llvm::Module& module, absl::string_view name,
      absl::Span<const Intrinsic::Type> types) const override {
    return nullptr;
  }
};

TEST(MathFunctionTest, SimdPrefix) {
  FakeMathFunction math_func;
  EXPECT_EQ(math_func.GenerateMangledSimdPrefix({Intrinsic::S(F32)}),
            "_ZGV_LLVM_N1s_f32");
  EXPECT_EQ(math_func.GenerateMangledSimdPrefix({Intrinsic::V(F32, 4)}),
            "_ZGV_LLVM_N4v_f32");
  EXPECT_EQ(math_func.GenerateMangledSimdPrefix({Intrinsic::V(F64, 4)}),
            "_ZGV_LLVM_N4v_f64");
  EXPECT_EQ(math_func.GenerateMangledSimdPrefix({Intrinsic::V(F32, 8)}),
            "_ZGV_LLVM_N8v_f32");
}

TEST(MathFunctionTest, VectorizedFunctionName) {
  FakeMathFunction math_func;
  EXPECT_EQ(math_func.GenerateVectorizedFunctionName({Intrinsic::S(F32)}),
            "xla.fake.f32");
  EXPECT_EQ(math_func.GenerateVectorizedFunctionName({Intrinsic::V(F32, 4)}),
            "xla.fake.v4f32");
  EXPECT_EQ(math_func.GenerateVectorizedFunctionName({Intrinsic::V(F64, 4)}),
            "xla.fake.v4f64");
}

}  // namespace
}  // namespace xla::codegen
