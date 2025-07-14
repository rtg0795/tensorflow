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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "xla/codegen/math/erf.h"
#include "xla/codegen/math/exp.h"
#include "xla/codegen/math/fptrunc.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/ldexp.h"
#include "xla/codegen/math/log1p.h"
#include "xla/codegen/math/string_interner.h"
#include "xla/codegen/math/vec_name_mangler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {
class LdexpF64MathFunction final : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return "ldexp"; }
  std::vector<std::string> TargetFunctions() const override {
    return {"xla.ldexp.f64.i32"};
  }
  std::vector<std::vector<Intrinsic::Type>> SupportedVectorTypes()
      const override {
    return {{Intrinsic::S(xla::F64), Intrinsic::S(xla::S32)},
            {Intrinsic::V(xla::F64, 2), Intrinsic::V(xla::S32, 2)},
            {Intrinsic::V(xla::F64, 4), Intrinsic::V(xla::S32, 4)},
            {Intrinsic::V(xla::F64, 8), Intrinsic::V(xla::S32, 8)}};
  }

  llvm::Function* CreateDefinition(
      llvm::Module& module, absl::string_view name,
      absl::Span<const Intrinsic::Type> types) const override {
    CHECK(types.size() == 2) << "Expected 2 types for ldexp";
    CHECK_OK(Intrinsic::VerifySameWidth(types[0], types[1]));
    return math::CreateLdexpF64(
        &module, Intrinsic::TypeToIrType(types.front(), module.getContext()));
  }
};

class ExpF64MathFunction final : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return "exp"; }
  std::vector<std::string> TargetFunctions() const override {
    return {"xla.exp.f64"};
  }
  std::vector<std::vector<Intrinsic::Type>> SupportedVectorTypes()
      const override {
    return {
        {Intrinsic::S(xla::F64)},
        {Intrinsic::V(xla::F64, 2)},
        {Intrinsic::V(xla::F64, 4)},
        {Intrinsic::V(xla::F64, 8)},
    };
  }

  llvm::Function* CreateDefinition(
      llvm::Module& module, absl::string_view name,
      absl::Span<const Intrinsic::Type> types) const override {
    CHECK(types.size() == 1) << "Expected 1 type for exp";
    return math::CreateExpF64(
        &module, Intrinsic::TypeToIrType(types.front(), module.getContext()));
  }
};

class FpextF32ToBf16MathFunction final : public MathFunction {
 public:
  absl::string_view FunctionName() const override {
    return "xla.fptrunc.f32.to.bf16";
  }

  std::vector<std::string> TargetFunctions() const override {
    return {"xla.fptrunc.f32.to.bf16"};
  }

  std::vector<std::vector<Intrinsic::Type>> SupportedVectorTypes()
      const override {
    return {
        {Intrinsic::S(xla::F32)},
        {Intrinsic::V(xla::F32, 2)},
        {Intrinsic::V(xla::F32, 4)},
        {Intrinsic::V(xla::F32, 8)},
    };
  }

  llvm::Function* CreateDefinition(
      llvm::Module& module, absl::string_view name,
      absl::Span<const Intrinsic::Type> types) const override {
    CHECK(types.size() == 2) << "Expected [from, to] type for fptrunc";
    auto from = types[0];
    auto to = types[1];
    return Intrinsic::FpTrunc::CreateDefinition(&module, from, to).value();
  }
};

template <PrimitiveType Type>
class Log1pMathFunction final : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return "xla.log1p"; }

  std::vector<std::string> TargetFunctions() const override {
    return {Intrinsic::Log1p::Name(Intrinsic::S(Type))};
  }

  std::vector<std::vector<Intrinsic::Type>> SupportedVectorTypes()
      const override {
    return {
        {Intrinsic::S(Type)},
        {Intrinsic::V(Type, 2)},
        {Intrinsic::V(Type, 4)},
        {Intrinsic::V(Type, 8)},
    };
  }

  llvm::Function* CreateDefinition(
      llvm::Module& module, absl::string_view name,
      absl::Span<const Intrinsic::Type> types) const override {
    CHECK(types.size() == 1) << "Expected 1 type for log1p";
    return Intrinsic::Log1p::CreateDefinition(&module, types.front()).value();
  }
};

class ErfF32MathFunction final : public MathFunction {
 public:
  absl::string_view FunctionName() const override { return "xla.erf"; }

  std::vector<std::string> TargetFunctions() const override {
    return {Intrinsic::Erf::Name(Intrinsic::S(F32))};
  }

  std::vector<std::vector<Intrinsic::Type>> SupportedVectorTypes()
      const override {
    return {{Intrinsic::S(F32)},
            {Intrinsic::V(F32, 2)},
            {Intrinsic::V(F32, 4)},
            {Intrinsic::V(F32, 8)}};
  }

  llvm::Function* CreateDefinition(
      llvm::Module& module, absl::string_view name,
      absl::Span<const Intrinsic::Type> types) const override {
    CHECK(types.size() == 1) << "Expected 1 type for erf";
    return Intrinsic::Erf::CreateDefinition(&module, types.front()).value();
  }
};

MathFunctionLib::MathFunctionLib() {
  math_functions_.push_back(std::make_unique<LdexpF64MathFunction>());
  math_functions_.push_back(std::make_unique<ExpF64MathFunction>());
  math_functions_.push_back(std::make_unique<FpextF32ToBf16MathFunction>());
  math_functions_.push_back(std::make_unique<Log1pMathFunction<F16>>());
  math_functions_.push_back(std::make_unique<Log1pMathFunction<F32>>());
  math_functions_.push_back(std::make_unique<Log1pMathFunction<F64>>());
  math_functions_.push_back(std::make_unique<ErfF32MathFunction>());
}

namespace {

// Iterate all function calls in LLVM IR and call callback.
void VisitFunctionCalls(llvm::Module& module,
                        std::function<void(llvm::CallInst&)> callback) {
  for (llvm::Function& function : module) {
    for (llvm::BasicBlock& block : function) {
      for (llvm::Instruction& inst : block) {
        if (llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          callback(*call);
        }
      }
    }
  }
}

// Returns the VecCallInfo that we need to generate definitions for all calls
// to math approximations in the module. Assumes that the module has already
// been optimized and that all calls to math approximations are unary.
absl::flat_hash_map<absl::string_view, absl::flat_hash_set<PrimitiveType>>
GetCalledApproximatableFunctions(
    llvm::Module& module,
    absl::flat_hash_map<absl::string_view, absl::string_view> targets) {
  absl::flat_hash_map<absl::string_view, absl::flat_hash_set<PrimitiveType>>
      called_targets;
  VisitFunctionCalls(module, [&](const llvm::CallInst& call) {
    if (auto it = targets.find(call.getCalledFunction()->getName());
        it != targets.end()) {
      called_targets[it->second].insert(
          llvm_ir::PrimitiveTypeFromIrType(call.getArgOperand(0)->getType()));
    }
  });
  return called_targets;
}

}  // anonymous namespace

std::vector<llvm::VecDesc> MathFunctionLib::Vectorizations() {
  std::vector<llvm::VecDesc> vec_descs;
  for (const auto& math_func : math_functions_) {
    for (const std::string& target_function : math_func->TargetFunctions()) {
      absl::string_view target_function_interned =
          math::StringInterner::Get().Intern(target_function);
      for (const auto& vector_types : math_func->SupportedVectorTypes()) {
        absl::string_view vec_name = math::StringInterner::Get().Intern(
            math_func->GenerateVectorizedFunctionName(vector_types));
        size_t vector_width = vector_types.front().vector_width().value_or(1);
        llvm::VecDesc vec_desc = {
            target_function_interned,
            vec_name,
            llvm::ElementCount::getFixed(vector_width),
            false,
            math::StringInterner::Get().Intern(
                math_func->GenerateMangledSimdPrefix(vector_types)),
            std::nullopt};
        vec_descs.push_back(vec_desc);
        targets_[vec_name] = math_func->FunctionName();
      }
    }
  }
  return vec_descs;
}

void CreateDefinitionAndReplaceDeclaration(
    llvm::Module& module, absl::string_view name,
    absl::Span<const Intrinsic::Type> types, MathFunction& math_func) {
  // The Vectorization pass may have already inserted a declaration
  // of this function that we need to rename and later remove to avoid
  // name collisions.
  llvm::Function* existing_func = module.getFunction(name);
  if (existing_func && existing_func->isDeclaration()) {
    existing_func->setName(std::string(name) + ".old_decl");
  }
  llvm::Function* definition = math_func.CreateDefinition(module, name, types);
  definition->setLinkage(llvm::Function::InternalLinkage);
  definition->addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::verifyFunction(*definition);
  if (existing_func && existing_func->isDeclaration()) {
    // Remove the declaration and replace all uses with the
    // new definition.
    existing_func->replaceAllUsesWith(definition);
    existing_func->eraseFromParent();
  }
}

absl::flat_hash_set<absl::string_view> MathFunctionLib::RewriteMathFunctions(
    llvm::Module& module) {
  // Find each called target function, generate the definition and insert it
  // into the module.
  // Keep track of the function names we replaced so we can remove them from
  // llvm.compiler.used later.
  absl::flat_hash_set<absl::string_view> replaced_functions;
  for (const auto& [function_name, dtypes] :
       GetCalledApproximatableFunctions(module, targets_)) {
    for (const auto& math_func : math_functions_) {
      if (math_func->FunctionName() == function_name) {
        for (const auto& types : math_func->SupportedVectorTypes()) {
          auto vector_type = types.front();
          // All types must have the same vector width.
          // This could be relaxed in the future.
          CHECK(std::all_of(types.begin(), types.end(),
                            [&](const Intrinsic::Type& type) {
                              return type.vector_width() ==
                                     vector_type.vector_width();
                            }))
              << "All types must have the same vector width.";
          if (dtypes.contains(vector_type.element_type())) {
            absl::string_view name = math::StringInterner::Get().Intern(
                math_func->GenerateVectorizedFunctionName(types));
            CreateDefinitionAndReplaceDeclaration(module, name, types,
                                                  *math_func);
            replaced_functions.insert(name);
          }
        }
      }
    }
  }

  CHECK(!llvm::verifyModule(module)) << "Module is invalid after optimization\n"
                                     << llvm_ir::DumpToString(&module);
  return replaced_functions;
}

std::string MathFunction::GenerateVectorizedFunctionName(
    absl::Span<const Intrinsic::Type> types) const {
  std::vector<std::string> names;
  std::transform(types.begin(), types.end(), std::back_inserter(names),
                 std::mem_fn(&Intrinsic::Type::name));
  return absl::StrCat("xla.", FunctionName(), ".", absl::StrJoin(names, "."));
}
std::string MathFunction::GenerateMangledSimdPrefix(
    absl::Span<const Intrinsic::Type> types) const {
  std::vector<math::VecParamCardinality> param_cardinalities;
  std::vector<std::string> names;
  auto front = types.front();
  for (const auto& type : types) {
    if (type.is_scalar()) {
      param_cardinalities.push_back(math::VecParamCardinality::kScalar);
    } else {
      param_cardinalities.push_back(math::VecParamCardinality::kVector);
    }
    names.push_back(Intrinsic::ScalarName(type.element_type()));
    CHECK(type.vector_width() == front.vector_width())
        << "All types must have the same vector width.";
  }
  return absl::StrCat(
      math::GetMangledNamePrefix(IsMasked(), front.vector_width().value_or(1),
                                 param_cardinalities),
      "_", absl::StrJoin(names, "."));
}
}  // namespace xla::codegen
