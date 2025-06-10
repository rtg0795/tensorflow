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

#ifndef XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_PASS_OPTIONS_H_
#define XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_PASS_OPTIONS_H_

#include "llvm/Support/CommandLine.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Pass/PassOptions.h"

namespace xla {
namespace sdy {

struct StablehloExportPipelineOptions
  : public mlir::PassPipelineOptions<StablehloExportPipelineOptions> {
  Option<bool> exportShardingConstraintsToCopy{
    *this, "export-sharding-constraints-to-copy",
  llvm::cl::desc("Export sharding constraints to MHLO copy. "
                 "Else export them to StableHLO @Sharding custom calls."),
  llvm::cl::init(true)};
};

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_STABLEHLO_ROUND_TRIP_PASS_OPTIONS_H_
