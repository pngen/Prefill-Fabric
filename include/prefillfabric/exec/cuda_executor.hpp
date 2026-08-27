// Prefill Fabric - CUDA executor (vendor-neutral behind Executor).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <memory>
#include "prefillfabric/executor.hpp"

namespace prefillfabric {

// CUDA backend executor. Performs real device allocation, transfer, kernel
// launch, synchronization, result verification, and cleanup. On a non-CUDA
// build available() returns false and execute() reports cuda_unavailable; a
// required CUDA validation test never silently skips.
class CudaExecutor : public Executor {
 public:
  CudaExecutor();
  explicit CudaExecutor(int device_index);
  ~CudaExecutor() override = default;

  Result<ExecutorResult> execute(const ExecutionGroup& group,
                                 const TokenResolver& tokens) override;
  std::string name() const override { return "cuda"; }
  bool available() const override;
  DeviceDescriptor device() const override;
  std::string backend() const override { return kBackendCuda; }

  static Result<DeviceDescriptor> query_device(int device_index);
  static bool cuda_available();

 private:
  int device_index_;
  std::uint64_t baseline_avail_mem_ = 0;
};

bool cuda_executor_available() noexcept;
std::shared_ptr<Executor> create_cuda_executor();

}  // namespace prefillfabric
