// Prefill Fabric - CPU executor (real, verifiable bounded numerical work).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <memory>
#include "prefillfabric/executor.hpp"

namespace prefillfabric {

// Executor that performs deterministic bounded prefill-like numerical work
// over the provided token payloads and produces verifiable digests. Used for
// portable correctness, state-machine, concurrency and multiprocess validation.
class CpuExecutor : public Executor {
 public:
  CpuExecutor();
  explicit CpuExecutor(DeviceDescriptor device);

  Result<ExecutorResult> execute(const ExecutionGroup& group,
                                 const TokenResolver& tokens) override;

  std::string name() const override { return "cpu"; }
  bool available() const override { return true; }
  DeviceDescriptor device() const override { return device_; }
  std::string backend() const override { return kBackendCpu; }

 private:
  DeviceDescriptor device_;
};

// Factory registered so the runtime can construct a CPU executor generically.
class CpuExecutorFactory : public ExecutorFactory {
 public:
  std::string name() const override { return "cpu"; }
  bool available() const override { return true; }
  std::unique_ptr<Executor> create() override { return std::make_unique<CpuExecutor>(); }
};

}  // namespace prefillfabric
