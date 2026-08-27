// Prefill Fabric - CUDA executor stub (compiled only when CUDA is disabled).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/exec/cuda_executor.hpp"

namespace prefillfabric {

CudaExecutor::CudaExecutor() : device_index_(0) {}
CudaExecutor::CudaExecutor(int d) : device_index_(d) {}

bool CudaExecutor::cuda_available() { return false; }
bool CudaExecutor::available() const { return false; }

Result<ExecutorResult> CudaExecutor::execute(const ExecutionGroup&, const TokenResolver&) {
  return Result<ExecutorResult>::err(ErrorCode::cuda_unavailable, "CUDA build disabled");
}

DeviceDescriptor CudaExecutor::device() const {
  DeviceDescriptor d; d.backend = kBackendCuda; d.available = false; return d;
}

Result<DeviceDescriptor> CudaExecutor::query_device(int) {
  return Result<DeviceDescriptor>::err(ErrorCode::cuda_unavailable, "no CUDA device");
}

bool cuda_executor_available() noexcept { return false; }
std::shared_ptr<Executor> create_cuda_executor() { return nullptr; }

}  // namespace prefillfabric
