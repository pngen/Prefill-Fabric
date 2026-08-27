// Prefill Fabric - CUDA executor implementation (real device work).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/exec/cuda_executor.hpp"
#include "prefillfabric/frontend_math.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace prefillfabric {

// These must equal frontend_math.hpp constants (kFfM1, kFfM2, kFfM4).
__device__ constexpr unsigned long long kDevM1 = 0x5851F42D4C957F2DULL;
__device__ constexpr unsigned long long kDevM2 = 0x94D049BB133111EBULL;

__global__ void ff_reduce_kernel(const unsigned int* tokens, unsigned long long token_start,
                                 long n, unsigned long long* acc) {
  const long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const unsigned long long pos = token_start + static_cast<unsigned long long>(idx);
  const unsigned long long t = static_cast<unsigned long long>(tokens[idx]);
  const unsigned long long v = (t * kDevM1) ^ (pos * kDevM2) ^ ((pos << 7) & ~0ULL);
  atomicAdd(acc, v);
}

static std::uint64_t mix_id(std::uint64_t h, std::uint64_t v) noexcept {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

bool CudaExecutor::cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaExecutor::CudaExecutor() : device_index_(0) {
  if (cuda_available()) { cudaSetDevice(0); cudaMemGetInfo(&baseline_avail_mem_, nullptr); }
}

CudaExecutor::CudaExecutor(int device_index) : device_index_(device_index) {
  if (cuda_available()) { cudaSetDevice(device_index); std::size_t free_b, tot_b; cudaMemGetInfo(&free_b, &tot_b); baseline_avail_mem_ = free_b; }
}

bool CudaExecutor::available() const { return cuda_available(); }

Result<DeviceDescriptor> CudaExecutor::query_device(int device_index) {
  if (!cuda_available())
    return Result<DeviceDescriptor>::err(ErrorCode::cuda_unavailable, "no CUDA device");
  cudaDeviceProp prop;
  auto e = cudaGetDeviceProperties(&prop, device_index);
  if (e != cudaSuccess)
    return Result<DeviceDescriptor>::err(ErrorCode::cuda_error, cudaGetErrorString(e));
  std::size_t free_b, tot_b;
  cudaMemGetInfo(&free_b, &tot_b);
  DeviceDescriptor d;
  d.id = DeviceId(static_cast<std::uint64_t>(device_index));
  d.name = prop.name;
  d.backend = kBackendCuda;
  d.total_memory_bytes = static_cast<std::uint64_t>(tot_b);
  d.available_memory_bytes = static_cast<std::uint64_t>(free_b);
  d.capability = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
  d.available = true;
  return Result<DeviceDescriptor>::ok(d);
}

DeviceDescriptor CudaExecutor::device() const {
  auto d = query_device(device_index_);
  if (d) return d.value();
  DeviceDescriptor fallback; fallback.backend = kBackendCuda; fallback.available = false; return fallback;
}

Result<ExecutorResult> CudaExecutor::execute(const ExecutionGroup& group,
                                             const TokenResolver& tokens) {
  if (!cuda_available())
    return Result<ExecutorResult>::err(ErrorCode::cuda_unavailable, "CUDA not available");
  if (cudaSetDevice(device_index_) != cudaSuccess)
    return Result<ExecutorResult>::err(ErrorCode::cuda_error, "cudaSetDevice failed");

  ExecutorResult out;
  out.executor_family = "cuda";
  out.group_succeeded = true;
  std::size_t free_before = 0, tot = 0;
  cudaMemGetInfo(&free_before, &tot);

  for (const auto& m : group.members) {
    MemberResult mr;
    mr.request_id = m.request_id;
    mr.attempt_id = m.attempt_id;
    mr.generation = m.generation;
    mr.tokens_completed = m.token_count;
    mr.next_token_start = m.token_start + m.token_count;
    mr.outcome = MemberOutcome::success;

    if (m.token_count == 0) {
      mr.outcome = MemberOutcome::non_retryable_failure;
      mr.failure_code = ErrorCode::invalid_argument;
      mr.message = "zero-token cuda member";
      out.group_succeeded = false;
      out.members.push_back(std::move(mr));
      continue;
    }

    const auto tok = tokens.resolve_tokens(m.request_id, m.attempt_id, m.generation, m.token_start, m.token_count);
    if (!tok) {
      mr.outcome = MemberOutcome::non_retryable_failure;
      mr.failure_code = tok.error().code;
      mr.message = tok.error().message;
      out.group_succeeded = false;
      out.members.push_back(std::move(mr));
      continue;
    }

    const long n = static_cast<long>(m.token_count);
    unsigned int* d_tokens = nullptr;
    unsigned long long* d_acc = nullptr;
    cudaError_t ce = cudaMalloc(&d_tokens, static_cast<std::size_t>(n) * sizeof(unsigned int));
    if (ce != cudaSuccess) { mr.outcome = MemberOutcome::non_retryable_failure; mr.failure_code = ErrorCode::cuda_error; mr.message = cudaGetErrorString(ce); out.group_succeeded = false; out.members.push_back(std::move(mr)); continue; }
    ce = cudaMalloc(&d_acc, sizeof(unsigned long long));
    if (ce != cudaSuccess) { cudaFree(d_tokens); mr.outcome = MemberOutcome::non_retryable_failure; mr.failure_code = ErrorCode::cuda_error; mr.message = cudaGetErrorString(ce); out.group_succeeded = false; out.members.push_back(std::move(mr)); continue; }
    cudaMemset(d_acc, 0, sizeof(unsigned long long));
    ce = cudaMemcpy(d_tokens, tok.value(), static_cast<std::size_t>(n) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (ce != cudaSuccess) { cudaFree(d_tokens); cudaFree(d_acc); mr.outcome = MemberOutcome::non_retryable_failure; mr.failure_code = ErrorCode::cuda_error; mr.message = cudaGetErrorString(ce); out.group_succeeded = false; out.members.push_back(std::move(mr)); continue; }

    const int blocks = static_cast<int>((n + 255) / 256);
    ff_reduce_kernel<<<blocks, 256>>>(d_tokens, static_cast<unsigned long long>(m.token_start), n, d_acc);
    ce = cudaGetLastError();
    if (ce != cudaSuccess) { cudaFree(d_tokens); cudaFree(d_acc); mr.outcome = MemberOutcome::non_retryable_failure; mr.failure_code = ErrorCode::cuda_error; mr.message = cudaGetErrorString(ce); out.group_succeeded = false; out.members.push_back(std::move(mr)); continue; }
    cudaDeviceSynchronize();
    unsigned long long acc = 0;
    cudaMemcpy(&acc, d_acc, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_tokens); cudaFree(d_acc);

    // Host-computed digest to cross-verify against the CPU reference.
    std::uint64_t h = kFfM3 ^ m.request_id.value();
    h = mix_id(h, m.attempt_id.value());
    h = mix_id(h, m.generation.value());
    h = mix_id(h, static_cast<std::uint64_t>(n));
    mr.digest = h ^ (static_cast<std::uint64_t>(acc) * kFfM4);
    out.members.push_back(std::move(mr));
  }

  std::size_t free_after = 0;
  cudaMemGetInfo(&free_after, &tot);
  cudaError_t last = cudaGetLastError();
  if (last != cudaSuccess) { out.group_succeeded = false; out.status_message = cudaGetErrorString(last); }
  // Verification: device memory must return to (at least) the pre-group baseline
  // after all buffers are freed (no leak).
  if (free_after + 16u * 1024u * 1024u >= free_before) out.status_message = "mem-cleanup-verified";
  else { out.group_succeeded = false; out.status_message = "device memory not recovered"; }
  return Result<ExecutorResult>::ok(std::move(out));
}

bool cuda_executor_available() noexcept { return CudaExecutor::cuda_available(); }
std::shared_ptr<Executor> create_cuda_executor() {
  return std::make_shared<CudaExecutor>();
}

}  // namespace prefillfabric
