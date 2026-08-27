// Prefill Fabric - executor abstraction, device/worker descriptors, groups.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <array>
#include <memory>
#include <unordered_map>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"
#include "prefillfabric/request.hpp"
#include "prefillfabric/clock.hpp"
#include "prefillfabric/compatibility.hpp"

namespace prefillfabric {

struct DeviceDescriptor {
  DeviceId id;
  std::string name;
  std::string backend = kBackendCpu;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t available_memory_bytes = 0;
  std::string capability;      // e.g. "sm_120" for CUDA, "" for CPU
  bool available = false;

  DeviceDescriptor() = default;
  DeviceDescriptor(DeviceId d, std::string n, std::string be, std::uint64_t tot, std::uint64_t avail, std::string cap, bool availflag)
      : id(d), name(std::move(n)), backend(std::move(be)), total_memory_bytes(tot),
        available_memory_bytes(avail), capability(std::move(cap)), available(availflag) {}
};

// A worker in a distributed deployment. WorkerBootId changes on every real
// process restart and is part of the stale-authority check.
struct WorkerDescriptor {
  WorkerId worker_id;
  WorkerBootId boot_id;
  std::string host;
  std::uint32_t port = 0;
  std::string backend = kBackendAny;
  std::vector<DeviceDescriptor> devices;
  WorkerState state = WorkerState::unknown;
  std::uint64_t capacity_units = 1;
  std::string info;
};

// A single member of an execution group (one chunk of one request/attempt).
struct ExecutableMember {
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  Generation parent_generation;
  std::uint64_t token_start = 0;
  std::uint64_t token_count = 0;
  WorkKind work_kind = WorkKind::full_prompt;
  std::uint64_t memory_estimate_bytes = 0;
  std::uint64_t compute_estimate = 0;
  WorkerId assigned_worker;   // nil for in-process execution
  std::uint32_t member_index = 0;
};

// The unit of execution submitted to an Executor.
struct ExecutionGroup {
  GroupId group_id;
  CompatibilityKey key;
  std::string backend = kBackendAny;
  DeviceId device;
  std::vector<ExecutableMember> members;
  std::uint64_t total_token_count = 0;
  std::uint64_t max_memory_estimate_bytes = 0;   // max member memory
  std::uint64_t sum_compute_estimate = 0;

  std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(members.size()); }
  bool empty() const noexcept { return members.empty(); }
};

// Per-member terminal/continuation outcome. The group is allowed to have a
// mixture of outcomes (a failing member does not force the whole group to a
// single common fate).
struct MemberResult {
  RequestId request_id;
  AttemptId attempt_id;
  Generation generation;
  MemberOutcome outcome = MemberOutcome::success;
  std::uint64_t tokens_completed = 0;
  std::uint64_t digest = 0;               // verifiable output of the executor
  ErrorCode failure_code = ErrorCode::ok;
  std::string message;
  std::uint64_t next_token_start = 0;     // valid when needs_next_chunk
  bool requires_next_chunk = false;
  // Distributed authority the result was produced under. In-process drivers
  // stamp these from the local worker / current epoch before reporting.
  Epoch epoch;
  WorkerId worker;
  WorkerBootId boot;

  // Convenience states used by the scheduler.
  bool is_terminal_success() const noexcept { return outcome == MemberOutcome::success && !requires_next_chunk; }
  bool is_success() const noexcept { return outcome == MemberOutcome::success; }
};

struct ExecutorResult {
  bool group_succeeded = false;
  std::vector<MemberResult> members;
  std::string executor_family;
  Nanoseconds elapsed_ns = 0;
  std::string status_message;
};

// Resolves actual token payloads for a request/attempt token range. The
// scheduler does not interpret token contents; only executors consume them.
// resolve_tokens returns a stable pointer valid for the duration of execute().
class TokenResolver {
 public:
  virtual ~TokenResolver() = default;
  virtual Result<const std::uint32_t*> resolve_tokens(RequestId r, AttemptId a,
                                                        Generation g, std::uint64_t offset,
                                                        std::uint64_t count) const = 0;
};

// Simple in-memory token resolver backed by a vector of token vectors.
class InMemoryTokenResolver : public TokenResolver {
 public:
  // Registers a request's token data. Returns ok, or invalid_argument if the
  // data length is shorter than the request's prompt.
  Result<void> register_request(const PrefillRequest& req, std::vector<std::uint32_t> tokens) {
    const auto n = req.prompt_token_count;
    if (tokens.size() != n)
      return Result<void>::err(ErrorCode::invalid_argument, "token buffer length != prompt_token_count");
    tokens_[req.request_id] = std::move(tokens);
    return Result<void>::ok();
  }

  Result<const std::uint32_t*> resolve_tokens(RequestId r, AttemptId, Generation,
                                               std::uint64_t offset,
                                               std::uint64_t count) const override {
    const auto it = tokens_.find(r);
    if (it == tokens_.end() || it->second.size() < offset + count)
      return Result<const std::uint32_t*>::err(ErrorCode::unknown_request, "no token data for request");
    return Result<const std::uint32_t*>::ok(it->second.data() + static_cast<std::size_t>(offset));
  }

  bool contains(RequestId r) const noexcept { return tokens_.find(r) != tokens_.end(); }

 private:
  std::unordered_map<RequestId, std::vector<std::uint32_t>, HashStrongId> tokens_;
};

// VENDOR-NEUTRAL executor contract. CUDA is one concrete backend behind this
// interface; the runtime core never refers to CUDA types directly.
class Executor {
 public:
  virtual ~Executor() = default;

  // Execute a formed group and produce per-member results. Must perform real
  // bounded numerical work and produce verifiable outputs. Must not block the
  // scheduler on anything that could self-deadlock.
  virtual Result<ExecutorResult> execute(const ExecutionGroup& group,
                                         const TokenResolver& tokens) = 0;

  virtual std::string name() const = 0;
  virtual bool available() const = 0;
  virtual DeviceDescriptor device() const = 0;
  virtual std::string backend() const = 0;
};

// A factory used to construct the vendor-neutral executable frontend.
class ExecutorFactory {
 public:
  virtual ~ExecutorFactory() = default;
  virtual std::string name() const = 0;
  virtual bool available() const = 0;
  virtual std::unique_ptr<Executor> create() = 0;
};

}  // namespace prefillfabric
