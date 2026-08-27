// Prefill Fabric - the prefill scheduler core.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"
#include "prefillfabric/clock.hpp"
#include "prefillfabric/request.hpp"
#include "prefillfabric/policy.hpp"
#include "prefillfabric/compatibility.hpp"
#include "prefillfabric/executor.hpp"
#include "prefillfabric/memory.hpp"
#include "prefillfabric/observability.hpp"
#include "prefillfabric/explain.hpp"
#include "prefillfabric/persistence.hpp"

namespace prefillfabric {

// The scheduling authority under which an operation is performed. Used to
// reject stale completions (old epoch, old worker boot, obsolete attempt or
// chunk generation).
struct Authority {
  Epoch epoch;
  WorkerId worker;
  WorkerBootId boot;
};

// A group formed and memory-reserved by run_cycle, ready to dispatch.
struct DispatchedGroup {
  GroupId group_id;
  ExecutionGroup group;
  WorkerId assigned_worker;
  ReservationId reservation_id;
  std::uint64_t reserved_bytes = 0;
};

struct DispatchBatch {
  std::vector<DispatchedGroup> groups;
  std::vector<std::string> notes;
  std::uint64_t cycle_scheduled_tokens = 0;
  bool any_dispatch() const noexcept { return !groups.empty(); }
};

// Thrown-free driver result for cancel/retry operations.
struct CancelResult {
  bool applied = false;
  bool was_terminal = false;
  ErrorCode code = ErrorCode::ok;
  std::string message;
};

// The core prefill scheduler. Thread-safe. One instance owns scheduling,
// chunking, packing, fairness, memory governance, epoch/generation and
// persistence state. It never performs network or blocking I/O while holding
// its internal lock, and never invokes callbacks under the lock.
class PrefillScheduler {
 public:
  explicit PrefillScheduler(SchedulerConfig cfg, std::shared_ptr<Clock> clock);
  ~PrefillScheduler();
  PrefillScheduler(const PrefillScheduler&) = delete;
  PrefillScheduler& operator=(const PrefillScheduler&) = delete;

  // ---- Submission / admission ----
  // Validate and admit a request. Token data may be supplied separately so
  // the CPU/CUDA executors can do real work (register via submit_with_tokens).
  Result<void> submit(const PrefillRequest& req);
  Result<void> submit_with_tokens(const PrefillRequest& req, std::vector<std::uint32_t> tokens);

  // ---- Scheduling cycle ----
  // Form compatible groups, reserve memory, assign workers. Returns groups that
  // the caller/coordinator must execute and then report via report_completion.
  Result<DispatchBatch> run_cycle();

  // ---- Completion ----
  // Apply an authoritative completion. Rejects stale epoch/boot/attempt/
  // generation authority and duplicate/terminal completions. Never mutates
  // state for stale authority.
  Result<void> report_completion(const MemberResult& result);

  // ---- Cancellation ----
  CancelResult cancel(RequestId req);
  CancelResult cancel_generation(RequestId req, Generation only_gen);

  // ---- Retry ----
  Result<void> force_retry(RequestId req);

  // ---- Worker lifecycle (distributed) ----
  Result<void> register_worker(const WorkerDescriptor& w);
  Result<void> unregister_worker(WorkerId w);
  Result<void> worker_lost(WorkerId w);
  std::vector<WorkerDescriptor> workers() const;
  std::optional<WorkerDescriptor> worker(WorkerId w) const;
  WorkerId local_worker_id() const;

  // ---- Epoch rollover ----
  Result<void> roll_epoch();
  Epoch current_epoch() const;

  // ---- Executor wiring (in-process) ----
  void attach_executor(std::shared_ptr<Executor> ex, const std::string& backend = kBackendCpu);
  std::shared_ptr<Executor> executor() const;

  // In-process convenience: run cycles, execute on the attached executor, and
  // report completions until no further dispatchable work remains.
  Result<void> drive_until_quiescent(std::size_t max_cycles = 4096);

  // ---- Observability ----
  Stats stats() const;
  Snapshot snapshot() const;
  Result<Explain> explain(RequestId req) const;
  EventLog& events();
  const EventLog& events() const;
  MemoryGovernor& memory();
  std::uint64_t memory_reserved() const;
  std::uint64_t memory_available() const;

  // ---- Persistence ----
  Result<void> persist(const std::string& path) const;
  Result<void> recover(const std::string& path);
  PersistedState export_state() const;
  Result<void> import_state(const PersistedState& s);

  // ---- Introspection (tests / validation) ----
  const SchedulerConfig& config() const noexcept;
  std::size_t request_count() const;
  std::size_t queued_count() const;
  std::size_t running_count() const;
  bool is_terminal(RequestId req) const;
  std::optional<Lifecycle> lifecycle(RequestId req) const;
  Generation current_generation(RequestId req) const;
  // In-flight groups currently dispatched (with their member requests).
  struct InFlightInfo { GroupId group_id; WorkerId worker; std::vector<RequestId> requests; Lifecycle lifecycle; };
  std::vector<InFlightInfo> in_flight() const;
  std::uint64_t next_generation_value() const;
  std::uint64_t next_attempt_value() const;
  void set_ring_note(bool enabled) noexcept;  // silence note accumulation

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace prefillfabric
