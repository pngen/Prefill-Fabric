// Prefill Fabric - scheduler policy configuration.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include "prefillfabric/types.hpp"
#include "prefillfabric/clock.hpp"

namespace prefillfabric {

// Explicit, inspectable scheduler configuration. The scheduler resolves each
// policy component separately; it is never a single hidden score.
struct SchedulerConfig {
  // ---- Chunking ----
  std::uint64_t max_tokens_per_chunk = 4096;
  std::uint64_t max_work_per_chunk = 1ULL << 40;
  std::uint64_t max_memory_per_chunk = 512ULL << 20;
  std::uint64_t max_prefill_budget_per_cycle = 1ULL << 40;
  bool preemption_enabled = true;

  // ---- Group formation ----
  std::uint32_t max_requests_per_group = 16;
  std::uint64_t max_tokens_per_group = 8192;
  std::uint64_t max_compute_per_group = 1ULL << 42;
  std::uint64_t max_memory_per_group = 1ULL << 30;
  Nanoseconds max_pack_wait_ns = 4000000;
  std::uint32_t min_preferred_occupancy = 1;
  bool gather_for_occupancy = true;

  // ---- Memory governance ----
  std::uint64_t device_memory_bytes = 8ULL << 30;
  std::uint64_t memory_headroom_bytes = 256ULL << 20;
  double memory_headroom_fraction = 0.05;
  bool enforce_memory_headroom = true;

  // ---- Admission / backpressure ----
  std::uint64_t max_admitted_requests = 100000;
  std::uint64_t per_tenant_outstanding_limit = 10000;
  std::uint64_t per_tenant_admitted_token_limit = 1ULL << 34;
  std::uint64_t max_queued_requests = 100000;
  bool reject_on_memory_pressure = true;

  // ---- Fairness ----
  double aging_rate = 0.001;
  bool service_by_tokens = true;
  double fairness_quantum = 1.0;

  // ---- Deadline / latency ----
  Nanoseconds deadline_slack_ns = 1000000;
  bool drop_expired = true;

  // ---- Retry ----
  int default_max_retries = 2;

  // ---- Determinism / accounting ----
  bool deterministic_tie_break = true;
  bool track_service_accounting = true;

  // Effective usable memory budget after headroom.
  std::uint64_t usable_memory_bytes() const noexcept {
    std::uint64_t headroom = memory_headroom_bytes;
    const std::uint64_t frac = static_cast<std::uint64_t>(
        static_cast<double>(device_memory_bytes) * memory_headroom_fraction);
    if (frac > headroom) headroom = frac;
    return device_memory_bytes > headroom ? device_memory_bytes - headroom : 0;
  }
};

}  // namespace prefillfabric
