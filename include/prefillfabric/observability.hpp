// Prefill Fabric - observability: snapshot, stats, events.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <atomic>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"
#include "prefillfabric/clock.hpp"

namespace prefillfabric {

// Per-tenant service accounting (used both in Stats and for fairness input).
struct TenantService {
  double weight = 1.0;
  std::uint64_t scheduled_tokens = 0;
  std::uint64_t completed_requests = 0;
  Nanoseconds wait_total_ns = 0;
  std::uint64_t outstanding = 0;
  double normalized_service = 0.0;
  std::uint64_t admitted_tokens = 0;
};

// Per-device work accounting.
struct DeviceWork {
  std::uint64_t groups_run = 0;
  std::uint64_t tokens_processed = 0;
  Nanoseconds busy_total_ns = 0;
  std::uint64_t peak_reserved_bytes = 0;
};

// Cumulative scheduler statistics. Copyable snapshot for external consumers.
struct Stats {
  std::uint64_t submitted = 0;
  std::uint64_t admitted = 0;
  std::uint64_t rejected = 0;
  std::uint64_t cancelled = 0;
  std::uint64_t retried = 0;
  std::uint64_t completed = 0;
  std::uint64_t failed_non_retryable = 0;
  std::uint64_t expired = 0;
  std::uint64_t deadline_missed = 0;
  std::uint64_t groups_formed = 0;
  std::uint64_t mem_underflow_flag = 0;

  std::uint64_t scheduled_tokens = 0;
  std::uint64_t completed_tokens = 0;
  std::uint64_t chunk_count = 0;
  std::uint64_t stale_authority_rejected = 0;
  std::uint64_t duplicate_completion_rejected = 0;
  std::uint64_t queue_delay_exceeded = 0;

  // Point-in-time values.
  std::uint64_t queue_depth = 0;
  std::uint64_t waiting_tokens = 0;
  std::uint64_t running_requests = 0;
  std::uint64_t remaining_prefill_tokens = 0;
  std::uint64_t reserved_memory_bytes = 0;
  std::uint64_t available_memory_bytes = 0;

  Nanoseconds queue_latency_total_ns = 0;
  Nanoseconds execution_latency_total_ns = 0;
  std::uint64_t queue_latency_samples = 0;
  std::uint64_t execution_latency_samples = 0;

  std::map<ErrorCode, std::uint64_t> rejection_by_reason;
  std::map<std::uint32_t, std::uint64_t> group_size_distribution;
  std::map<TenantId, TenantService, std::less<TenantId>> per_tenant;
  std::map<DeviceId, DeviceWork, std::less<DeviceId>> per_device;

  // Reset all counters.
  void clear() { *this = Stats{}; }
};

// A structured runtime event.
struct Event {
  Nanoseconds time_ns = 0;
  std::string type;
  RequestId request_id;
  AttemptId attempt_id;
  std::string detail;
};

// Bounded-retention event stream. Retains the latest `capacity` events.
class EventLog {
 public:
  explicit EventLog(std::size_t capacity = 10000) : capacity_(capacity) {}

  void record(Nanoseconds t, const std::string& type, RequestId r, AttemptId a,
              std::string detail = "") {
    std::lock_guard<std::mutex> lk(mtx_);
    events_.push_back(Event{t, type, r, a, std::move(detail)});
    if (events_.size() > capacity_) events_.pop_front();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return events_.size();
  }

  std::vector<Event> snapshot() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return {events_.begin(), events_.end()};
  }

 private:
  mutable std::mutex mtx_;
  std::deque<Event> events_;
  std::size_t capacity_;
};

// Simple struct to carry point-in-time scheduler state for observability.
struct Snapshot {
  std::uint64_t queue_depth = 0;
  std::uint64_t waiting_tokens = 0;
  std::uint64_t running_requests = 0;
  std::uint64_t reserved_memory_bytes = 0;
  std::uint64_t remaining_prefill_tokens = 0;
  std::uint64_t completed_requests = 0;
  std::vector<RequestId> waiting_request_ids;
  std::vector<RequestId> running_request_ids;
  Stats stats;
};

}  // namespace prefillfabric
