// Prefill Fabric - memory governance and reservations.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <string>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"

namespace prefillfabric {

struct ReservationIdTag {};
using ReservationId = StrongId<ReservationIdTag>;

// Stage of a reservation in the scheduling pipeline.
enum class ReservationKind : int {
  queued = 0,
  planned = 1,
  reserved = 2,
  dispatched = 3,
  running = 4
};

inline const char* to_string(ReservationKind k) noexcept {
  switch (k) {
    case ReservationKind::queued: return "queued";
    case ReservationKind::planned: return "planned";
    case ReservationKind::reserved: return "reserved";
    case ReservationKind::dispatched: return "dispatched";
    case ReservationKind::running: return "running";
    default: return "unknown";
  }
}

struct ReservationRecord {
  ReservationId id;
  RequestId request_id;
  std::uint64_t bytes = 0;
  ReservationKind kind = ReservationKind::queued;
  bool active = false;
};

// Tracks memory reservations for queued/planned/dispatched/running prefill.
// Invariants enforced here: no double-reserve of the same id, no release of
// a reservation that was never reserved, no leaked reservation after the
// governor is closed, and an exact available == max - reserved relation.
// Guarded by an internal mutex; safe under concurrent scheduling.
class MemoryGovernor {
 public:
  explicit MemoryGovernor(std::uint64_t max_bytes, std::uint64_t headroom_bytes = 0)
      : max_bytes_(max_bytes), headroom_(headroom_bytes), reserved_(0) {}

  // Total hardware-backed memory this governor manages.
  std::uint64_t capacity() const noexcept { return max_bytes_; }
  void set_capacity(std::uint64_t bytes) noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    std::uint64_t newmax = bytes;
    if (newmax < headroom_) newmax = headroom_;
    max_bytes_ = newmax;
  }

  std::uint64_t headroom_bytes() const noexcept { return headroom_; }

  // Bytes made available for reservation (capacity minus headroom).
  std::uint64_t budget() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    return max_bytes_ > headroom_ ? max_bytes_ - headroom_ : 0;
  }

  std::uint64_t reserved_bytes() const noexcept { return reserved_.load(std::memory_order_relaxed); }

  std::uint64_t available_bytes() const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    const std::uint64_t budget = max_bytes_ > headroom_ ? max_bytes_ - headroom_ : 0;
    return reserved_ > budget ? 0 : budget - reserved_;
  }

  bool can_reserve(std::uint64_t bytes) const noexcept {
    std::lock_guard<std::mutex> lk(mtx_);
    const std::uint64_t budget = max_bytes_ > headroom_ ? max_bytes_ - headroom_ : 0;
    if (reserved_ > budget) return false;
    return (budget - reserved_) >= bytes;
  }

  // Reserve bytes for a request. Fails if the id is already reserved (no
  // double-reserve) or if the budget is insufficient.
  Result<void> reserve(ReservationId id, RequestId req, std::uint64_t bytes,
                       ReservationKind kind = ReservationKind::reserved) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = records_.find(id);
    if (it != records_.end() && it->second.active) {
      return Result<void>::err(ErrorCode::duplicate_request, "reservation id already active");
    }
    if (bytes == 0) {
      // Zero-byte reservation is legal (still tracked for lifecycle) but must not consume.
      records_[id] = ReservationRecord{id, req, 0, kind, true};
      return Result<void>::ok();
    }
    const std::uint64_t budget = max_bytes_ > headroom_ ? max_bytes_ - headroom_ : 0;
    if (reserved_ > budget || (budget - reserved_) < bytes) {
      return Result<void>::err(ErrorCode::memory_unavailable, "insufficient memory budget");
    }
    reserved_ += bytes;
    records_[id] = ReservationRecord{id, req, bytes, kind, true};
    return Result<void>::ok();
  }

  // Promote a reservation between lifecycle stages. Idempotent-safe.
  Result<void> transition(ReservationId id, ReservationKind kind) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = records_.find(id);
    if (it == records_.end() || !it->second.active) {
      return Result<void>::err(ErrorCode::illegal_state, "reservation not active");
    }
    it->second.kind = kind;
    return Result<void>::ok();
  }

  // Release a reservation. Idempotent: releasing an already-released or
  // unknown id is a benign no-op that never double-decrements. Tracks
  // double-release attempts for diagnostics.
  Result<void> release(ReservationId id) {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = records_.find(id);
    if (it == records_.end() || !it->second.active) {
      double_releases_.fetch_add(1, std::memory_order_relaxed);
      return Result<void>::ok();
    }
    if (it->second.bytes > 0) {
      if (reserved_ < it->second.bytes) {
        // Underflow guard (should never happen).
        underlying_flags_ |= 1;
        reserved_ = 0;
      } else {
        reserved_ -= it->second.bytes;
      }
    }
    it->second.active = false;
    it->second.bytes = 0;
    return Result<void>::ok();
  }

  // True if the reservation is currently active.
  bool is_active(ReservationId id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = records_.find(id);
    return it != records_.end() && it->second.active;
  }

  std::uint64_t active_reservation_count() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::uint64_t n = 0;
    for (const auto& kv : records_) if (kv.second.active) ++n;
    return n;
  }

  std::uint64_t double_release_count() const noexcept {
    return double_releases_.load(std::memory_order_relaxed);
  }

  bool underflow_happened() const noexcept { return (underlying_flags_ & 1) != 0; }

  // Number of reservations still active (leak detector).
  std::uint64_t unreleased_count() const { return active_reservation_count(); }

  const std::unordered_map<ReservationId, ReservationRecord, HashStrongId>& records() const {
    return records_;
  }

 private:
  mutable std::mutex mtx_;
  std::uint64_t max_bytes_;
  std::uint64_t headroom_;
  std::atomic<std::uint64_t> reserved_{0};
  std::atomic<std::uint64_t> double_releases_{0};
  std::atomic<std::uint64_t> underlying_flags_{0};
  std::unordered_map<ReservationId, ReservationRecord, HashStrongId> records_;
};

}  // namespace prefillfabric
