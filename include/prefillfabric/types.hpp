// Prefill Fabric - strong typed identities and value types.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <string>
#include <functional>
#include <cstddef>
#include <utility>

namespace prefillfabric {

// Strongly typed 64-bit identity. Each distinct tag creates a distinct type
// so that RequestId, AttemptId, WorkerId ... cannot be silently interchanged.
template <typename Tag>
class StrongId {
 public:
  using Value = std::uint64_t;

  constexpr StrongId() noexcept : value_(0) {}
  explicit constexpr StrongId(Value v) noexcept : value_(v) {}
  constexpr Value value() const noexcept { return value_; }
  constexpr bool is_nil() const noexcept { return value_ == 0; }
  explicit constexpr operator bool() const noexcept { return value_ != 0; }

  friend constexpr bool operator==(StrongId a, StrongId b) noexcept { return a.value_ == b.value_; }
  friend constexpr bool operator!=(StrongId a, StrongId b) noexcept { return a.value_ != b.value_; }
  friend constexpr bool operator<(StrongId a, StrongId b) noexcept { return a.value_ < b.value_; }
  friend constexpr bool operator<=(StrongId a, StrongId b) noexcept { return a.value_ <= b.value_; }
  friend constexpr bool operator>(StrongId a, StrongId b) noexcept { return a.value_ > b.value_; }
  friend constexpr bool operator>=(StrongId a, StrongId b) noexcept { return a.value_ >= b.value_; }

 private:
  Value value_;
};

struct RequestIdTag {};
struct AttemptIdTag {};
struct TenantIdTag {};
struct WorkerIdTag {};
struct WorkerBootIdTag {};
struct EpochTag {};
struct AdapterIdTag {};
struct DeviceIdTag {};
struct GroupIdTag {};
struct GenerationTag {};

using RequestId = StrongId<RequestIdTag>;
using AttemptId = StrongId<AttemptIdTag>;
using TenantId = StrongId<TenantIdTag>;
using WorkerId = StrongId<WorkerIdTag>;
using WorkerBootId = StrongId<WorkerBootIdTag>;
using Epoch = StrongId<EpochTag>;
using AdapterId = StrongId<AdapterIdTag>;
using DeviceId = StrongId<DeviceIdTag>;
using GroupId = StrongId<GroupIdTag>;
using Generation = StrongId<GenerationTag>;

struct HashStrongId {
  template <typename Tag>
  std::size_t operator()(StrongId<Tag> id) const noexcept {
    return std::hash<std::uint64_t>{}(id.value());
  }
};

// Model identity: a name plus a revision. Revision is a string to allow a
// variety of versioning schemes (hashes, tags, semver).
struct ModelKey {
  std::string name;
  std::string revision;

  bool operator==(const ModelKey& o) const noexcept { return name == o.name && revision == o.revision; }
  bool operator!=(const ModelKey& o) const noexcept { return !(*this == o); }
  bool operator<(const ModelKey& o) const noexcept {
    if (name != o.name) return name < o.name;
    return revision < o.revision;
  }
};

struct HashModelKey {
  std::size_t operator()(const ModelKey& k) const noexcept {
    std::size_t h = std::hash<std::string>{}(k.name);
    h ^= std::hash<std::string>{}(k.revision) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

// Numeric precision/mode that participates in execution compatibility.
enum class NumericMode : int {
  fp32 = 0,
  fp16 = 1,
  bf16 = 2,
  fp8 = 3,
  int8 = 4,
  mixed = 5
};

inline const char* to_string(NumericMode m) noexcept {
  switch (m) {
    case NumericMode::fp32: return "fp32";
    case NumericMode::fp16: return "fp16";
    case NumericMode::bf16: return "bf16";
    case NumericMode::fp8: return "fp8";
    case NumericMode::int8: return "int8";
    case NumericMode::mixed: return "mixed";
    default: return "unknown";
  }
}

// Tokenizer / vocabulary compatibility constraint that may gate packing.
struct VocabSpec {
  std::string tokenizer;
  std::uint64_t vocab_size = 0;
  bool operator==(const VocabSpec& o) const noexcept {
    return tokenizer == o.tokenizer && vocab_size == o.vocab_size;
  }
  bool operator!=(const VocabSpec& o) const noexcept { return !(*this == o); }
};

struct HashVocabSpec {
  std::size_t operator()(const VocabSpec& v) const noexcept {
    std::size_t h = std::hash<std::string>{}(v.tokenizer);
    h ^= std::hash<std::uint64_t>{}(v.vocab_size) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

// Latency class requested by a request. These form a partial order for
// scheduling urgency (faster classes are more urgent).
enum class LatencyClass : int {
  best_effort = 0,
  standard = 1,
  latency_sensitive = 2,
  interactive = 3,
  real_time = 4
};

inline const char* to_string(LatencyClass c) noexcept {
  switch (c) {
    case LatencyClass::best_effort: return "best_effort";
    case LatencyClass::standard: return "standard";
    case LatencyClass::latency_sensitive: return "latency_sensitive";
    case LatencyClass::interactive: return "interactive";
    case LatencyClass::real_time: return "real_time";
    default: return "unknown";
  }
}

// Execution backend identifier. This is an open enum-like string that the
// executor layer resolves; it is NOT a type-system assumption in the runtime.
inline constexpr const char* kBackendCpu = "cpu";
inline constexpr const char* kBackendCuda = "cuda";
inline constexpr const char* kBackendAny = "*";

// Input layout family that participates in compatibility.
enum class InputLayout : int {
  seq_major = 0,
  batch_major = 1,
  packed = 2,
  block = 3
};

inline const char* to_string(InputLayout l) noexcept {
  switch (l) {
    case InputLayout::seq_major: return "seq_major";
    case InputLayout::batch_major: return "batch_major";
    case InputLayout::packed: return "packed";
    case InputLayout::block: return "block";
    default: return "unknown";
  }
}

// Prefill work kinds (see the prefill work model).
enum class WorkKind : int {
  full_prompt = 0,
  partial = 1,
  chunked = 2,
  continuation = 3,
  reused_suffix = 4,       // reused-prefix + uncached suffix
  retried = 5,
  cancelled = 6,
  deadline_expired = 7,
  split = 8,
  merged = 9,              // formed/packed
  worker_recovered = 10,
  stale_rejected = 11
};

inline const char* to_string(WorkKind k) noexcept {
  switch (k) {
    case WorkKind::full_prompt: return "full_prompt";
    case WorkKind::partial: return "partial";
    case WorkKind::chunked: return "chunked";
    case WorkKind::continuation: return "continuation";
    case WorkKind::reused_suffix: return "reused_suffix";
    case WorkKind::retried: return "retried";
    case WorkKind::cancelled: return "cancelled";
    case WorkKind::deadline_expired: return "deadline_expired";
    case WorkKind::split: return "split";
    case WorkKind::merged: return "merged";
    case WorkKind::worker_recovered: return "worker_recovered";
    case WorkKind::stale_rejected: return "stale_rejected";
    default: return "unknown";
  }
}

// Adapter / base relationship that gates packing compatibility.
enum class AdapterRelation : int {
  none = 0,
  same_base = 1,
  base_adapter = 2,
  distinct = 3
};

inline const char* to_string(AdapterRelation r) noexcept {
  switch (r) {
    case AdapterRelation::none: return "none";
    case AdapterRelation::same_base: return "same_base";
    case AdapterRelation::base_adapter: return "base_adapter";
    case AdapterRelation::distinct: return "distinct";
    default: return "unknown";
  }
}

// Request lifecycle states.
enum class Lifecycle : int {
  submitted = 0,
  admitted = 1,
  queued = 2,
  formed = 3,
  reserved = 4,
  dispatched = 5,
  running = 6,
  awaiting_retry = 7,
  completed = 8,
  failed_non_retryable = 9,
  cancelled = 10,
  expired = 11,
  rejected = 12
};

inline const char* to_string(Lifecycle s) noexcept {
  switch (s) {
    case Lifecycle::submitted: return "submitted";
    case Lifecycle::admitted: return "admitted";
    case Lifecycle::queued: return "queued";
    case Lifecycle::formed: return "formed";
    case Lifecycle::reserved: return "reserved";
    case Lifecycle::dispatched: return "dispatched";
    case Lifecycle::running: return "running";
    case Lifecycle::awaiting_retry: return "awaiting_retry";
    case Lifecycle::completed: return "completed";
    case Lifecycle::failed_non_retryable: return "failed_non_retryable";
    case Lifecycle::cancelled: return "cancelled";
    case Lifecycle::expired: return "expired";
    case Lifecycle::rejected: return "rejected";
    default: return "unknown";
  }
}

// Per-member outcome of an execution group.
enum class MemberOutcome : int {
  success = 0,
  needs_next_chunk = 1,
  retryable_failure = 2,
  non_retryable_failure = 3,
  cancelled = 4,
  expired = 5,
  stale_rejected = 6
};

inline const char* to_string(MemberOutcome m) noexcept {
  switch (m) {
    case MemberOutcome::success: return "success";
    case MemberOutcome::needs_next_chunk: return "needs_next_chunk";
    case MemberOutcome::retryable_failure: return "retryable_failure";
    case MemberOutcome::non_retryable_failure: return "non_retryable_failure";
    case MemberOutcome::cancelled: return "cancelled";
    case MemberOutcome::expired: return "expired";
    case MemberOutcome::stale_rejected: return "stale_rejected";
    default: return "unknown";
  }
}

// Worker health states.
enum class WorkerState : int {
  unknown = 0,
  registered = 1,
  ready = 2,
  busy = 3,
  lost = 4,
  retired = 5
};

inline const char* to_string(WorkerState s) noexcept {
  switch (s) {
    case WorkerState::unknown: return "unknown";
    case WorkerState::registered: return "registered";
    case WorkerState::ready: return "ready";
    case WorkerState::busy: return "busy";
    case WorkerState::lost: return "lost";
    case WorkerState::retired: return "retired";
    default: return "unknown";
  }
}

}  // namespace prefillfabric
