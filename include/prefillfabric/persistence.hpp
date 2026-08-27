// Prefill Fabric - deterministic persistence and recovery.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include "prefillfabric/types.hpp"
#include "prefillfabric/result.hpp"
#include "prefillfabric/compatibility.hpp"

namespace prefillfabric {

inline constexpr std::uint32_t kPersistenceMagic = 0x50524631u;  // "PRF1"
inline constexpr std::uint32_t kPersistenceVersion = 1u;

// Deterministic little-endian byte writer.
class ByteWriter {
 public:
  void u8(std::uint8_t v) { buf_.push_back(static_cast<std::uint8_t>(v)); }
  void u16(std::uint16_t v) {
    buf_.push_back(static_cast<std::uint8_t>(v & 0xffu));
    buf_.push_back(static_cast<std::uint8_t>((v >> 8) & 0xffu));
  }
  void u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
  }
  void u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
  }
  void i64(std::int64_t v) { u64(static_cast<std::uint64_t>(v)); }
  void f64(double v) {
    std::uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "double size");
    std::memcpy(&bits, &v, sizeof(bits));
    u64(bits);
  }
  void str(const std::string& s) {
    u64(static_cast<std::uint64_t>(s.size()));
    buf_.insert(buf_.end(), s.begin(), s.end());
  }
  void raw(const std::uint8_t* p, std::size_t n) { buf_.insert(buf_.end(), p, p + n); }
  void id(std::uint64_t v) { u64(v); }
  const std::vector<std::uint8_t>& bytes() const { return buf_; }
  std::size_t size() const { return buf_.size(); }

  template <typename Tag> void strong(StrongId<Tag> s) { u64(s.value()); }

 private:
  std::vector<std::uint8_t> buf_;
};

// Bounds-checked byte reader. Every read validates length so an attacker or
// corrupt file cannot drive out-of-bounds access or a runaway allocation.
class ByteReader {
 public:
  explicit ByteReader(const std::uint8_t* data, std::size_t len) : p_(data), len_(len) {}
  explicit ByteReader(const std::vector<std::uint8_t>& v) : p_(v.data()), len_(v.size()) {}

  Result<std::uint8_t> u8() {
    if (off_ + 1 > len_) return Result<std::uint8_t>::err(err_trunc());
    return Result<std::uint8_t>::ok(p_[off_++]);
  }
  Result<std::uint16_t> u16() {
    if (off_ + 2 > len_) return Result<std::uint16_t>::err(err_trunc());
    std::uint16_t v = static_cast<std::uint16_t>(p_[off_]) |
        static_cast<std::uint16_t>(p_[off_ + 1]) << 8;
    off_ += 2;
    return Result<std::uint16_t>::ok(v);
  }
  Result<std::uint32_t> u32() {
    if (off_ + 4 > len_) return Result<std::uint32_t>::err(err_trunc());
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(p_[off_ + i]) << (8 * i);
    off_ += 4;
    return Result<std::uint32_t>::ok(v);
  }
  Result<std::uint64_t> u64() {
    if (off_ + 8 > len_) return Result<std::uint64_t>::err(err_trunc());
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(p_[off_ + i]) << (8 * i);
    off_ += 8;
    return Result<std::uint64_t>::ok(v);
  }
  Result<std::int64_t> i64() {
    auto v = u64();
    if (!v) return Result<std::int64_t>::err(v.error());
    return Result<std::int64_t>::ok(static_cast<std::int64_t>(v.value()));
  }
  Result<double> f64() {
    auto v = u64();
    if (!v) return Result<double>::err(v.error());
    double d;
    const auto bits = v.value();
    std::memcpy(&d, &bits, sizeof(d));
    return Result<double>::ok(d);
  }
  Result<std::string> str() {
    auto len = u64();
    if (!len) return Result<std::string>::err(len.error());
    const std::uint64_t n = len.value();
    if (n > (len_ - off_)) return Result<std::string>::err(err_trunc());
    if (n > (1ULL << 32)) return Result<std::string>::err(ErrorCode::corrupt_state, "string length unreasonable");
    std::string s(reinterpret_cast<const char*>(p_ + off_), static_cast<std::size_t>(n));
    off_ += static_cast<std::size_t>(n);
    return Result<std::string>::ok(std::move(s));
  }
  Result<std::vector<std::uint8_t>> raw_bytes(std::size_t n) {
    if (n > (len_ - off_)) return Result<std::vector<std::uint8_t>>::err(err_trunc());
    std::vector<std::uint8_t> v(p_ + off_, p_ + off_ + n);
    off_ += n;
    return Result<std::vector<std::uint8_t>>::ok(std::move(v));
  }
  std::size_t remaining() const noexcept { return len_ - off_; }
  std::size_t offset() const noexcept { return off_; }

  template <typename Tag> Result<StrongId<Tag>> strong() {
    auto v = u64();
    if (!v) return Result<StrongId<Tag>>::err(v.error());
    return Result<StrongId<Tag>>::ok(StrongId<Tag>(v.value()));
  }

 private:
  static ErrorInfo err_trunc() {
    return ErrorInfo(ErrorCode::truncation, "truncated persistence blob");
  }
  const std::uint8_t* p_;
  std::size_t len_;
  std::size_t off_ = 0;
};

// Semantic records forming a persisted authoritative snapshot.
struct PersistedRequestRecord {
  RequestId request_id;
  AttemptId attempt_id;
  TenantId tenant_id;
  ModelKey model;
  AdapterId adapter_id;
  AdapterRelation relation = AdapterRelation::none;
  std::uint64_t prompt_token_count = 0;
  std::uint64_t cached_tokens = 0;
  std::uint64_t completed_uncached_tokens = 0;
  Generation current_generation;
  Generation parent_generation;
  std::uint64_t next_chunk_token_start = 0;
  WorkKind work_kind = WorkKind::full_prompt;
  Lifecycle lifecycle = Lifecycle::submitted;
  int priority = 0;
  double tenant_weight = 1.0;
  double normalized_service = 0.0;
  std::uint64_t last_chunk_token_count = 0;
  std::uint64_t completed_token_total = 0;
  bool needs_recovery_full_prefill = false;
  VocabSpec vocab;
  NumericMode numeric_mode = NumericMode::fp32;
  LatencyClass latency_class = LatencyClass::standard;
  std::int64_t deadline_ns = 0;
  std::int64_t arrival_ns = 0;
  bool has_deadline = false;
  bool has_queue_delay = false;
  std::int64_t max_queue_delay_ns = 0;
};

struct PersistedWorkerRecord {
  WorkerId worker_id;
  WorkerBootId boot_id;
  std::string host;
  std::uint32_t port = 0;
  std::string backend = kBackendAny;
  WorkerState state = WorkerState::unknown;
  std::uint64_t capacity_units = 1;
};

struct PersistedTenantRecord {
  TenantId tenant_id;
  double weight = 1.0;
  std::uint64_t scheduled_tokens = 0;
  std::uint64_t completed_requests = 0;
  std::int64_t wait_total_ns = 0;
  std::uint64_t outstanding = 0;
  double normalized_service = 0.0;
};

struct PersistedState {
  std::uint32_t magic = kPersistenceMagic;
  std::uint32_t version = kPersistenceVersion;
  Epoch epoch;
  std::uint64_t next_generation_value = 1;
  std::uint64_t next_attempt_value = 1;
  std::vector<PersistedRequestRecord> requests;
  std::vector<PersistedWorkerRecord> workers;
  std::vector<PersistedTenantRecord> tenants;
  bool was_clean_shutdown = false;
};

// Serialize to a self-contained, checksummed blob.
Result<std::vector<std::uint8_t>> serialize_state(const PersistedState& state);

// Deserialize, validating magic, version, checksum and bounds.
Result<PersistedState> deserialize_state(const std::vector<std::uint8_t>& blob);

// Convenience file persistence.
class FilePersistence {
 public:
  Result<void> save(const PersistedState& state, const std::string& path) const;
  Result<PersistedState> load(const std::string& path) const;
};

}  // namespace prefillfabric
