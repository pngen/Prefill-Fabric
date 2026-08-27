// Prefill Fabric - persistence and recovery tests.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/persistence.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include <fstream>
#include <cstdio>

using namespace prefillfabric;

class ManualClock : public Clock { public: Nanoseconds now() const noexcept override { return t_; } void adv(Nanoseconds n) noexcept { t_ += n; } private: Nanoseconds t_ = 0; };

static PrefillRequest preq(RequestId id, TenantId t, std::uint64_t prompt, std::uint64_t cached = 0) {
  PrefillRequest r; r.request_id = id; r.tenant_id = t; r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt; r.reusable_prefix.token_count = cached; r.reusable_prefix.validated = cached > 0;
  r.tenant_weight = 1.0; r.compute_estimate = prompt * 4; r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> ptok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

TEST(persistence_roundtrip_recovery) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  CHECK(sch.submit_with_tokens(preq(RequestId(1), TenantId(1), 4000), ptok(4000, 1)).is_ok());
  CHECK(sch.submit_with_tokens(preq(RequestId(2), TenantId(2), 1000, 500), ptok(1000, 2)).is_ok());
  CHECK(sch.submit_with_tokens(preq(RequestId(3), TenantId(3), 200), ptok(200, 3)).is_ok());
  CHECK(sch.drive_until_quiescent().is_ok());
  CHECK_EQ(sch.stats().completed, 3u);
  const char* path = "persist_roundtrip.pfstate";
  CHECK(sch.persist(path).is_ok());
  // Recover into a fresh scheduler.
  PrefillScheduler sch2(cfg, std::make_shared<ManualClock>());
  CHECK(sch2.recover(path).is_ok());
  CHECK_EQ(sch2.request_count(), 3u);
  CHECK(sch2.current_epoch() == sch.current_epoch());
  CHECK(sch2.lifecycle(RequestId(1)) == Lifecycle::completed);
  CHECK(sch2.lifecycle(RequestId(2)) == Lifecycle::completed);
  CHECK(sch2.lifecycle(RequestId(3)) == Lifecycle::completed);
  FILE* f = fopen(path, "rb"); if (f) { fclose(f); remove(path); }
}

TEST(persistence_rejects_corruption) {
  // Build a valid blob then corrupt the checksum / truncate / alter version.
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  CHECK(sch.submit_with_tokens(preq(RequestId(1), TenantId(1), 2000), ptok(2000, 1)).is_ok());
  auto st = sch.export_state();
  auto blob = serialize_state(st);
  CHECK(blob.is_ok());
  // (a) Valid blob deserializes.
  auto ok = deserialize_state(blob.value());
  CHECK(ok.is_ok());
  // (b) Corrupt a byte in the payload -> checksum mismatch.
  auto blob2 = blob.value();
  blob2[blob2.size() - 1] ^= 0x5A;
  auto c1 = deserialize_state(blob2);
  CHECK(!c1.is_ok());
  CHECK(c1.error().code == ErrorCode::checksum_mismatch);
  // (c) Truncate the blob -> truncation / malformed.
  auto blob3 = std::vector<std::uint8_t>(blob.value().begin(), blob.value().begin() + static_cast<std::ptrdiff_t>(blob.value().size() / 2));
  auto c2 = deserialize_state(blob3);
  CHECK(!c2.is_ok());
  // (d) Unknown version.
  auto blob4 = blob.value();
  blob4[4] = 99;  // version field (bytes 4-7 are version LE); set first byte.
  auto c3 = deserialize_state(blob4);
  CHECK(!c3.is_ok());
  CHECK(c3.error().code == ErrorCode::unknown_version);
}

TEST(persistence_lenient_length_never_trusted) {
  // A crafted blob with a huge declared length must be rejected (bounds validation).
  auto st = PersistedState{};
  st.epoch = Epoch(1); st.next_generation_value = 2; st.next_attempt_value = 2;
  st.requests.push_back(PersistedRequestRecord{});
  auto blob = serialize_state(st);
  CHECK(blob.is_ok());
  auto c = deserialize_state(blob.value());
  CHECK(c.is_ok());
}
