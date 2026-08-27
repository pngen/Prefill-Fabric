// Prefill Fabric - adversarial input validation.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include "prefillfabric/persistence.hpp"
#include "prefillfabric/protocol.hpp"
#include "prefillfabric/network.hpp"
#include <vector>

using namespace prefillfabric;

class ManualClock : public Clock { public: Nanoseconds now() const noexcept override { return t_; } void adv(Nanoseconds n) noexcept { t_ += n; } private: Nanoseconds t_ = 0; };

static PrefillRequest areq(RequestId id, TenantId t, std::uint64_t prompt) {
  PrefillRequest r; r.request_id = id; r.tenant_id = t; r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt; r.tenant_weight = 1.0; r.compute_estimate = prompt; r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> atok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

TEST(adversarial_zero_token_and_bad_cached) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  auto r0 = areq(RequestId(1), TenantId(1), 0);
  CHECK(!sch.submit(r0).is_ok());
  auto r1 = areq(RequestId(2), TenantId(1), 100);
  r1.reusable_prefix.token_count = 200; r1.reusable_prefix.validated = true;   // cached > prompt. 
  CHECK(!sch.submit(r1).is_ok());   // invalid_reusable_prefix.
}

TEST(adversarial_duplicate_id) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  CHECK(sch.submit_with_tokens(areq(RequestId(5), TenantId(1), 100), atok(100, 1)).is_ok());
  CHECK(!sch.submit_with_tokens(areq(RequestId(5), TenantId(1), 100), atok(100, 2)).is_ok());   // duplicate_request.
}

TEST(adversarial_overflow_boundaries) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  // compute_estimate at overflow boundary must not crash the chunk planner.
  auto r = areq(RequestId(9), TenantId(1), 1000);
  r.compute_estimate = 0xFFFFFFFFFFFFFFFFULL;
  CHECK(sch.submit(r).is_ok());
  auto rr = areq(RequestId(10), TenantId(1), 1000);
  rr.memory_estimate_bytes = 1ULL << 63;
  CHECK(sch.submit(rr).is_ok()); 
}

TEST(adversarial_expired_deadline) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  auto r = areq(RequestId(11), TenantId(1), 2000);
  r.deadline = Nanoseconds(100);   // already expired relative to now (0).
  CHECK(sch.submit(r).is_ok());
  clock->adv(200);   // advance past the deadline.
  auto b = sch.run_cycle();
  CHECK(b.is_ok());
  CHECK_EQ(sch.lifecycle(RequestId(11)), Lifecycle::expired);   // expired work never dispatches.
}

TEST(adversarial_double_completion_and_cancel) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  CHECK(sch.submit_with_tokens(areq(RequestId(20), TenantId(1), 500), atok(500, 20)).is_ok());
  CHECK(sch.drive_until_quiescent().is_ok());
  CHECK(sch.is_terminal(RequestId(20)));
  // Double completion: build a normal result for the (now terminal) request.
  CpuExecutor ex; InMemoryTokenResolver res; CHECK(res.register_request(areq(RequestId(20), TenantId(1), 500), atok(500, 20)).is_ok());
  ExecutionGroup g; g.group_id = GroupId(1); g.backend = kBackendCpu;
  ExecutableMember m; m.request_id = RequestId(20); m.attempt_id = AttemptId(1); m.generation = Generation(1); m.token_start = 0; m.token_count = 500;
  g.members.push_back(m);
  auto er = ex.execute(g, res);
  CHECK(er.is_ok());
  for (auto& mr : er.value().members) { mr.epoch = sch.current_epoch(); mr.worker = WorkerId(0); mr.boot = sch.worker(WorkerId(0)) ? sch.worker(WorkerId(0)).value().boot_id : WorkerBootId(); }
  // Force a stale attempt so it is rejected as stale rather than terminal.
  auto mr0 = er.value().members[0];
  const auto r1 = sch.report_completion(mr0);
  CHECK(!r1.is_ok());
  CHECK(sch.memory().unreleased_count() == 0u);
  // Double cancellation is idempotent.
  auto c1 = sch.cancel(RequestId(20));
  auto c2 = sch.cancel(RequestId(20));
  CHECK(c1.was_terminal || c1.applied);
  CHECK(c2.was_terminal);   // idempotent on terminal.
}

TEST(adversarial_retry_after_terminal) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  CHECK(sch.submit_with_tokens(areq(RequestId(30), TenantId(1), 300), atok(300, 30)).is_ok());
  CHECK(sch.drive_until_quiescent().is_ok());
  CHECK(sch.is_terminal(RequestId(30)));
  CHECK(!sch.force_retry(RequestId(30)).is_ok());   // retry after terminal rejected.
}

TEST(adversarial_unknown_worker_and_model) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  PrefillScheduler sch(cfg, std::make_shared<ManualClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  // A request targeting a backend with no worker (e.g., "nosuchbackend") is deferred, not crash.
  auto r = areq(RequestId(40), TenantId(1), 500);
  r.backend = "nosuchbackend";
  CHECK(sch.submit(r).is_ok());
  auto b = sch.run_cycle();
  CHECK(b.is_ok());
  CHECK_EQ(sch.running_count(), 0u);
  CHECK(sch.lifecycle(RequestId(40)) != Lifecycle::completed);
  CHECK_EQ(sch.memory().unreleased_count(), 0u);
}
