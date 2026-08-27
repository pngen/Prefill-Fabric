// Prefill Fabric - core unit tests.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/chunking.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include "prefillfabric/frontend_math.hpp"

using namespace prefillfabric;

class ManualClock : public Clock {
 public:
  Nanoseconds now() const noexcept override { return t_; }
  void advance(Nanoseconds n) noexcept { t_ += n; }
 private:
  Nanoseconds t_ = 0;
};

static PrefillRequest make_req(RequestId id, TenantId tenant, std::uint64_t prompt,
                              std::uint64_t cached = 0, int priority = 0,
                              LatencyClass lc = LatencyClass::standard) {
  PrefillRequest r;
  r.request_id = id;
  r.tenant_id = tenant;
  r.model.name = "m";
  r.model.revision = "1";
  r.prompt_token_count = prompt;
  r.reusable_prefix.token_count = cached;
  r.reusable_prefix.validated = (cached > 0);
  r.priority = priority;
  r.latency_class = lc;
  r.tenant_weight = 1.0;
  r.compute_estimate = prompt * 4;
  r.memory_estimate_bytes = prompt * 64;
  r.input_shape.seq_len = prompt;
  r.input_shape.hidden_dim = 512;
  r.input_shape.num_heads = 8;
  r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> tokens_for(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n);
  std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu);
  }
  return v;
}

TEST(chunk_planner_deterministic) {
  SchedulerConfig cfg;
  cfg.max_tokens_per_chunk = 1000;
  ChunkPlanner cp(cfg);
  auto req = make_req(RequestId(1), TenantId(1), 5000);
  auto p1 = cp.plan(req, AttemptId(1), Generation(0));
  auto p2 = cp.plan(req, AttemptId(1), Generation(0));
  CHECK_EQ(p1.chunks.size(), p2.chunks.size());
  CHECK_EQ(p1.total_chunk_tokens, 5000u);
  CHECK_EQ(p1.chunks.size(), 5u);
  CHECK_EQ(p1.chunks[0].token_count, 1000u);
  CHECK_EQ(p1.chunks[4].token_count, 1000u);
  CHECK_EQ(p1.chunks[0].token_start, 0u);
  CHECK_EQ(p1.chunks[1].token_start, 1000u);
  for (std::size_t i = 0; i < p1.chunks.size(); ++i) {
    CHECK_EQ(p1.chunks[i].token_start, p2.chunks[i].token_start);
    CHECK_EQ(p1.chunks[i].token_count, p2.chunks[i].token_count);
  }
}

TEST(chunk_planner_reused_prefix_remainder) {
  SchedulerConfig cfg;
  cfg.max_tokens_per_chunk = 2000;
  ChunkPlanner cp(cfg);
  auto req = make_req(RequestId(1), TenantId(1), 32768, 28672);
  CHECK_EQ(req.remaining_prefill_tokens(), 4096u);
  auto p = cp.plan(req, AttemptId(1), Generation(0));
  CHECK_EQ(p.total_chunk_tokens, 4096u);
  CHECK_EQ(p.chunks.size(), 3u);
  CHECK_EQ(p.chunks[0].work_kind, WorkKind::reused_suffix);
  CHECK_EQ(p.chunks[0].token_start, 0u);
  CHECK_EQ(p.chunks[0].token_count, 2000u);
}

TEST(cpu_executor_verifiable_digest) {
  auto req = make_req(RequestId(7), TenantId(2), 64);
  auto toks = tokens_for(64, 42);
  InMemoryTokenResolver r;
  auto reg = r.register_request(req, toks);
  CHECK(reg.is_ok());

  CpuExecutor ex;
  ExecutionGroup g;
  g.group_id = GroupId(1);
  g.backend = kBackendCpu;
  g.device = DeviceId(1);
  ExecutableMember m;
  m.request_id = RequestId(7);
  m.attempt_id = AttemptId(1);
  m.generation = Generation(5);
  m.token_start = 0;
  m.token_count = 64;
  g.members.push_back(m);
  g.total_token_count = 64;

  auto res = ex.execute(g, r);
  CHECK(res.is_ok());
  auto& er = res.value();
  CHECK_EQ(er.members.size(), 1u);
  CHECK(er.members[0].outcome == MemberOutcome::success);
  CHECK_EQ(er.members[0].tokens_completed, 64u);
  const auto expected = ff_member_digest(RequestId(7), AttemptId(1), Generation(5),
                                          toks.data(), 0, 64);
  CHECK_EQ(er.members[0].digest, expected);
}

TEST(scheduler_basic_completion) {
  SchedulerConfig cfg;
  cfg.max_tokens_per_chunk = 1000;
  cfg.device_memory_bytes = 1ULL << 30;
  auto clock = std::make_shared<SteadyClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);

  auto r1 = make_req(RequestId(1), TenantId(1), 2500);
  auto s1 = sch.submit_with_tokens(r1, tokens_for(2500, 1));
  CHECK(s1.is_ok());
  auto r2 = make_req(RequestId(2), TenantId(1), 100);
  auto s2 = sch.submit_with_tokens(r2, tokens_for(100, 2));
  CHECK(s2.is_ok());

  auto drive = sch.drive_until_quiescent();
  CHECK(drive.is_ok());
  CHECK(sch.is_terminal(RequestId(1)));
  CHECK(sch.is_terminal(RequestId(2)));
  CHECK(sch.memory().unreleased_count() == 0u);
  const auto st = sch.stats();
  CHECK_EQ(st.completed, 2u);
  CHECK_EQ(st.completed_tokens, 2600u);
}

TEST(scheduler_cached_prefix_no_work) {
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 1ULL << 30;
  auto clock = std::make_shared<SteadyClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  // Fully cached prompt -> immediate completion, zero execution.
  auto r = make_req(RequestId(9), TenantId(1), 5000, 5000);
  auto s = sch.submit_with_tokens(r, tokens_for(5000, 9));
  CHECK(s.is_ok());
  CHECK(sch.lifecycle(RequestId(9)) == Lifecycle::completed);
  CHECK(sch.memory().unreleased_count() == 0u);
}
