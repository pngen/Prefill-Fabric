// Prefill Fabric - CUDA validation on a real NVIDIA device.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/exec/cuda_executor.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/frontend_math.hpp"

using namespace prefillfabric;

static std::vector<std::uint32_t> ctok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

static PrefillRequest creq(RequestId id, TenantId t, std::uint64_t prompt, int prio = 0) {
  PrefillRequest r; r.request_id = id; r.tenant_id = t; r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt; r.priority = prio; r.tenant_weight = 1.0; r.backend = kBackendCuda;
  r.compute_estimate = prompt * 4; r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
  return r;
}

TEST(cuda_executor_single_request_matches_cpu) {
  CHECK(CudaExecutor::cuda_available());  // must be present -- this is NOT a skip.
  const auto dev = CudaExecutor::query_device(0);
  CHECK(dev.is_ok());
  CHECK_EQ(dev.value().capability, "sm_120");
  CudaExecutor ce(0);
  auto req = creq(RequestId(1), TenantId(1), 512);
  auto toks = ctok(512, 7);
  InMemoryTokenResolver r;
  CHECK(r.register_request(req, toks).is_ok());
  ExecutionGroup g; g.group_id = GroupId(1); g.backend = kBackendCuda; g.device = DeviceId(0);
  ExecutableMember m; m.request_id = RequestId(1); m.attempt_id = AttemptId(1); m.generation = Generation(3);
  m.token_start = 0; m.token_count = 512; g.members.push_back(m); g.total_token_count = 512;
  auto res = ce.execute(g, r);
  CHECK(res.is_ok());
  CHECK_EQ(res.value().members.size(), 1u);
  CHECK(res.value().members[0].outcome == MemberOutcome::success);
  const auto expected = ff_member_digest(RequestId(1), AttemptId(1), Generation(3), toks.data(), 0, 512);
  CHECK_EQ(res.value().members[0].digest, expected);
}

TEST(cuda_executor_packed_group) {
  CHECK(CudaExecutor::cuda_available());
  CudaExecutor ce(0);
  InMemoryTokenResolver r;
  ExecutionGroup g; g.group_id = GroupId(2); g.backend = kBackendCuda; g.device = DeviceId(0);
  const int n = 4;
  const std::uint64_t chunk = 128;
  for (int i = 0; i < n; ++i) {
    auto req = creq(RequestId(100 + i), TenantId(1), chunk);
    auto toks = ctok(chunk, 10 + i);
    CHECK(r.register_request(req, toks).is_ok());
    ExecutableMember m; m.request_id = req.request_id; m.attempt_id = AttemptId(1); m.generation = Generation(1);
    m.token_start = 0; m.token_count = chunk; g.members.push_back(m); g.total_token_count += chunk;
  }
  auto res = ce.execute(g, r);
  CHECK(res.is_ok());
  CHECK_EQ(res.value().members.size(), static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    auto req = creq(RequestId(100 + i), TenantId(1), chunk);
    auto toks = ctok(chunk, 10 + i);
    const auto exp = ff_member_digest(req.request_id, AttemptId(1), Generation(1), toks.data(), 0, chunk);
    CHECK_EQ(res.value().members[static_cast<std::size_t>(i)].digest, exp);
  }
}

TEST(cuda_scheduler_long_prompt_chunks_memory) {
  CHECK(CudaExecutor::cuda_available());
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 12ULL << 30;
  cfg.max_tokens_per_chunk = 512;
  auto clock = std::make_shared<SteadyClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CudaExecutor>(), kBackendCuda);
  auto req = creq(RequestId(5), TenantId(2), 5000);
  CHECK(sch.submit_with_tokens(req, ctok(5000, 99)).is_ok());
  auto d = sch.drive_until_quiescent();
  CHECK(d.is_ok());
  CHECK(sch.lifecycle(RequestId(5)) == Lifecycle::completed);
  CHECK(sch.memory().unreleased_count() == 0u);
  auto st = sch.stats();
  CHECK_EQ(st.completed_tokens, 5000u);
}
