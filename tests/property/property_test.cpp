// Prefill Fabric - randomized/property validation (fixed reproducible seed).
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include <random>
#include <cstdio>

using namespace prefillfabric;

class ManualClock : public Clock { public: Nanoseconds now() const noexcept override { return t_; } void adv(Nanoseconds n) noexcept { t_ += n; } private: Nanoseconds t_ = 0; };

static std::vector<std::uint32_t> ptok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

static void manual_drive(PrefillScheduler& sch, std::shared_ptr<ManualClock> clk,
                         const TokenResolver& resolver, std::size_t max_cycles = 8000) {
  CpuExecutor ex;
  const Epoch ep = sch.current_epoch();
  const WorkerBootId boot = sch.worker(WorkerId(0)) ? sch.worker(WorkerId(0)).value().boot_id : WorkerBootId();
  for (std::size_t c = 0; c < max_cycles; ++c) {
    auto b = sch.run_cycle();
    if (!b) break;
    bool any = false;
    for (const auto& dg : b.value().groups) {
      any = true;
      auto er = ex.execute(dg.group, resolver);
      if (er) for (auto& mr : er.value().members) { mr.epoch = ep; mr.worker = WorkerId(0); mr.boot = boot; sch.report_completion(mr); }
      else for (const auto& m : dg.group.members) { MemberResult mr; mr.request_id = m.request_id; mr.attempt_id = m.attempt_id; mr.generation = m.generation; mr.outcome = MemberOutcome::retryable_failure; mr.epoch = ep; mr.worker = WorkerId(0); mr.boot = boot; sch.report_completion(mr); }
    }
    clk->adv(100000);
    if (!any && sch.queued_count() == 0) break;
  }
}

TEST(property_random_heavy_invariants) {
  const std::uint32_t seed = 20260501u;   // printed for reproducibility.
  std::mt19937 rng(seed);
  std::printf("property seed=%u\n", seed);

  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30; cfg.max_tokens_per_chunk = 1024; cfg.max_pack_wait_ns = 1000000;
  auto clock = std::make_shared<ManualClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  auto resolver = std::make_shared<InMemoryTokenResolver>();

  const int N = 400;
  std::uint64_t next_id = 1;
  std::vector<RequestId> ids;
  for (int i = 0; i < N; ++i) {
    PrefillRequest r;
    r.request_id = RequestId(next_id);
    r.tenant_id = TenantId(1 + rng() % 8);
    r.model.name = "m"; r.model.revision = "1";
    std::uint64_t prompt = 1 + rng() % 5000;
    r.prompt_token_count = prompt;
    std::uint64_t cached = rng() % (prompt + 1);
    r.reusable_prefix.token_count = cached; r.reusable_prefix.validated = cached > 0;
    r.priority = static_cast<int>(rng() % 10);
    r.tenant_weight = 0.5 + static_cast<double>(rng() % 8) * 0.5;
    r.latency_class = static_cast<LatencyClass>(rng() % 5);
    if (rng() % 2) r.deadline = clock->now() + static_cast<Nanoseconds>(rng() % 50000000);
    r.max_retries = static_cast<int>(rng() % 3);
    r.compute_estimate = prompt * 4;
    r.memory_estimate_bytes = prompt * 16;
    r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
    auto toks = ptok(prompt, next_id);
    resolver->register_request(r, toks);
    CHECK(sch.submit(r).is_ok());
    ids.push_back(r.request_id);
    ++next_id;
  }

  // Random cancellation of a subset (waiting or running; authoritative at boundaries).
  for (int i = 0; i < 30; ++i) {
    const auto ri = ids[rng() % ids.size()];
    auto cr = sch.cancel(ri);
    CHECK(cr.applied || cr.was_terminal);
  }

  manual_drive(sch, clock, *resolver);

  // Invariants after closure.
  CHECK_EQ(sch.memory().unreleased_count(), 0u);
  CHECK(sch.memory().underflow_happened() == false);
  auto sn = sch.snapshot();
  CHECK_EQ(sn.queue_depth, 0u);
  CHECK_EQ(sn.running_requests, 0u);
  CHECK_EQ(sn.remaining_prefill_tokens, 0u);
  const auto st = sch.stats();
  const std::uint64_t terminal = st.completed + st.cancelled + st.expired + st.failed_non_retryable;
  CHECK_EQ(terminal, static_cast<std::uint64_t>(N));
  CHECK_EQ(st.duplicate_completion_rejected, 0u);
  // Per-tenant service finite and non-negative.
  for (const auto& kv : st.per_tenant) {
    CHECK(kv.second.weight > 0);
    CHECK(kv.second.scheduled_tokens <= (1ULL << 40));
    CHECK(kv.second.normalized_service >= 0.0);
  }
}
