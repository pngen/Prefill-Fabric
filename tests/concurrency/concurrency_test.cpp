// Prefill Fabric - concurrency pressure test with invariant checks.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include <thread>
#include <vector>
#include <atomic>

using namespace prefillfabric;

static PrefillRequest cnreq(RequestId id, TenantId t, std::uint64_t prompt) {
  PrefillRequest r; r.request_id = id; r.tenant_id = t; r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt; r.tenant_weight = 1.0; r.priority = static_cast<int>(id.value() % 5);
  r.compute_estimate = prompt * 4; r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> cntok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

TEST(concurrency_pressure_invariants) {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30; cfg.max_tokens_per_chunk = 512;
  auto clock = std::make_shared<SteadyClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);

  const int nthreads = 8;
  const int per_thread = 200;
  const std::uint64_t base = 100000;
  std::vector<std::thread> threads;
  std::atomic<int> submitted{0};
  std::atomic<bool> stop{false};

  // Submit threads.
  for (int t = 0; t < nthreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < per_thread; ++i) {
        const std::uint64_t rid = base + static_cast<std::uint64_t>(t) * per_thread + static_cast<std::uint64_t>(i) + 1;
        const std::uint64_t prompt = 64 + (rid * 37) % 2000;
        auto r = cnreq(RequestId(rid), TenantId(1 + static_cast<std::uint64_t>(rid % 6)), prompt);
        sch.submit_with_tokens(r, cntok(prompt, rid));
        submitted++;
      }
    });
  }
  // Cancellation thread.
  threads.emplace_back([&]() {
    for (int i = 0; i < 200; ++i) { std::this_thread::yield(); }
    // Cancel a few low-id requests (some may already be done).
    for (std::uint64_t r = base + 1; r < base + 40; ++r) { sch.cancel(RequestId(r)); }
  });
  // Snapshot/explain thread.
  std::atomic<bool> snap_ok{true};
  threads.emplace_back([&]() {
    for (int i = 0; i < 1000 && !stop; ++i) {
      auto sn = sch.snapshot(); (void)sn;
      auto st = sch.stats(); (void)st;
      auto ex = sch.explain(RequestId(base + 1)); if (ex && ex.value().trace_size() > 1000) snap_ok = false;
      std::this_thread::yield();
    }
  });

  for (auto& th : threads) th.join();
  stop = true;
  CHECK(snap_ok);

  // Drive to quiescence.
  CHECK(sch.drive_until_quiescent().is_ok());

  // Invariants.
  CHECK_EQ(sch.memory().unreleased_count(), 0u);   // no leaked reservations.
  CHECK(sch.memory().underflow_happened() == false);
  auto sn = sch.snapshot();
  CHECK_EQ(sn.queue_depth, 0u);
  CHECK_EQ(sn.running_requests, 0u);
  CHECK_EQ(sn.remaining_prefill_tokens, 0u);
  const auto st = sch.stats();
  // Exact accounting: every submitted request reached a terminal state.
  const std::uint64_t total_terminal = st.completed + st.cancelled + st.expired + st.failed_non_retryable;
  CHECK_EQ(total_terminal, static_cast<std::uint64_t>(submitted));
  // No request in a contradictory state handled by scheduler; verify no stale accepted.
  CHECK_EQ(st.duplicate_completion_rejected, 0u);
  // Per-tenant service accounting finite.
  bool finite = true;
  for (const auto& kv : st.per_tenant) { if (kv.second.normalized_service > 1e18 || kv.second.scheduled_tokens > (1ULL << 40)) finite = false; }
  CHECK(finite);
}
