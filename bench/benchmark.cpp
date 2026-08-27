// Prefill Fabric - benchmark of real runtime work.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/chunking.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include "prefillfabric/persistence.hpp"
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace prefillfabric;
using ClockNS = std::chrono::steady_clock;

static double secs_since(ClockNS::time_point t0) { return std::chrono::duration<double>(ClockNS::now() - t0).count(); }

static std::vector<std::uint32_t> batok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

static PrefillRequest breq(RequestId id, TenantId t, std::uint64_t prompt) {
  PrefillRequest r; r.request_id = id; r.tenant_id = t; r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt; r.tenant_weight = 1.0; r.priority = static_cast<int>(id.value() % 8);
  r.compute_estimate = prompt * 4; r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
  return r;
}

int main() {
  std::printf("== Prefill Fabric benchmark ==\n");
  const std::uint64_t N = 2000;

  // 1) Submission + scheduling + completion throughput (CPU executor).
  {
    SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30; cfg.max_tokens_per_chunk = 512;
    PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
    sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
    auto t0 = ClockNS::now();
    for (std::uint64_t i = 0; i < N; ++i) sch.submit_with_tokens(breq(RequestId(i + 1), TenantId(1 + (i % 4)), 512), batok(512, i + 1));
    const double submit_s = secs_since(t0);
    t0 = ClockNS::now();
    sch.drive_until_quiescent();
    const double sched_s = secs_since(t0);
    const auto st = sch.stats();
    std::printf("submit_throughput=%.0f req/s (%llu reqs in %.3fs)\n", N / (submit_s > 0 ? submit_s : 1e-9), (unsigned long long)N, submit_s);
    std::printf("sched_complete=%.0f req/s groups=%llu tokens=%llu (%.3fs)\n", N / (sched_s > 0 ? sched_s : 1e-9), (unsigned long long)st.groups_formed, (unsigned long long)st.completed_tokens, sched_s);
    std::printf("cpu_executor=%.0f tok/s\n", (double)st.completed_tokens / (sched_s > 0 ? sched_s : 1e-9));
  }

  // 2) Chunk-plan creation throughput.
  {
    SchedulerConfig cfg; cfg.max_tokens_per_chunk = 1024;
    ChunkPlanner cp(cfg);
    auto req = breq(RequestId(1), TenantId(1), 10000);
    auto t0 = ClockNS::now();
    std::uint64_t chunks = 0;
    for (int i = 0; i < 100000; ++i) { auto p = cp.plan(req, AttemptId(1), Generation(0)); chunks += p.chunks.size(); }
    const double s = secs_since(t0);
    std::printf("chunk_plan=%.0f plans/s (%llu chunks)\n", 100000 / (s > 0 ? s : 1e-9), (unsigned long long)chunks);
  }

  // 3) Packed group formation (compatibility key + grouping).
  {
    SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30; cfg.max_tokens_per_chunk = 1024; cfg.max_requests_per_group = 16;
    PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
    sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
    for (std::uint64_t i = 0; i < 2000; ++i) sch.submit_with_tokens(breq(RequestId(i + 1), TenantId(1 + (i % 2)), 1024), batok(1024, i + 1));
    auto t0 = ClockNS::now();
    auto b = sch.run_cycle();
    const double s = secs_since(t0);
    std::printf("group_formation=%.0f groups/s (%zu groups, %llu tokens)\n", (double)b.value().groups.size() / (s > 0 ? s : 1e-9), b.value().groups.size(), (unsigned long long)b.value().cycle_scheduled_tokens);
  }

  // 4) Snapshot + persistence serialization/recovery throughput.
  {
    SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
    PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
    sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
    for (std::uint64_t i = 0; i < 2000; ++i) sch.submit_with_tokens(breq(RequestId(i + 1), TenantId(1), 512), batok(512, i + 1));
    sch.drive_until_quiescent();
    auto t0 = ClockNS::now();
    for (int i = 0; i < 100; ++i) { auto s = sch.snapshot(); (void)s; }
    const double snap_s = secs_since(t0);
    std::printf("snapshot=%.0f sn/s\n", 100 / (snap_s > 0 ? snap_s : 1e-9));
    t0 = ClockNS::now();
    auto st = sch.export_state();
    auto blob = serialize_state(st);
    const double ser_s = secs_since(t0);
    t0 = ClockNS::now();
    auto dec = deserialize_state(blob.value());
    const double deser_s = secs_since(t0);
    std::printf("persist_serialize=%.0f states/s deserialize=%.0f states/s (bytes=%zu)\n", 1 / (ser_s > 0 ? ser_s : 1e-9), 1 / (deser_s > 0 ? deser_s : 1e-9), blob.value().size());
  }

  // 5) Queue depths: submit 10K requests, measure submission throughput.
  {
    SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
    PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
    const std::uint64_t Q = 10000;
    auto t0 = ClockNS::now();
    for (std::uint64_t i = 0; i < Q; ++i) sch.submit(breq(RequestId(i + 1), TenantId(1), 256));
    const double s = secs_since(t0);
    std::printf("queue_10k_submit=%.0f req/s (depth=%zu)\n", Q / (s > 0 ? s : 1e-9), sch.queued_count());
  }

  std::printf("== benchmark complete ==\n");
  return 0;
}