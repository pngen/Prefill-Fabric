// Prefill Fabric - real framed TCP multiprocess test.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "tests/common/Process.hpp"
#include "prefillfabric/client.hpp"
#include "prefillfabric/network.hpp"
#include <chrono>
#include <thread>

#ifndef PFABRIC_EXE
#define PFABRIC_EXE "pfabric"
#endif

using namespace prefillfabric;

static PrefillRequest mreq(RequestId id, TenantId t, std::uint64_t prompt, int prio = 0, int w = 1) {
  PrefillRequest r; r.request_id = id; r.tenant_id = t; r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt; r.priority = prio; r.tenant_weight = w;
  r.compute_estimate = prompt * 4; r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt; r.input_shape.hidden_dim = 512; r.input_shape.num_heads = 8; r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> mtok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

static std::uint16_t mfree() { auto l = Listener(0); auto r = l.start(); if (!r) return 20000; auto p = l.port(); l.close(); return p; }

TEST(real_framed_tcp_multiprocess) {
  CHECK(net_init().is_ok());
  const std::uint16_t port = mfree();
  const std::string coord = "127.0.0.1:" + std::to_string(port);
  pf_test::Proc cp;
  CHECK(cp.spawn(PFABRIC_EXE, "serve --port " + std::to_string(port), true));
  pf_test::Proc w1, w2;
  CHECK(w1.spawn(PFABRIC_EXE, "worker --coord " + coord + " --id 1 --backend cpu", false));
  CHECK(w2.spawn(PFABRIC_EXE, "worker --coord " + coord + " --id 2 --backend cpu", false));
  Client c("127.0.0.1", port);
  CHECK(c.connect().is_ok());
  for (int i = 0; i < 600; ++i) { auto s = c.query_state(); if (s) { std::size_t ready = 0; for (const auto& w : s.value().workers) if (w.state == WorkerState::ready) ++ready; if (ready >= 2) break; } std::this_thread::sleep_for(std::chrono::milliseconds(10)); }

  // Heterogeneous requests: long multi-chunk, short, multiple tenants.
  const RequestId L1(1), S2(2), L3(3), S4(4), M5(5);
  CHECK(c.submit(mreq(L1, TenantId(10), 20000), mtok(20000, 1)).is_ok());
  CHECK(c.submit(mreq(S2, TenantId(10), 120), mtok(120, 2)).is_ok());
  CHECK(c.submit(mreq(L3, TenantId(20), 16000, 2), mtok(16000, 3)).is_ok());
  CHECK(c.submit(mreq(S4, TenantId(20), 80), mtok(80, 4)).is_ok());
  CHECK(c.submit(mreq(M5, TenantId(30), 2000, 0, 3), mtok(2000, 5)).is_ok());

  for (int i = 0; i < 6000; ++i) {
    bool all = true;
    const RequestId ids[] = { L1, S2, L3, S4, M5 };
    for (const auto rid : ids) { auto ex = c.query_explain(rid); if (!ex) { all = false; break; } const auto lc = ex.value().lifecycle;
      if (lc != Lifecycle::completed && lc != Lifecycle::cancelled && lc != Lifecycle::expired && lc != Lifecycle::failed_non_retryable) { all = false; break; } }
    if (all) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  auto st = c.query_stats();
  CHECK(st.is_ok());
  CHECK_EQ(st.value().completed, 5u);
  CHECK_EQ(st.value().running_requests, 0u);
  CHECK_EQ(st.value().queue_depth, 0u);
  // Compatible packing occurred (groups of >1 request).
  CHECK_GE(st.value().groups_formed, 1u);
  // Fairness: both a long and a short request completed (heterogeneous progress).
  auto e1 = c.query_explain(L1); auto e2 = c.query_explain(S2);
  CHECK(e1.is_ok() && e2.is_ok());
  CHECK(e1.value().lifecycle == Lifecycle::completed);
  CHECK(e2.value().lifecycle == Lifecycle::completed);
  // No leaked reservations (running/queued drained); exact token accounting.
  CHECK_EQ(st.value().completed_tokens, 38200u);

  cp.kill(); w1.kill(); w2.kill();
  net_cleanup();
}
