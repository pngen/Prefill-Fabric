// Prefill Fabric - atomic multiprocess restart/epoch/stale-authority closure.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "tests/common/TestHarness.hpp"
#include "tests/common/Process.hpp"
#include "prefillfabric/client.hpp"
#include "prefillfabric/network.hpp"
#include "prefillfabric/protocol.hpp"
#include "prefillfabric/scheduler.hpp"
#include <windows.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>

using namespace prefillfabric;

static PrefillRequest req_of(RequestId id, TenantId t, std::uint64_t prompt, int prio = 0, int tweight = 1) {
  PrefillRequest r;
  r.request_id = id;
  r.tenant_id = t;
  r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = prompt;
  r.priority = prio;
  r.tenant_weight = tweight;
  r.compute_estimate = prompt * 4;
  r.memory_estimate_bytes = prompt * 16;
  r.input_shape.seq_len = prompt;
  r.input_shape.hidden_dim = 512;
  r.input_shape.num_heads = 8;
  r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> toks(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n);
  std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

static std::uint16_t find_free_port() {
  auto l = Listener(0);
  auto r = l.start();
  if (!r) return 10000;
  const auto p = l.port();
  l.close();
  return p;
}

// Read a line from a captured stdout (poll).
static std::string read_line(std::string& buf, pf_test::Proc& p) {
  for (int i = 0; i < 300; ++i) {
    buf += p.read_stdout(4096);
    const auto nl = buf.find('\n');
    if (nl != std::string::npos) { const std::string line = buf.substr(0, nl); buf.erase(0, nl + 1); return line; }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return "";
}

// Wait until coordinator reports 2 ready workers.
static bool wait_workers2(Client& c, int attempts = 600) {
  for (int i = 0; i < attempts; ++i) {
    auto st = c.query_state();
    if (st) {
      std::size_t ready = 0;
      for (const auto& w : st.value().workers) if (w.state == WorkerState::ready) ++ready;
      if (ready >= 2) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

// Send a completion frame over a fresh client connection and return the reply.
static ErrorCode send_completion(const std::string& host, std::uint16_t port, const Message& comp) {
  auto s = Socket::connect(host, port);
  if (!s) return ErrorCode::io_failure;
  Socket& sk = s.value();
  Message h; h.kind = MsgKind::hello;
  auto hb = encode_frame(h);
  if (!hb) return ErrorCode::io_failure;
  sk.send_frame(hb.value().data(), hb.value().size());
  auto cb = encode_frame(comp);
  if (!cb) return ErrorCode::io_failure;
  sk.send_frame(cb.value().data(), cb.value().size());
  auto fr = sk.recv_frame();
  if (!fr) return ErrorCode::io_failure;
  auto dm = decode_frame(fr.value());
  if (!dm) return ErrorCode::malformed_frame;
  if (dm.value().kind == MsgKind::error) return dm.value().error.code;
  return ErrorCode::ok;
}
#ifndef PFABRIC_EXE
#define PFABRIC_EXE "pfabric"
#endif

TEST(atomic_multiprocess_restart_epoch_stale_closure) {
  CHECK(net_init().is_ok());
  const std::uint16_t port = find_free_port();
  const std::string host = "127.0.0.1";
  const std::string coord = host + ":" + std::to_string(port);

  pf_test::Proc coord_proc;
  const bool csp = coord_proc.spawn(PFABRIC_EXE, "serve --port " + std::to_string(port), true);
  CHECK(csp);
  if (!csp) { net_cleanup(); return; }
  { std::string buf; const std::string line = read_line(buf, coord_proc); CHECK(line.find("PREFILL_FABRIC_PORT") != std::string::npos); }

  pf_test::Proc w1, w2;
  const bool s1 = w1.spawn(PFABRIC_EXE, "worker --coord " + coord + " --id 1 --backend cpu", false);
  const bool s2 = w2.spawn(PFABRIC_EXE, "worker --coord " + coord + " --id 2 --backend cpu", false);
  CHECK(s1); CHECK(s2);

  Client c(host, port);
  CHECK(c.connect().is_ok());
  CHECK(wait_workers2(c));

  auto st0 = c.query_state();
  CHECK(st0.is_ok());
  const Epoch epoch0 = st0.value().epoch;
  WorkerBootId boot1_0;
  for (const auto& w : st0.value().workers) if (w.worker_id == WorkerId(1)) boot1_0 = w.boot_id;
  CHECK(!boot1_0.is_nil());

  const RequestId r_short(1), r_med(2), r_long(3), r_t4(4), r_t5(5);
  CHECK(c.submit(req_of(r_short, TenantId(10), 100), toks(100, 1)).is_ok());
  CHECK(c.submit(req_of(r_med, TenantId(10), 2000), toks(2000, 2)).is_ok());
  CHECK(c.submit(req_of(r_long, TenantId(20), 20000), toks(20000, 3)).is_ok());
  CHECK(c.submit(req_of(r_t4, TenantId(30), 3000, 5), toks(3000, 4)).is_ok());
  CHECK(c.submit(req_of(r_t5, TenantId(30), 1200, 0, 4), toks(1200, 5)).is_ok());

  int progress = 0;
  for (int i = 0; i < 3000 && progress < 2; ++i) {
    auto ex = c.query_explain(r_short);
    auto exl = c.query_explain(r_long);
    int done = 0, started = 0;
    if (ex && exl) {
      if (ex.value().lifecycle == Lifecycle::completed) done = 1;
      if (exl.value().lifecycle != Lifecycle::queued && exl.value().lifecycle != Lifecycle::submitted) started = 1;
    }
    if (done && started) { progress = 2; break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  CHECK_EQ(progress, 2);
  auto stA = c.query_stats();
  CHECK(stA.is_ok());
  CHECK_GE(stA.value().completed, 1u);

  auto exl = c.query_explain(r_long);
  CHECK(exl.is_ok());
  const AttemptId long_attempt0 = exl.value().attempt_id;
  const Generation long_gen0 = exl.value().generation;
  { std::ofstream f("stale_artifact.pfstate", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&epoch0), sizeof(epoch0));
    const std::uint64_t b1 = boot1_0.value();
    f.write(reinterpret_cast<const char*>(&b1), sizeof(b1));
    const std::uint64_t b2 = long_attempt0.value();
    f.write(reinterpret_cast<const char*>(&b2), sizeof(b2));
    const std::uint64_t b3 = long_gen0.value();
    f.write(reinterpret_cast<const char*>(&b3), sizeof(b3)); }

  w1.kill();
  bool lost_seen = false;
  for (int i = 0; i < 600; ++i) {
    auto s = c.query_state();
    if (s) for (const auto& w : s.value().workers) if (w.worker_id == WorkerId(1) && w.state == WorkerState::lost) { lost_seen = true; break; }
    if (lost_seen) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(lost_seen);

  pf_test::Proc w1b;
  CHECK(w1b.spawn(PFABRIC_EXE, "worker --coord " + coord + " --id 1 --backend cpu", false));
  WorkerBootId boot1_new;
  for (int i = 0; i < 600; ++i) {
    auto s = c.query_state();
    if (s) for (const auto& w : s.value().workers) if (w.worker_id == WorkerId(1) && w.state == WorkerState::ready) { boot1_new = w.boot_id; break; }
    if (!boot1_new.is_nil() && boot1_new != boot1_0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  CHECK(!boot1_new.is_nil());
  CHECK(boot1_new != boot1_0);

  auto rolled = c.roll_epoch();
  CHECK(rolled.is_ok());
  const Epoch epoch1 = rolled.value();
  CHECK(epoch1 != epoch0);

  auto exl2 = c.query_explain(r_long);
  CHECK(exl2.is_ok());
  const AttemptId long_attempt_now = exl2.value().attempt_id;
  const Generation long_gen_now = exl2.value().generation;

  { Message comp; comp.kind = MsgKind::completion;
    MemberResult mr; mr.request_id = r_long; mr.attempt_id = long_attempt_now; mr.generation = long_gen_now;
    mr.epoch = epoch0; mr.worker = WorkerId(1); mr.boot = boot1_0; mr.outcome = MemberOutcome::success;
    mr.tokens_completed = 100; comp.member_results.push_back(mr);
    CHECK_EQ(send_completion(host, port, comp), ErrorCode::stale_epoch); }

  { Message comp; comp.kind = MsgKind::completion;
    MemberResult mr; mr.request_id = r_long; mr.attempt_id = long_attempt_now; mr.generation = long_gen_now;
    mr.epoch = epoch1; mr.worker = WorkerId(1); mr.boot = boot1_0; mr.outcome = MemberOutcome::success;
    mr.tokens_completed = 100; comp.member_results.push_back(mr);
    CHECK_EQ(send_completion(host, port, comp), ErrorCode::stale_worker); }

  { Message comp; comp.kind = MsgKind::completion;
    MemberResult mr; mr.request_id = r_long; mr.attempt_id = AttemptId(long_attempt_now.value() + 999999); mr.generation = long_gen_now;
    mr.epoch = epoch1; mr.worker = WorkerId(1); mr.boot = boot1_new; mr.outcome = MemberOutcome::success;
    mr.tokens_completed = 100; comp.member_results.push_back(mr);
    CHECK_EQ(send_completion(host, port, comp), ErrorCode::stale_attempt); }

  { Message comp; comp.kind = MsgKind::completion;
    MemberResult mr; mr.request_id = r_long; mr.attempt_id = long_attempt_now; mr.generation = Generation(long_gen_now.value() + 999999);
    mr.epoch = epoch1; mr.worker = WorkerId(1); mr.boot = boot1_new; mr.outcome = MemberOutcome::success;
    mr.tokens_completed = 100; comp.member_results.push_back(mr);
    CHECK_EQ(send_completion(host, port, comp), ErrorCode::stale_generation); }

  const RequestId r_fresh(1000);
  CHECK(c.submit(req_of(r_fresh, TenantId(40), 1500), toks(1500, 1000)).is_ok());
  for (int i = 0; i < 3000; ++i) {
    auto ex = c.query_explain(r_fresh);
    if (ex && ex.value().lifecycle == Lifecycle::completed) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  auto exf = c.query_explain(r_fresh);
  CHECK(exf.is_ok());
  CHECK(exf.value().lifecycle == Lifecycle::completed);

  for (int i = 0; i < 4000; ++i) {
    bool all_term = true;
    const RequestId ids[] = { r_short, r_med, r_long, r_t4, r_t5, r_fresh };
    for (const auto rid : ids) {
      auto ex = c.query_explain(rid);
      if (!ex) { all_term = false; break; }
      const auto lc = ex.value().lifecycle;
      if (lc != Lifecycle::completed && lc != Lifecycle::cancelled && lc != Lifecycle::expired && lc != Lifecycle::failed_non_retryable) { all_term = false; break; }
    }
    if (all_term) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  auto stF = c.query_stats();
  CHECK(stF.is_ok());
  CHECK_EQ(stF.value().running_requests, 0u);
  CHECK_EQ(stF.value().queue_depth, 0u);
  CHECK_GE(stF.value().completed, 5u);

  coord_proc.kill();
  w1b.kill();
  w2.kill();
  net_cleanup();
}

