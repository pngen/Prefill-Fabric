// Prefill Fabric - CLI.
// Copyright (c) 2026 Summon Software Labs. Apache-2.0.
#include "prefillfabric/scheduler.hpp"
#include "prefillfabric/coordinator.hpp"
#include "prefillfabric/worker.hpp"
#include "prefillfabric/client.hpp"
#include "prefillfabric/exec/cpu_executor.hpp"
#include "prefillfabric/exec/cuda_executor.hpp"
#include "prefillfabric/chunking.hpp"
#include "prefillfabric/explain.hpp"
#include "prefillfabric/network.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
#include <map>

using namespace prefillfabric;

static int usage() {
  std::printf("Prefill Fabric CLI\n");
  std::printf("  serve    --port N                  run coordinator\n");
  std::printf("  worker   --coord host:port --id N --backend cpu|cuda\n");
  std::printf("  submit   --coord host:port --prompt N --tenant T [--cached C] [--priority P] [--backend B] [--model M] [--max-retries R]\n");
  std::printf("  cancel   --coord host:port --request ID\n");
  std::printf("  status   --coord host:port\n");
  std::printf("  workers  --coord host:port\n");
  std::printf("  stats    --coord host:port\n");
  std::printf("  snapshot --coord host:port\n");
  std::printf("  explain  --coord host:port --request ID\n");
  std::printf("  recover  --file state.pfstate\n");
  std::printf("  benchmark --count N --prompt P [--threads T]\n");
  return 1;
}

static std::string arg(const std::vector<std::string>& args, const std::string& name,
                       const std::string& def = "") {
  for (std::size_t i = 0; i + 1 < args.size(); ++i)
    if (args[i] == name) return args[i + 1];
  return def;
}

// split "host:port".
static void parse_hostport(const std::string& s, std::string& host, std::uint16_t& port) {
  const auto pos = s.rfind(':');
  if (pos == std::string::npos) { host = s; port = 0; return; }
  host = s.substr(0, pos);
  port = static_cast<std::uint16_t>(std::atoi(s.c_str() + pos + 1));
}

static int cmd_serve(const std::vector<std::string>& args) {
  const std::uint16_t port = static_cast<std::uint16_t>(std::atoi(arg(args, "--port", "0").c_str()));
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 8ULL << 30;
  auto clock = std::make_shared<SteadyClock>();
  Coordinator coord(cfg, clock, port);
  auto r = coord.start();
  if (!r) { std::cerr << "coordinator start failed: " << r.error().message << "\n"; return 1; }
  std::printf("PREFILL_FABRIC_PORT %u\n", coord.port());
  std::fflush(stdout);
  // Run until interrupted.
  while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
}

static int cmd_worker(const std::vector<std::string>& args) {
  std::string host; std::uint16_t port;
  parse_hostport(arg(args, "--coord", "127.0.0.1:0"), host, port);
  const WorkerId id(std::strtoull(arg(args, "--id", "1").c_str(), nullptr, 10));
  const std::string backend = arg(args, "--backend", "cpu");
  std::shared_ptr<Executor> ex;
  if (backend == "cuda") {
    ex = create_cuda_executor();
    if (!ex || !ex->available()) { std::cerr << "cuda backend unavailable in this build; use cpu\n"; return 1; }
  } else {
    ex = std::make_shared<CpuExecutor>();
  }
  Worker w(id, host, port, ex, backend);
  auto r = w.run();
  if (!r) { std::cerr << "worker error: " << r.error().message << "\n"; return 1; }
  return 0;
}
static PrefillRequest make_request(const std::vector<std::string>& args) {
  PrefillRequest r;
  const std::uint64_t now = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  r.request_id = RequestId(now & 0xFFFFFFFFFFFFULL);
  r.tenant_id = TenantId(std::strtoull(arg(args, "--tenant", "1").c_str(), nullptr, 10));
  r.model.name = arg(args, "--model", "model");
  r.model.revision = arg(args, "--revision", "1");
  r.prompt_token_count = std::strtoull(arg(args, "--prompt", "1024").c_str(), nullptr, 10);
  const std::uint64_t cached = std::strtoull(arg(args, "--cached", "0").c_str(), nullptr, 10);
  r.reusable_prefix.token_count = cached;
  r.reusable_prefix.validated = cached > 0;
  r.priority = std::atoi(arg(args, "--priority", "0").c_str());
  r.backend = arg(args, "--backend", kBackendAny);
  r.max_retries = std::atoi(arg(args, "--max-retries", "2").c_str());
  r.tenant_weight = 1.0;
  r.compute_estimate = r.prompt_token_count * 4;
  r.memory_estimate_bytes = r.prompt_token_count * 16;
  r.input_shape.seq_len = r.prompt_token_count;
  r.input_shape.hidden_dim = 512;
  r.input_shape.num_heads = 8;
  r.input_shape.head_dim = 64;
  return r;
}

static std::vector<std::uint32_t> gen_tokens(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n);
  std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu);
  }
  return v;
}

static int cmd_submit(const std::vector<std::string>& args) {
  std::string host; std::uint16_t port;
  parse_hostport(arg(args, "--coord", "127.0.0.1:0"), host, port);
  Client c(host, port);
  auto cn = c.connect();
  if (!cn) { std::cerr << "connect failed: " << cn.error().message << "\n"; return 1; }
  auto req = make_request(args);
  auto res = c.submit(req, gen_tokens(req.prompt_token_count, req.request_id.value()));
  if (!res) { std::cerr << "submit rejected: " << res.error().message << " (code=" << static_cast<int>(res.error().code) << ")\n"; return 1; }
  std::printf("accepted request=%llu\n", static_cast<unsigned long long>(res.value().value()));
  return 0;
}

static int cmd_cancel(const std::vector<std::string>& args) {
  std::string host; std::uint16_t port;
  parse_hostport(arg(args, "--coord", "127.0.0.1:0"), host, port);
  Client c(host, port);
  if (!c.connect()) return 1;
  const RequestId req(std::strtoull(arg(args, "--request", "0").c_str(), nullptr, 10));
  auto cr = c.cancel(req);
  if (!cr) { std::cerr << "cancel failed: " << cr.error().message << "\n"; return 1; }
  std::printf("cancel applied=%d terminal=%d code=%s\n", cr.value().applied ? 1 : 0,
              cr.value().was_terminal ? 1 : 0, to_string(cr.value().code));
  return 0;
}

static int cmd_state(const std::vector<std::string>& args, const std::string& verb) {
  std::string host; std::uint16_t port;
  parse_hostport(arg(args, "--coord", "127.0.0.1:0"), host, port);
  Client c(host, port);
  if (!c.connect()) { std::cerr << "connect failed\n"; return 1; }
  if (verb == "stats") {
    auto st = c.query_stats();
    if (!st) { std::cerr << "stats failed: " << st.error().message << "\n"; return 1; }
    const auto& s = st.value();
    std::printf("submitted=%llu admitted=%llu completed=%llu cancelled=%llu retried=%llu failed=%llu expired=%llu\n",
                (unsigned long long)s.submitted, (unsigned long long)s.admitted, (unsigned long long)s.completed,
                (unsigned long long)s.cancelled, (unsigned long long)s.retried, (unsigned long long)s.failed_non_retryable,
                (unsigned long long)s.expired);
    std::printf("groups=%llu scheduled_tokens=%llu completed_tokens=%llu\n",
                (unsigned long long)s.groups_formed, (unsigned long long)s.scheduled_tokens,
                (unsigned long long)s.completed_tokens);
    std::printf("stale_rejected=%llu duplicate_rejected=%llu queue_depth=%llu waiting_tokens=%llu running=%llu\n",
                (unsigned long long)s.stale_authority_rejected, (unsigned long long)s.duplicate_completion_rejected,
                (unsigned long long)s.queue_depth, (unsigned long long)s.waiting_tokens, (unsigned long long)s.running_requests);
    return 0;
  }
  auto st = c.query_state();
  if (!st) { std::cerr << "state failed: " << st.error().message << "\n"; return 1; }
  std::printf("epoch=%llu workers=%zu\n", (unsigned long long)st.value().epoch.value(), st.value().workers.size());
  for (const auto& w : st.value().workers)
    std::printf("  worker id=%llu boot=%llu backend=%s state=%s\n",
                (unsigned long long)w.worker_id.value(), (unsigned long long)w.boot_id.value(),
                w.backend.c_str(), to_string(w.state));
  return 0;
}

static int cmd_explain(const std::vector<std::string>& args) {
  std::string host; std::uint16_t port;
  parse_hostport(arg(args, "--coord", "127.0.0.1:0"), host, port);
  Client c(host, port);
  if (!c.connect()) return 1;
  const RequestId req(std::strtoull(arg(args, "--request", "0").c_str(), nullptr, 10));
  auto ex = c.query_explain(req);
  if (!ex) { std::cerr << "explain failed: " << ex.error().message << "\n"; return 1; }
  const auto& m = ex.value();
  std::printf("request=%llu attempt=%llu lifecycle=%s\n",
              (unsigned long long)m.request_id.value(), (unsigned long long)m.attempt_id.value(), to_string(m.lifecycle));
  for (const auto& t : m.explain.trace) std::printf("  - %s\n", t.c_str());
  return 0;
}

static int cmd_recover(const std::vector<std::string>& args) {
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 8ULL << 30;
  auto clock = std::make_shared<SteadyClock>();
  PrefillScheduler sch(cfg, clock);
  auto r = sch.recover(arg(args, "--file"));
  if (!r) { std::cerr << "recover failed: " << r.error().message << " (" << static_cast<int>(r.error().code) << ")\n"; return 1; }
  std::printf("recovered requests=%zu epoch=%llu\n", sch.request_count(), (unsigned long long)sch.current_epoch().value());
  return 0;
}

static int cmd_benchmark(const std::vector<std::string>& args) {
  const std::uint64_t count = std::strtoull(arg(args, "--count", "1000").c_str(), nullptr, 10);
  const std::uint64_t prompt = std::strtoull(arg(args, "--prompt", "1024").c_str(), nullptr, 10);
  SchedulerConfig cfg;
  cfg.device_memory_bytes = 8ULL << 30;
  cfg.max_tokens_per_chunk = 512;
  auto clock = std::make_shared<SteadyClock>();
  PrefillScheduler sch(cfg, clock);
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  const auto t0 = clock->now();
  for (std::uint64_t i = 0; i < count; ++i) {
    PrefillRequest r;
    r.request_id = RequestId(i + 1);
    r.tenant_id = TenantId(1 + (i % 4));
    r.model.name = "m"; r.model.revision = "1";
    r.prompt_token_count = prompt;
    r.priority = static_cast<int>(i % 8);
    r.tenant_weight = 1.0;
    r.compute_estimate = prompt * 4;
    r.memory_estimate_bytes = prompt * 16;
    r.input_shape.seq_len = prompt;
    sch.submit_with_tokens(r, gen_tokens(prompt, i + 1));
  }
  auto d = sch.drive_until_quiescent();
  if (!d) { std::cerr << "drive failed: " << d.error().message << "\n"; return 1; }
  const auto t1 = clock->now();
  const auto st = sch.stats();
  const double secs = static_cast<double>(t1 - t0) / 1e9;
  std::printf("requests=%llu completed=%llu tokens=%llu elapsed=%.3fs rate=%.0f req/s groups=%llu\n",
              (unsigned long long)count, (unsigned long long)st.completed, (unsigned long long)st.completed_tokens,
              secs, count / (secs > 0 ? secs : 1e-9), (unsigned long long)st.groups_formed);
  return 0;
}

int main(int argc, char** argv) {
  net_init();
  if (argc < 2) return usage();
  std::vector<std::string> args(argv + 1, argv + argc);
  const std::string verb = args[0];
  if (verb == "serve") return cmd_serve(args);
  if (verb == "worker") return cmd_worker(args);
  if (verb == "submit") return cmd_submit(args);
  if (verb == "cancel") return cmd_cancel(args);
  if (verb == "stats") return cmd_state(args, "stats");
  if (verb == "status" || verb == "workers" || verb == "waiting" || verb == "running") return cmd_state(args, verb);
  if (verb == "snapshot") return cmd_state(args, "snapshot");
  if (verb == "roll") { std::string host; std::uint16_t port; parse_hostport(arg(args, "--coord", "127.0.0.1:0"), host, port); Client c(host, port); if (!c.connect()) { std::cerr << "connect failed\n"; return 1; } auto e = c.roll_epoch(); if (!e) { std::cerr << "roll failed: " << e.error().message << "\n"; return 1; } std::printf("epoch=%llu\n", (unsigned long long)e.value().value()); return 0; }
  if (verb == "explain" || verb == "inspect") return cmd_explain(args);
  if (verb == "recover") return cmd_recover(args);
  if (verb == "benchmark") return cmd_benchmark(args);
  return usage();
}

