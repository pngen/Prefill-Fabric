// Example: distributed coordinator + worker over the framed TCP control plane.
#include <prefillfabric/coordinator.hpp>
#include <prefillfabric/worker.hpp>
#include <prefillfabric/client.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
#include <thread>
#include <chrono>
using namespace prefillfabric;
int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffer so output survives process exit.
  net_init();
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  Coordinator coord(cfg, std::make_shared<SteadyClock>(), 0);
  if (!coord.start().is_ok()) return 1;
  const std::uint16_t port = coord.port();
  std::thread wt([port]() { Worker w(WorkerId(1), "127.0.0.1", port, std::make_shared<CpuExecutor>(), kBackendCpu); w.run(); });
  wt.detach();   // run the worker independently for the duration of this demo.
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  Client c("127.0.0.1", port);
  if (!c.connect().is_ok()) return 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
  r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 2000;
  r.compute_estimate = 2000; r.memory_estimate_bytes = 2000*16; r.input_shape.seq_len = 2000;
  std::vector<std::uint32_t> tokens(2000, 5);
  auto sub = c.submit(r, tokens);
  bool done = false;
  for (int i = 0; i < 3000; ++i) { auto ex = c.query_explain(RequestId(1)); if (ex && ex.value().lifecycle == Lifecycle::completed) { done = true; break; } std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
  std::printf("distributed submit_ok=%d completed=%d\n", sub.is_ok() ? 1 : 0, done ? 1 : 0);
  coord.shutdown();
  net_cleanup();
  return done ? 0 : 1;
}