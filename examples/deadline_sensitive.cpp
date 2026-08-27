// Example: deadline-sensitive execution.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
#include <chrono>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 1000;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
  r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 3000;
  r.deadline = Nanoseconds(std::chrono::steady_clock::now().time_since_epoch().count() + 100000000);
  r.compute_estimate = 3000; r.memory_estimate_bytes = 3000*16; r.input_shape.seq_len = 3000;
  sch.submit_with_tokens(r, std::vector<std::uint32_t>(3000, 5));
  sch.drive_until_quiescent();
  std::printf("completed=%d\n", sch.is_terminal(RequestId(1)) ? 1 : 0);
  return sch.is_terminal(RequestId(1)) ? 0 : 1;
}