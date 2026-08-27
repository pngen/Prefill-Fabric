// Example: chunked long prompt completes.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 1000;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
  r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 6000;
  r.compute_estimate = 6000; r.memory_estimate_bytes = 6000*16; r.input_shape.seq_len = 6000;
  sch.submit_with_tokens(r, std::vector<std::uint32_t>(6000, 1));
  sch.drive_until_quiescent();
  std::printf("completed=%d\n", sch.is_terminal(RequestId(1)) ? 1 : 0);
  return sch.is_terminal(RequestId(1)) ? 0 : 1;
}