// Example: reusable-prefix remainder is scheduled, not the full prompt.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 1000;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
  r.model.name = "m"; r.model.revision = "1";
  r.prompt_token_count = 32000; r.reusable_prefix.token_count = 28000; r.reusable_prefix.validated = true;
  r.compute_estimate = 32000; r.memory_estimate_bytes = 32000*16; r.input_shape.seq_len = 32000;
  sch.submit_with_tokens(r, std::vector<std::uint32_t>(32000, 3));
  sch.drive_until_quiescent();
  std::printf("completed_tokens=%llu\n", (unsigned long long)sch.stats().completed_tokens);
  return sch.stats().completed_tokens == 4000 ? 0 : 1;
}