// Example: retry / per-member outcomes.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 1024;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  for (std::uint64_t i = 0; i < 4; ++i) {
    PrefillRequest r; r.request_id = RequestId(1 + i); r.tenant_id = TenantId(1);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 2048; r.max_retries = 2;
    r.compute_estimate = 2048; r.memory_estimate_bytes = 2048*16; r.input_shape.seq_len = 2048;
    sch.submit_with_tokens(r, std::vector<std::uint32_t>(2048, (std::uint32_t)i));
  }
  sch.drive_until_quiescent();
  std::printf("completed=%llu leaks=%llu\n", (unsigned long long)sch.stats().completed, (unsigned long long)sch.memory().unreleased_count());
  return sch.memory().unreleased_count() == 0 ? 0 : 1;
}