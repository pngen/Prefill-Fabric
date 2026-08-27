// Example: weighted tenants make forward progress.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 2000;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  for (std::uint64_t i = 0; i < 8; ++i) {
    PrefillRequest r; r.request_id = RequestId(1 + i); r.tenant_id = TenantId(1);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 4000; r.tenant_weight = 4.0;
    r.compute_estimate = 4000; r.memory_estimate_bytes = 4000*16; r.input_shape.seq_len = 4000;
    sch.submit_with_tokens(r, std::vector<std::uint32_t>(4000, (std::uint32_t)i));
  }
  for (std::uint64_t i = 0; i < 8; ++i) {
    PrefillRequest r; r.request_id = RequestId(100 + i); r.tenant_id = TenantId(2);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 300; r.tenant_weight = 1.0;
    r.compute_estimate = 300; r.memory_estimate_bytes = 300*16; r.input_shape.seq_len = 300;
    sch.submit_with_tokens(r, std::vector<std::uint32_t>(300, (std::uint32_t)i));
  }
  sch.drive_until_quiescent();
  std::printf("completed=%llu\n", (unsigned long long)sch.stats().completed);
  return sch.stats().completed == 16 ? 0 : 1;
}