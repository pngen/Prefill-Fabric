// Example: compatible requests pack into groups.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 2000; cfg.max_requests_per_group = 8;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  for (std::uint64_t i = 0; i < 8; ++i) {
    PrefillRequest r; r.request_id = RequestId(1 + i); r.tenant_id = TenantId(1);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 2000;
    r.compute_estimate = 2000; r.memory_estimate_bytes = 2000*16; r.input_shape.seq_len = 2000;
    sch.submit_with_tokens(r, std::vector<std::uint32_t>(2000, (std::uint32_t)i));
  }
  sch.drive_until_quiescent();
  std::printf("groups=%llu completed=%llu\n", (unsigned long long)sch.stats().groups_formed, (unsigned long long)sch.stats().completed);
  return sch.stats().completed == 8 ? 0 : 1;
}