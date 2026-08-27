// Example: mixed prompt lengths all complete.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 512;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  for (std::uint64_t i = 1; i <= 6; ++i) {
    PrefillRequest r; r.request_id = RequestId(i); r.tenant_id = TenantId(1);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = i * 1000;
    r.compute_estimate = r.prompt_token_count; r.memory_estimate_bytes = r.prompt_token_count*16;
    r.input_shape.seq_len = r.prompt_token_count;
    sch.submit_with_tokens(r, std::vector<std::uint32_t>(r.prompt_token_count, (std::uint32_t)i));
  }
  sch.drive_until_quiescent();
  bool all = true; for (std::uint64_t i = 1; i <= 6; ++i) all = all && sch.is_terminal(RequestId(i));
  std::printf("all completed=%d tokens=%llu\n", all ? 1 : 0, (unsigned long long)sch.stats().completed_tokens);
  return all ? 0 : 1;
}