// Example: cancellation of waiting and in-flight work.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.max_tokens_per_chunk = 500;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  PrefillRequest a; a.request_id = RequestId(1); a.tenant_id = TenantId(1);
  a.model.name = "m"; a.model.revision = "1"; a.prompt_token_count = 20000;
  a.compute_estimate = 20000; a.memory_estimate_bytes = 20000*16; a.input_shape.seq_len = 20000;
  sch.submit_with_tokens(a, std::vector<std::uint32_t>(20000, 1));
  PrefillRequest b; b.request_id = RequestId(2); b.tenant_id = TenantId(1);
  b.model.name = "m"; b.model.revision = "1"; b.prompt_token_count = 200;
  b.compute_estimate = 200; b.memory_estimate_bytes = 200*16; b.input_shape.seq_len = 200;
  sch.submit_with_tokens(b, std::vector<std::uint32_t>(200, 2));
  auto cr = sch.cancel(RequestId(2));
  sch.drive_until_quiescent();
  std::printf("b_terminal=%d cancelled=%llu\n", sch.is_terminal(RequestId(2)) ? 1 : 0, (unsigned long long)sch.stats().cancelled);
  return sch.memory().unreleased_count() == 0 ? 0 : 1;
}