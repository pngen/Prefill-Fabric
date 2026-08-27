// Example: persist authoritative state and recover it.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <cstdio>
#include <cstdio>
using namespace prefillfabric;
int main() {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30;
  {
    PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
    sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
    PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 1500;
    r.compute_estimate = 1500; r.memory_estimate_bytes = 1500*16; r.input_shape.seq_len = 1500;
    sch.submit_with_tokens(r, std::vector<std::uint32_t>(1500, 9));
    sch.drive_until_quiescent();
    sch.persist("example.pfstate");
  }
  PrefillScheduler sch2(cfg, std::make_shared<SteadyClock>());
  auto rec = sch2.recover("example.pfstate");
  std::printf("recovered=%d requests=%zu\n", rec.is_ok() ? 1 : 0, sch2.request_count());
  std::remove("example.pfstate");
  return rec.is_ok() ? 0 : 1;
}