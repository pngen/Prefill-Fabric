// Example: CUDA prefill if a CUDA device is available.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cuda_executor.hpp>
#include <cstdio>
using namespace prefillfabric;
int main() {
  auto cuda_ex = create_cuda_executor();
  if (!cuda_ex || !cuda_ex->available()) {
    std::printf("CUDA unavailable on this machine; example did not run on GPU.\n");
    return 0;
  }
  SchedulerConfig cfg; cfg.device_memory_bytes = 12ULL << 30; cfg.max_tokens_per_chunk = 512;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(cuda_ex, kBackendCuda);
  PrefillRequest r; r.request_id = RequestId(1); r.tenant_id = TenantId(1);
  r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 4000; r.backend = kBackendCuda;
  r.compute_estimate = 4000; r.memory_estimate_bytes = 4000*16; r.input_shape.seq_len = 4000;
  sch.submit_with_tokens(r, std::vector<std::uint32_t>(4000, 3));
  sch.drive_until_quiescent();
  std::printf("completed=%d tokens=%llu\n", sch.is_terminal(RequestId(1)) ? 1 : 0, (unsigned long long)sch.stats().completed_tokens);
  return sch.is_terminal(RequestId(1)) ? 0 : 1;
}