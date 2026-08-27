// External downstream consumer of installed PrefillFabric.
#include <prefillfabric/scheduler.hpp>
#include <prefillfabric/exec/cpu_executor.hpp>
#include <prefillfabric/compatibility.hpp>
#include <prefillfabric/explain.hpp>
#include <prefillfabric/persistence.hpp>
#include <cstdio>
using namespace prefillfabric;

static std::vector<std::uint32_t> ctok(std::uint64_t n, std::uint64_t seed) {
  std::vector<std::uint32_t> v(n); std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ seed;
  for (std::size_t i = 0; i < v.size(); ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[i] = static_cast<std::uint32_t>(s & 0xFFFFFFFFu); }
  return v;
}

int main() {
  SchedulerConfig cfg; cfg.device_memory_bytes = 8ULL << 30; cfg.max_tokens_per_chunk = 1024;
  PrefillScheduler sch(cfg, std::make_shared<SteadyClock>());
  sch.attach_executor(std::make_shared<CpuExecutor>(), kBackendCpu);
  // Two compatible requests that pack together.
  for (std::uint64_t i = 1; i <= 2; ++i) {
    PrefillRequest r; r.request_id = RequestId(i); r.tenant_id = TenantId(1);
    r.model.name = "m"; r.model.revision = "1"; r.prompt_token_count = 1500;
    r.compute_estimate = 1500; r.memory_estimate_bytes = 1500*16; r.input_shape.seq_len = 1500;
    sch.submit_with_tokens(r, ctok(1500, i));
  }
  auto d = sch.drive_until_quiescent();
  const bool completed = sch.is_terminal(RequestId(1)) && sch.is_terminal(RequestId(2));
  const auto st = sch.stats();
  auto ex1 = sch.explain(RequestId(1));
  // Persist + recover round-trip.
  sch.persist("consumer.pfstate");
  PrefillScheduler sch2(cfg, std::make_shared<SteadyClock>());
  const bool rec = sch2.recover("consumer.pfstate").is_ok();
  std::remove("consumer.pfstate");
  const bool noleak = sch.memory().unreleased_count() == 0;
  std::printf("consumer completed=%d ok=%d groups=%llu explained=%zu recovered=%d noleak=%d\n",
              completed ? 1 : 0, d.is_ok() ? 1 : 0, (unsigned long long)st.groups_formed,
              ex1 ? ex1.value().trace_size() : 0, rec ? 1 : 0, noleak ? 1 : 0);
  return (completed && d.is_ok() && st.completed == 2 && ex1 && ex1.value().trace_size() > 0 && rec && noleak) ? 0 : 1;
}